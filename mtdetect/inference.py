
import sys
import logging

import mtdetect.dataset as dataset
import mtdetect.utils.utils as utils

import torch
import sklearn.metrics
from sklearn.utils.multiclass import unique_labels
import numpy as np

logger = logging.getLogger("mtdetect")

def inference(model, inputs_and_outputs, loss_function=None, device=None, threshold=0.5):
    # Inputs and outputs
    urls = inputs_and_outputs["url_tokens"]
    attention_mask = inputs_and_outputs["url_attention_mask"]
    labels = inputs_and_outputs["labels"]
    model_outputs = None

    # Move to device
    urls = urls.to(device)
    attention_mask = attention_mask.to(device)

    if loss_function:
        # Move to device
        labels = labels.to(device)

    # Inference
    model_outputs = model(urls, attention_mask)
    outputs = model_outputs.logits
    bs = outputs.shape[0]

    assert outputs.shape == (bs, 1), outputs.shape

    # Calcule loss
    outputs = outputs.squeeze(1) # (batch_size, 1) -> (batch_size,)
    outputs_classification = torch.sigmoid(outputs).cpu().detach().tolist()
    outputs_classification = list(map(lambda n: int(n >= threshold), outputs_classification))

    if loss_function:
        loss = loss_function(outputs, labels)

    results = {
        "outputs": outputs,
        "outputs_classification_detach_list": outputs_classification,
        "loss": loss if loss_function else None,
    }

    return results

@torch.no_grad()
def inference_eval(model, tokenizer, _dataset, loss_function=None, device=None, threshold=0.5, monolingual=None):
    training = model.training

    model.eval()

    all_loss = []
    all_outputs = []
    all_labels = []
    total_tokens = 0
    dataloader = _dataset.dataloader

    for batch in dataloader:
        total_tokens += sum([len(urls[urls != tokenizer.pad_token_id]) for urls in batch["url_tokens"]])

        # Inference
        results = inference(model, batch, loss_function=loss_function, device=device, threshold=threshold)
        outputs_classification = results["outputs_classification_detach_list"]
        labels = batch["labels"].cpu()
        labels = torch.round(labels).type(torch.long)
        loss = results["loss"].cpu().detach().item()

        all_outputs.extend(outputs_classification)
        all_labels.extend(labels.tolist())
        all_loss.append(loss)

    assert total_tokens == _dataset.total_tokens, f"{total_tokens} != {_dataset.total_tokens}"

    all_outputs = torch.as_tensor(all_outputs)
    all_labels = torch.as_tensor(all_labels)
    all_loss = torch.as_tensor(all_loss)
    metrics = get_metrics(all_outputs, all_labels)
    metrics["loss_average"] = all_loss.mean().item()

    if training:
        # Restore model status
        model.train()

    return metrics

@torch.no_grad()
def inference_from_stdin(model, tokenizer, batch_size, loss_function=None, device=None, max_length_tokens=None, threshold=0.5, monolingual=None, dataset_workers=-1):
    training = model.training

    model.eval()

    all_loss = []
    all_outputs = []
    all_labels = []
    total_tokens = 0
    do_eval = False
    columns = None
    input_data = []
    output_data = []
    max_length_tokens = max_length_tokens if max_length_tokens is not None else tokenizer.model_max_length

    for l in sys.stdin:
        s = l.rstrip("\r\n").split('\t')

        if columns is None:
            columns = len(s)
            do_eval = (len(s) == 3) or (monolingual and (len(s) == 2)) # True if labels are provided

            if not do_eval:
                loss_function = None

        assert columns == len(s), f"{columns} != {len(s)}: {s}"
        assert len(s) == ((1 + int(do_eval)) if monolingual else (2 + int(do_eval))), s

        label = 0 # fake

        if do_eval:
            label = s[1 if monolingual else 2]

            assert label in ('0', '1'), f"{label} :from: {s}"

            label = float(label)

        s = tokenizer.sep_token.join(s[:1 if monolingual else 2])

        input_data.append(s)
        output_data.append(label) # might be a fake value

    _dataset = dataset.SmartBatchingURLsDataset(input_data, output_data, tokenizer, max_length_tokens,
                                                set_desc="stdin_inference")
    dataloader = _dataset.get_dataloader(batch_size, device, dataset_workers)

    for batch in dataloader:
        total_tokens += sum([len(urls[urls != tokenizer.pad_token_id]) for urls in batch["url_tokens"]])

        # Inference
        results = inference(model, batch, loss_function=loss_function, device=device, threshold=threshold)

        # Results
        if not do_eval:
            output = torch.sigmoid(results["outputs"]).cpu().detach().item()

            logger.info("%s\t%s", s, output)

            continue

        outputs_classification = results["outputs_classification_detach_list"]
        labels = batch["labels"].cpu()
        labels = torch.round(labels).type(torch.long)
        loss = results["loss"].cpu().detach().item()

        all_outputs.extend(outputs_classification)
        all_labels.extend(labels.tolist())
        all_loss.append(loss)

    assert total_tokens == _dataset.total_tokens, f"{total_tokens} != {_dataset.total_tokens}"

    if not do_eval:
        return None

    all_outputs = torch.as_tensor(all_outputs)
    all_labels = torch.as_tensor(all_labels)
    all_loss = torch.as_tensor(all_loss)
    metrics = get_metrics(all_outputs, all_labels)
    metrics["loss_average"] = all_loss.mean().item()

    if training:
        # Restore model status
        model.train()

    return metrics

def get_confusion_matrix(outputs_argmax, labels):
    assert len(unique_labels(labels.flatten())) <= 2

    classes = 2
    tp, fp, fn, tn = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)
    conf_mat = np.array([[torch.sum(torch.logical_and(outputs_argmax == c1, labels == c2)) for c1 in range(classes)] for c2 in range(classes)])

    # Sanity check
    sklearn_conf_mat = sklearn.metrics.confusion_matrix(labels, outputs_argmax, labels=list(range(classes)))

    assert not (conf_mat != sklearn_conf_mat).any()

    for c in range(classes):
        # Multiclass confusion matrix
        # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
        tp[c] = int(torch.sum(torch.logical_and(labels == c, outputs_argmax == c)))
        fp[c] = int(torch.sum(torch.logical_and(labels != c, outputs_argmax == c)))
        fn[c] = int(torch.sum(torch.logical_and(labels == c, outputs_argmax != c)))
        tn[c] = int(torch.sum(torch.logical_and(labels != c, outputs_argmax != c)))

    # Sanity check
    for c in range(classes):
        idxs = np.arange(classes)

        assert tp[c] == conf_mat[c][c], f"class {c} -> TP != confusion matrix"
        assert fp[c] == np.sum(conf_mat[idxs != c][:,c]), f"class {c} -> FP != confusion matrix"
        assert fn[c] == np.sum(conf_mat[c][idxs != c]), f"class {c} -> FN != confusion matrix"
        assert tn[c] == np.sum([conf_mat[c1][c2] for c1 in range(classes) for c2 in range(classes) if c1 != c and c2 != c]), f"class {c} -> TN != confusion matrix"

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "conf_mat": conf_mat,
    }

def get_metrics(outputs_argmax, labels):
    assert len(unique_labels(labels.flatten())) <= 2
    assert outputs_argmax.shape == labels.shape, f"{outputs_argmax.shape} != {labels.shape}"

    classes = 2
    num_elements = outputs_argmax.numel()
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, outputs_argmax, labels=list(range(classes)), zero_division=0)
    acc = (torch.sum(outputs_argmax == labels) / num_elements).cpu().detach().numpy().item()
    samples_per_class = [torch.sum(labels == c).item() for c in range(classes)]
    conf_mat = get_confusion_matrix(outputs_argmax, labels)
    tp, fp, fn, tn = conf_mat["tp"], conf_mat["fp"], conf_mat["fn"], conf_mat["tn"]
    macro_f1 = np.sum(f1) / f1.shape[0]
    mcc = sklearn.metrics.matthews_corrcoef(labels, outputs_argmax)

    return {
        "samples_per_class": samples_per_class,
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }
