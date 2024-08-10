
import sys
import logging

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
def inference_eval(model, tokenizer, dataset, loss_function=None, device=None, max_tokens=None, threshold=0.5, monolingual=None):
    training = model.training

    model.eval()

    all_loss = []
    all_outputs = []
    all_labels = []
    total_tokens = 0
    normal = False
    do_eval = False

    if dataset is None:
        # interactive
        dataloader = ["placeholder"]

        assert max_tokens is None, max_tokens
    elif dataset == '-':
        dataloader = sys.stdin
    else:
        dataloader = dataset.dataloader
        normal = True
        do_eval = True

    for batch in dataloader:
        if max_tokens and batch is None:
            # Batch is under construction using max_tokens...
            continue

        if batch == "placeholder":
            loss_function = None

            assert dataset is None

            if monolingual:
                s = input("Sentence: ")
            else:
                s1 = input("Source sentence: ")
                s2 = input("Target sentence: ")
                s = f"{s1}{tokenizer.sep_token}{s2}"

            if s not in ('', tokenizer.sep_token):
                dataloader.append("placeholder")
            else:
                break

        if not normal:
            if batch != "placeholder":
                s = batch.rstrip("\r\n").split('\t')

                if not do_eval:
                    do_eval = (len(s) == 3) or (monolingual and (len(s) == 2))

                assert len(s) == ((1 + int(do_eval)) if monolingual else (2 + int(do_eval))), s

                if do_eval:
                    label = s[1 if monolingual else 2]

                    assert label in ('0', '1'), label

                    label = float(label)
                else:
                    loss_function = None

                s = tokenizer.sep_token.join(s[:1 if monolingual else 2])

            tokens = utils.encode(tokenizer, [s], max_length=tokenizer.model_max_length, return_tensors=None, truncation=True)["input_ids"]
            tokens = torch.tensor(tokens)
            attention_mask = torch.LongTensor(torch.ones_like(tokens))
            labels = torch.tensor([label]) if do_eval else None
            batch = {
                "url_tokens": tokens,
                "url_attention_mask": attention_mask,
                "labels": labels,
                }

        total_tokens += sum([len(urls[urls != tokenizer.pad_token_id]) for urls in batch["url_tokens"]]) if normal else 0

        # Inference
        results = inference(model, batch, loss_function=loss_function, device=device, threshold=threshold)

        # Results
        if not normal and not do_eval:
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

    if not normal and not do_eval:
        return None

    if normal:
        assert total_tokens == dataset.total_tokens, f"{total_tokens} != {dataset.total_tokens}"

    all_outputs = torch.as_tensor(all_outputs)
    all_labels = torch.as_tensor(all_labels)
    metrics = get_metrics(all_outputs, all_labels)
    metrics["loss_average"] = sum(all_loss) / len(all_loss)

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
