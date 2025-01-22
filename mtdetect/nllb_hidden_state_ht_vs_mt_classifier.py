
import sys
import gzip
import math
import time
import pickle
import logging
import argparse

del sys.path[0] # remove wd path

import mtdetect.inference as inference
import mtdetect.utils.utils as utils

import torch
import torch.nn as nn
import transformers
import numpy as np

logger = logging.getLogger("mtdetect.nllb_hidden_state_ht_vs_mt_classifier")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, nlayers, projection_in=None, max_seq_len=512, embedding_dropout=0.5, dropout_p=0.5, classifier_dropout_p=0.5, lm_dropout_p=0.5,
                 num_labels=1, initial_layer_norm=False, initial_layer_norm_first=False, lm_projection_in=None, lang_model=None, debug_labels=False):
        super(TransformerModel, self).__init__()

        initial_dim = d_model if projection_in is None else projection_in
        initial_dim_layer_norm = initial_dim if initial_layer_norm_first else d_model
        self.initial_layer_norm_first = initial_layer_norm_first
        self.layer_norm = nn.LayerNorm(initial_dim_layer_norm) if initial_layer_norm else None
        self.pos_encoder = PositionalEncoding(d_model, embedding_dropout, max_seq_len=max_seq_len)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout_p, norm_first=False) #, activation="gelu", layer_norm_eps=1e-12)
        self.projection = None if projection_in is None else nn.Linear(projection_in, d_model)
        #self.lm_projection = None if lm_projection_in is None else nn.Linear(lm_projection_in, d_model)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.initializer_range = 0.02
        # classifier
        self.pooler = nn.Linear(d_model, d_model) # https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/models/bert/modeling_bert.py#L737
        self.pooler_activation = nn.Tanh()
        self.classifier_dropout = nn.Dropout(classifier_dropout_p)
        self.classifier = nn.Linear(d_model + (lm_projection_in if lm_projection_in is not None else 0) + (1 if debug_labels else 0), num_labels)
        self.lm_projection_in = lm_projection_in
        self.lm_dropout = nn.Dropout(lm_dropout_p)

        self.init_weights()

        # Store lang model after weights initialization to avoid problems......
        self.lang_model = lang_model

    # https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/models/bert/modeling_bert.py#L836
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig.initializer_range
    def init_weights(self):
        initializer_range = self.initializer_range

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, src, mask=None, lm_src=None, debug_labels=None):
        #src = src * math.sqrt(self.d_model)
        #src = self.pos_encoder(src)

        if self.initial_layer_norm_first:
            src = self.layer_norm(src) if self.layer_norm is not None else src
            src = self.projection(src) if self.projection is not None else src
        else:
            src = self.projection(src) if self.projection is not None else src
            src = self.layer_norm(src) if self.layer_norm is not None else src

        assert len(src.shape) == 3

        if lm_src is not None:
            assert mask is None, "Not supported"
            #assert self.lm_projection is not None

            #lm_src = self.lm_projection(lm_src)

            assert len(lm_src.shape) == 2
            assert src.shape[0] == lm_src.shape[0]
            #assert src.shape[2] == lm_src.shape[1]
            assert self.lm_projection_in == lm_src.shape[1]

            #lm_src = lm_src.unsqueeze(1)
            #src = torch.cat((lm_src, src), dim=1) # add lm_src as a "new" token in the first position

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=mask)

        # https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/models/bert/modeling_bert.py#L1680
        # Same dropout layer twice: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1558
        output = output[:,0,:] # Only the first token (CLS)
        output = self.classifier_dropout(output)
        output = self.pooler(output) 
        output = self.pooler_activation(output)
        output = self.classifier_dropout(output)

        if lm_src is not None:
            assert lm_src.shape[1] + output.shape[1] == self.d_model + self.lm_projection_in

            #output.zero_() # TODO remove (debug)
            #lm_src.zero_() # TODO remove (debug)

            lm_src = self.lm_dropout(lm_src)
            output = torch.cat((lm_src, output), dim=1)

        if debug_labels is not None:
            assert isinstance(debug_labels, torch.Tensor), type(debug_labels)

            debug_labels = debug_labels.detach().clone()

            assert torch.logical_or(debug_labels == 1, debug_labels == 0).cpu().detach().all().item()

            debug_labels = debug_labels * 2 - 1 # instead of 0s and 1s, -1s and 1s

            if len(debug_labels.shape) == 1:
                debug_labels = debug_labels.unsqueeze(1)

            assert len(debug_labels.shape) == 2, debug_labels.shape
            assert len(output.shape) == 2, output.shape
            assert debug_labels.shape[0] == output.shape[0], f"{debug_labels.shape} vs {output.shape}"
            assert debug_labels.shape[1] == 1, debug_labels.shape

            output = torch.cat((debug_labels, output), dim=1)

        output = self.classifier(output) # logits

        return output

def preprocess_tokens(tokens, tokenizer, max_length):
    assert len(tokens["input_ids"].shape) == 2
    current_bsz, ntokens = tokens["input_ids"].shape

    if ntokens > max_length:
        tokens["input_ids"] = tokens["input_ids"][:,:max_length]
        tokens["attention_mask"] = tokens["attention_mask"][:,:max_length]

        for idx in range(current_bsz):
            if tokens["input_ids"][idx,-1] != tokenizer.pad_token_id:
                tokens["input_ids"][idx,-1] = tokenizer.eos_token_id

    return tokens

def preprocess(text, tokenizer, device, max_length):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, padding=True).to(device)
    inputs = preprocess_tokens(inputs, tokenizer, max_length)

    return inputs

def get_model_last_hidden_state(model, src_inputs, trg_inputs, skip_modules=(), to_cpu=True, layer=-1):
    model_output = model(**src_inputs, decoder_input_ids=trg_inputs["input_ids"], output_hidden_states=True)
    results = {}

    for module, tokens in zip(("encoder", "decoder"), (src_inputs, trg_inputs)):
        if module in skip_modules:
            continue

        last_hidden_state = model_output[f"{module}_hidden_states"][layer]
        expected_shape = (*tokens["input_ids"].shape, model.config.d_model)

        assert len(last_hidden_state.shape) == 3, last_hidden_state.shape
        assert last_hidden_state.shape == expected_shape, f"{last_hidden_state.shape} vs {expected_shape}"

        results[f"{module}_last_hidden_state"] = last_hidden_state

        if to_cpu:
            results[f"{module}_last_hidden_state"] = results[f"{module}_last_hidden_state"].cpu()

        results[f"{module}_last_hidden_state"] = results[f"{module}_last_hidden_state"].detach()

#        for k in tokens.keys():
#            tokens[k] = tokens[k].cpu().detach()
#
#        results[f"{module}_tokens"] = tokens

    return results

def read(fn, limit=None):
    #data = {"src": [], "trg": [], "labels": []}
    data = []

    with open(fn) as fd:
        for idx, l in enumerate(fd):
            if limit is not None and idx >= limit:
                break

            src, trg, label = l.rstrip("\r\n").split('\t')
            label = float(label)

            #data["src"].append(src)
            #data["trg"].append(trg)
            #data["labels"].append(label)
            data.append((src, trg, label))

    return data

def read_pickle(fn, *args, **kwargs):
    all_fns = fn.split(':')
    data = []

    for fn in all_fns:
        result = _read_pickle(fn, *args, **kwargs)

        assert isinstance(result, list), type(result)
        assert isinstance(result[0], torch.Tensor), type(result[0])

        data.append(result)

    for idx in range(len(data) - 1):
        assert isinstance(data[idx + 0], list)
        assert isinstance(data[idx + 1], list)
        assert len(data[idx + 0]) == len(data[idx + 1]) # same number of batches

        for idx2 in range(len(data[idx])):
            assert isinstance(data[idx + 0][idx2], torch.Tensor)
            assert isinstance(data[idx + 1][idx2], torch.Tensor)
            assert data[idx + 0][idx2].shape == data[idx + 1][idx2].shape

    transposed = list(zip(*data))
    result = [torch.cat(t, dim=-1) for t in transposed]

    assert len(result) == len(data[0])

    for idx in range(len(result)):
        assert result[idx].shape[:-1] == data[0][idx].shape[:-1]
        assert result[idx].shape[-1] == data[0][idx].shape[-1] * len(all_fns)

    return result

def _read_pickle(fn, k=None, limit=None, max_split_size=None):
    if max_split_size is not None:
        assert k is not None, "Easier implementation"

    data = None
    open_func = gzip.open if fn.endswith(".gz") else open

    logger.info("Loading pickle file: %s (key: %s)", fn, k)

    with open_func(fn, "rb") as fd:
        data = pickle.load(fd)

    assert isinstance(data, dict), type(data)

    for _k1 in data.keys():
        if isinstance(data[_k1], dict):
            assert len(data[_k1].keys()) == 1, data[_k1].keys()

            # workaround
            key = list(data[_k1].keys())[0]
            data[_k1] = data[_k1][key]

#        assert isinstance(data[_k1], dict), type(data[_k1])
        assert isinstance(data[_k1], list), type(data[_k1])
        assert isinstance(data[_k1][0], torch.Tensor), type(data[_k1][0])

#        for _k2 in data[_k1].keys():
#            assert isinstance(data[_k1][_k2], list), type(data[_k1][_k2])
#            assert isinstance(data[_k1][_k2][0], torch.Tensor), type(data[_k1][_k2][0])

    if k is not None:
        data = data[k]

        if max_split_size:
            _data = []

            for data_idx, t in enumerate(data, 1):
                if t.shape[0] <= max_split_size:
                    _data.append(t)
                else:
                    n1 = max_split_size
                    n2 = t.shape[0]
                    idxs = [n1 * i for i in range(n2 // n1 + 1 + (0 if n2 % n1 == 0 else 1))]

                    assert len(idxs) > 1, idxs
                    n = 0

                    for idx in range(len(idxs) - 1):
                        _data.append(t[idxs[idx]:idxs[idx + 1]])

                        n += idxs[idx + 1] - idxs[idx]

                    if data_idx < len(data):
                        assert n == t.shape[0], f"{n} vs {t.shape[0]}"
                    else:
                        assert n >= t.shape[0], f"{n} vs {t.shape[0]}"

            data = _data

    if limit is not None:
        if k is not None:
            data = data[:limit]
        else:
            for _k1 in data.keys():
                data[_k1] = data[_k1][:limit]

    return data

def make_batches(data, bsz, pt_data=None, lm_data=None):
    assert bsz > 0

    idx = 0
    batch = []

    def get_result():
        assert len(batch) > 0

        result = [batch]

        if pt_data is not None:
            assert pt_data[idx].shape[0] == len(batch), f"{idx}: {pt_data[idx].shape[0]} vs {len(batch)}"

            result.append(pt_data[idx])
        else:
            result.append(None)

        if lm_data is not None:
            assert lm_data[idx].shape[0] == len(batch), f"{idx}: {lm_data[idx].shape[0]} vs {len(batch)}"

            result.append(lm_data[idx])
        else:
            result.append(None)

        return result

    for d in data:
        batch.append(d)

        if len(batch) >= bsz:
            yield tuple(get_result())

            idx += 1
            batch = []

    if len(batch) > 0:
        yield tuple(get_result())

        idx += 1
        batch = []

    if pt_data is not None:
        assert idx == len(pt_data)

    if lm_data is not None:
        assert idx == len(lm_data)

def apply_inference(model, data, mask=None, target=None, loss_function=None, threshold=0.5, loss_apply_sigmoid=False, data_lm=None, debug_labels=None):
    model_outputs = model(data, mask=mask, lm_src=data_lm, debug_labels=debug_labels)
    outputs = model_outputs
    outputs = outputs.squeeze(1)
    loss = None

    assert len(outputs.shape) == 1, outputs.shape

    if loss_function is not None and target is not None:
        loss = loss_function(torch.sigmoid(outputs) if loss_apply_sigmoid else outputs, target)

    outputs_classification = torch.sigmoid(outputs).cpu().detach().tolist()
    outputs_classification = list(map(lambda n: int(n >= threshold), outputs_classification))

    results = {
        "outputs": outputs,
        "outputs_classification_detach_list": outputs_classification,
        "loss": loss,
    }

    return results

def eval(model, translation_model, data_generator, direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, translation_tokenizer,
         max_length, max_new_tokens, print_result=False, print_desc='-', threshold=0.5, layer=-1, lang_model=None, lang_model_tokenizer=None, lm_frozen_params=False, max_length_encoder=512,
         debug=False):
    assert not print_result, "Code not working"

    training = model.training
    lang_model_training = False

    model.eval()

    if lang_model:
        lang_model_training = lang_model.training

        lang_model.eval()

    all_outputs = []
    all_labels = []
    print_idx = 0

    for batch, batch_pt, batch_lm in data_generator:
        src, trg, labels = zip(*batch)
        data_lm = None

        if direction == "trg2src":
            src, trg = trg, src

        if batch_pt is None:
            src = [f"{source_lang_token} {_src}{eos_token_token}" for _src in src]
            trg = [f"{decoder_start_token_token}{target_lang_token} {_trg}{eos_token_token}" for _trg in trg]
            src_inputs = preprocess(src, translation_tokenizer, device, max_length)
            trg_inputs = preprocess(trg, translation_tokenizer, device, max_new_tokens)
            translation_output = get_model_last_hidden_state(translation_model, src_inputs, trg_inputs, skip_modules=("encoder",), to_cpu=False, layer=layer)
            data = translation_output["decoder_last_hidden_state"].to(device)
        else:
            data = batch_pt.to(device)

        if batch_lm is not None:
            assert lang_model is not None
            assert lm_frozen_params

            data_lm = batch_lm.to(device)
        else:
            if lang_model:
                assert not lm_frozen_params

                batch = [f"{_src}{lang_model_tokenizer.sep_token}{_trg}" for _src, _trg in zip(src, trg)]
                classifier_token = get_lang_model_cls_token(batch, lang_model, lang_model_tokenizer, device, max_length_encoder, to_cpu=False, detach=True)
                data_lm = classifier_token.to(device)

        target = torch.tensor(labels).to(device)
        results = apply_inference(model, data, target=None, loss_function=None, threshold=threshold, loss_apply_sigmoid=False, data_lm=data_lm) #, debug_labels=target if debug else None)
        outputs_classification = results["outputs_classification_detach_list"]
        outputs = results["outputs"]
        outputs = torch.sigmoid(outputs).cpu().detach().tolist()
        labels = target.cpu()
        labels = torch.round(labels).type(torch.long)

        all_outputs.extend(outputs_classification)
        all_labels.extend(labels.tolist())

        if print_result:
            # TODO not working
            assert len(data["source_text"]) == len(outputs)
            assert len(data["target_text"]) == len(outputs)
            assert len(labels) == len(outputs)
            assert len(outputs) == len(outputs_classification)

            for source_text, target_text, output, label, output_classification_aux in zip(data["source_text"], data["target_text"], outputs, labels, outputs_classification):
                output_classification = int(output >= threshold)

                assert output_classification in (0, 1), output_classification
                assert output_classification == output_classification_aux
                assert label in (0, 1), label

                if output_classification == 1 and label == 1:
                    conf_mat_value = "tp"
                elif output_classification == 1 and label == 0:
                    conf_mat_value = "fp"
                elif output_classification == 0 and label == 1:
                    conf_mat_value = "fn"
                elif output_classification == 0 and label == 0:
                    conf_mat_value = "tn"
                else:
                    raise Exception(f"Unexpected values: {output_classification} vs {label}")

                print(f"inference: {print_desc}\t{print_idx}\t{output}\tlabel={label}\t{conf_mat_value}\t{source_text}\t{target_text}")

                print_idx += 1

    all_outputs = torch.as_tensor(all_outputs)
    all_labels = torch.as_tensor(all_labels)
    results = inference.get_metrics(all_outputs, all_labels)

    if training:
        model.train()

        if not lang_model_training and model.lang_model:
            # Previous .train() might have enabled the language model training...
            model.lang_model.eval()

    if lang_model_training:
        if hasattr(model, "lang_model"):
            model.lang_model.train()

            assert lang_model is model.lang_model
        else:
            lang_model.train()

    return results

def load_model(model_input, pretrained_model, device):
    local_model = bool(model_input)
    config = transformers.AutoConfig.from_pretrained(pretrained_model, num_labels=1)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
    tokenizer = utils.get_tokenizer(pretrained_model)

    if local_model:
        state_dict = torch.load(model_input, weights_only=True, map_location=device) # weights_only: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
                                                                                     # map_location: avoid creating a new process and using additional and useless memory

        model.load_state_dict(state_dict)

    model = model.to(device)

    return model, tokenizer

pooler_warning = False

# TODO apply_pooler=False ? previous models were not applying the pooler layer
def get_lang_model_cls_token(batch, lang_model, lang_model_tokenizer, device, max_length_encoder, to_cpu=True, detach=True, apply_pooler=True):
    if not apply_pooler:
        global pooler_warning

        pooler_warning = True

        logger.warning("Pooler layer is not being applied for the LM: no activation function will be applied either")

    assert isinstance(batch, list)
    assert isinstance(batch[0], str)

    inputs = lang_model_tokenizer.batch_encode_plus(batch, return_tensors=None, add_special_tokens=True, max_length=max_length_encoder,
                                                    return_attention_mask=False, truncation=True, padding="longest")
    lm_input_ids = inputs["input_ids"]
    lm_input_ids = torch.tensor(lm_input_ids)
    lm_attention_mask = utils.get_attention_mask(lang_model_tokenizer, lm_input_ids)
    lm_input_ids = lm_input_ids.to(device)
    lm_attention_mask = lm_attention_mask.to(device)

    assert lm_attention_mask.shape == lm_input_ids.shape

    output = lang_model(input_ids=lm_input_ids, attention_mask=lm_attention_mask, output_hidden_states=True)
    last_hidden_state = output["hidden_states"][-1]
    classifier_token = last_hidden_state[:,0,:]

    if apply_pooler:
        assert hasattr(lang_model, "pooler")

        _classifier_token = lang_model.pooler(last_hidden_state)
        _classifier_token = lang_model.dropout(_classifier_token)

        assert classifier_token.shape == _classifier_token.shape, f"{classifier_token.shape} vs {_classifier_token.shape}"

        classifier_token = _classifier_token

    if to_cpu:
        classifier_token = classifier_token.cpu()

    if detach:
        classifier_token = classifier_token.detach()

    assert classifier_token.shape == (len(batch), lang_model.config.hidden_size)

    return classifier_token

def main(args):
    debug = args.debug
    seed = args.seed

    if seed < 0:
        seed = int(time.time() * 1000)

    logger.info("Seed: %d", seed)

    transformers.set_seed(seed)

    train_fn = args.dataset_train_filename
    dev_fn = args.dataset_dev_filename
    test_fn = args.dataset_test_filename
    train_pickle_fn = args.pickle_train_filename
    dev_pickle_fn = args.pickle_dev_filename
    test_pickle_fn = args.pickle_test_filename
    _src_lang = args.source_lang
    _trg_lang = args.target_lang
    direction = args.direction
    batch_size = args.batch_size
    save_model_path = args.model_output
    load_model_path = args.model_input
    learning_rate = args.learning_rate
    num_layers = args.num_layers
    nhead = args.num_attention_heads
    do_inference = args.inference
    pretrained_model = args.pretrained_model
    pretrained_model_layer = args.pretrained_model_target_layer
    model_inference_skip_train = args.skip_training_set_during_inference
    skip_test_eval = args.skip_test_set_eval
    patience = args.patience
    patience_metric = args.dev_patience_metric
    epochs = args.epochs
    train_until_patience = args.train_until_patience
    threshold = args.threshold
    max_length_tokens = args.max_length_tokens
    optimizer_str = args.optimizer
    optimizer_args = args.optimizer_args
    scheduler_str = args.lr_scheduler
    scheduler_args = args.lr_scheduler_args
    dropout_p = args.dropout
    lm_dropout_p = args.lm_dropout
    limit = args.data_limit
    gradient_accumulation = args.gradient_accumulation
    actual_batch_size = batch_size * gradient_accumulation
    lm_pretrained_model = args.lm_pretrained_model
    lm_model_input = args.lm_model_input
    lm_frozen_params = args.lm_frozen_params
    lm_learning_rate = args.lm_learning_rate

    if lm_pretrained_model:
        logger.info("LM is going to be used: %s (local file: %s)", lm_pretrained_model, lm_model_input)

    if gradient_accumulation > 1:
        logger.info("Gradient accumulation enabled (i.e., >1): %d (note that if disabled, the same results would be obtained if dropout is disabled for both HT vs MT classifier and language model, train shuffle is disabled, and float precision errors are ignored)", gradient_accumulation)

    logger.info("Batch size: %d (actual batch size: %d)", batch_size, actual_batch_size)

    assert len(train_pickle_fn.split(':')) == len(dev_pickle_fn.split(':')) == len(test_pickle_fn.split(':'))

    # read data
    n_pickle_files = train_pickle_fn.count(':') + (0 if len(train_pickle_fn) == 0 else 1)
    train_data = read(train_fn, limit=None if limit is None else (limit * batch_size))
    dev_data = read(dev_fn, limit=None if limit is None else (limit * batch_size))
    test_data = read(test_fn, limit=None if limit is None else (limit * batch_size))
    train_pickle_data = read_pickle(train_pickle_fn, k="decoder_last_hidden_state", limit=limit, max_split_size=batch_size) if bool(train_pickle_fn) else None
    dev_pickle_data = read_pickle(dev_pickle_fn, k="decoder_last_hidden_state", limit=limit, max_split_size=batch_size) if bool(dev_pickle_fn) else None
    test_pickle_data = read_pickle(test_pickle_fn, k="decoder_last_hidden_state", limit=limit, max_split_size=batch_size) if bool(test_pickle_fn) else None
    all_pickle_data_loaded = bool(train_pickle_fn) and bool(dev_pickle_fn) and bool(test_pickle_fn)

    logger.info("Train: %d", len(train_data))
    logger.info("Dev: %d", len(dev_data))
    logger.info("Test: %d", len(test_data))

    train_pickle_data_n = sum([t.shape[0] for t in train_pickle_data]) if bool(train_pickle_fn) else len(train_data)
    dev_pickle_data_n = sum([t.shape[0] for t in dev_pickle_data]) if bool(dev_pickle_fn) else len(dev_data)
    test_pickle_data_n = sum([t.shape[0] for t in test_pickle_data]) if bool(test_pickle_fn) else len(test_data)

    assert len(train_data) == train_pickle_data_n, f"{len(train_data)} vs {train_pickle_data_n}"
    assert len(dev_data) == dev_pickle_data_n
    assert len(test_data) == test_pickle_data_n

    #random.shuffle(train_data) # in-place shuffle

    # variables
    src_lang, trg_lang = (_src_lang, _trg_lang) if direction == "src2trg" else (_trg_lang, _src_lang)
    source_lang_token = src_lang
    target_lang_token = trg_lang
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim_feedforward = 2048

    # translation model
    translation_tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=src_lang, tgt_lang=trg_lang)
    translation_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
    translation_model = translation_model.to(device if not all_pickle_data_loaded else "cpu").eval()
    max_length = min(translation_model.config.max_length, max_length_tokens)
    max_new_tokens = min(translation_model.generation_config.max_length, max_length_tokens)
    eos_token_token = translation_tokenizer.convert_ids_to_tokens(translation_model.generation_config.eos_token_id)
    decoder_start_token_token = translation_tokenizer.convert_ids_to_tokens(translation_model.generation_config.decoder_start_token_id)
    max_seq_len = max_new_tokens

    # LM
    lang_model, lang_model_tokenizer = load_model(lm_model_input, lm_pretrained_model, None) if lm_pretrained_model else (None, None)
    max_length_encoder = 512 # TODO use argument
    _max_length_encoder = max_length_encoder
    lang_model_batch_size = train_pickle_data[0].shape[0] # Same batch size as pickle files
    train_lm_data = [] if lang_model and lm_frozen_params else None
    dev_lm_data = [] if lang_model and lm_frozen_params else None
    test_lm_data = [] if lang_model and lm_frozen_params else None

    if lang_model:
        # Add LM data to inputs
        lang_model = lang_model.eval()
        lang_model = lang_model.to(device)

        assert n_pickle_files > 0, "LM is only supported with pickle files (easier implementation)"

        max_length_encoder = utils.get_encoder_max_length(lang_model, lang_model_tokenizer, max_length_tokens=_max_length_encoder)
        max_length_encoder = min(max_length_encoder, _max_length_encoder)

        logger.debug("Max length: %d", max_length_encoder)

        if lm_frozen_params:
            logger.info("LM parameters are frozen: computing embeddings just once")

            for desc, data_str, data_pickle, data_lm in (("train", train_data, train_pickle_data, train_lm_data), ("dev", dev_data, dev_pickle_data, dev_lm_data), ("test", test_data, test_pickle_data, test_lm_data)):
                logger.info("Generating LM data: %s", desc)

                all_texts = []
                idx = 0

                for source_text, target_text, label in data_str:
                    all_texts.append(f"{source_text}{lang_model_tokenizer.sep_token}{target_text}")

                while len(all_texts) > 0:
                    batch = all_texts[:lang_model_batch_size]

                    assert isinstance(data_pickle[idx], torch.Tensor), type(data_pickle[idx])
                    assert len(batch) == data_pickle[idx].shape[0], f"{data_pickle[idx].shape} vs {len(batch)}"
                    assert data_pickle[idx].shape[2] == translation_model.config.d_model, data_pickle[idx].shape

                    classifier_token = get_lang_model_cls_token(batch, lang_model, lang_model_tokenizer, device, max_length_encoder)

                    data_lm.append(classifier_token)

                    # update remaining data to process
                    all_texts = all_texts[lang_model_batch_size:]
                    idx += 1

                    if idx % 100 == 0:
                        logger.debug("Batches of data processed using the LM: %s: %d", desc, idx)

                assert idx == len(data_pickle) # last batch of data
                assert len(data_str) == sum([t.shape[0] for t in data_lm]), desc

                logger.debug("Total batches of data processed using the LM: %s: %d", desc, idx)

    # classifier
    #projection_in = []
    #projection_in.append(max(n_pickle_files, 1) * translation_model.config.d_model)  # 1024 for facebook/nllb-200-distilled-600M
    #projection_in.append(lang_model.config.hidden_size if lang_model else 0)
    projection_in = max(n_pickle_files, 1) * translation_model.config.d_model  # 1024 for facebook/nllb-200-distilled-600M
    #projection_in = None
    lm_projection_in = lang_model.config.hidden_size if lang_model else None
    d_model = 512 # TODO use argument
    #d_model = 128
    #d_model = translation_model.config.d_model

    #logger.info("Projection: %d * %d + %d = %s -> %d", max(n_pickle_files, 1), translation_model.config.d_model, projection_in[1], " + ".join(map(str, projection_in)), d_model)
    logger.info("Projection for the MT model: %d * %d = %d -> %d", max(n_pickle_files, 1), translation_model.config.d_model, projection_in, d_model)

    #if lang_model:
    #    logger.info("Projection for the LM: %d -> %d", lm_projection_in, d_model)

    #projection_in = sum(projection_in)

    # sanity-check
    for idx, (_p, _d) in enumerate(((train_pickle_data, train_data), (dev_pickle_data, dev_data), (test_pickle_data, test_data))):
        if _p:
            assert isinstance(_p, list), type(_p)

            s = 0

            for idx2, d in enumerate(_p, 1):
                assert isinstance(d, torch.Tensor), type(d)
                assert d.shape[-1] == projection_in, f"{d.shape}[-1] vs {projection_in}"

                if idx2 == len(_p):
                    assert d.shape[0] <= batch_size
                else:
                    assert d.shape[0] == batch_size

                s += d.shape[0]

            assert s == len(_d), f"{idx}: {s} vs {len(_d)}"

    model = TransformerModel(d_model, nhead, dim_feedforward, num_layers,
                             projection_in=projection_in, max_seq_len=max_seq_len,
                             embedding_dropout=dropout_p, dropout_p=dropout_p, classifier_dropout_p=dropout_p,
                             lm_dropout_p=lm_dropout_p, initial_layer_norm=False, lm_projection_in=lm_projection_in,
                             lang_model=lang_model if lang_model and not lm_frozen_params else None,
                            )
#                             debug_labels=True if debug else False)

    if load_model_path:
        logger.info("Loading init model: %s", load_model_path)

        model_state_dict = torch.load(load_model_path, weights_only=True, map_location=device)
        current_model_state_dict_keys = set(model.state_dict().keys())

#        if lang_model and "lm_projection.weight" in current_model_state_dict_keys and "lm_projection.weight" not in model_state_dict:
#            assert lm_projection_in is not None
#
#            expected_shape = (lm_projection_in, d_model)
#
#            assert model.state_dict()["lm_projection.weight"].shape == expected_shape, f"{model.state_dict()['lm_projection.weight'].shape} vs {expected_shape}"
#
#            # Our model has a LM projection layer that is new
#            logger.warning("Fixing shape of layers...")
#
#            model_state_dict["lm_projection.weight"] = nn.Parameter(model.state_dict()["lm_projection.weight"], requires_grad=False)
#            model_state_dict["lm_projection.bias"] = nn.Parameter(model.state_dict()["lm_projection.bias"], requires_grad=False)

        load_model_state_dict_keys = set(model_state_dict.keys())
        model_state_dict_keys_intersection = set.intersection(current_model_state_dict_keys, load_model_state_dict_keys)
        model_state_dict_keys_load_current_diff = set.difference(load_model_state_dict_keys, current_model_state_dict_keys)
        model_state_dict_keys_current_load_diff = set.difference(current_model_state_dict_keys, load_model_state_dict_keys)

        logger.debug("Model keys (current: %d; load: %d; intersection: %d): load - current: %s: current - load: %s",
                     len(current_model_state_dict_keys), len(load_model_state_dict_keys), len(model_state_dict_keys_intersection), model_state_dict_keys_load_current_diff, model_state_dict_keys_current_load_diff)

        lang_model_dict_keys = set([f"lang_model.{k}" for k in lang_model.state_dict().keys()])

        assert current_model_state_dict_keys != load_model_state_dict_keys
        assert current_model_state_dict_keys == set.union(load_model_state_dict_keys, lang_model_dict_keys)

        model.load_state_dict(model_state_dict, strict=False)

    if do_inference:
        model = model.eval()
    else:
        model = model.train()

    model = model.to(device)

    for p in model.parameters():
        p.requires_grad_(not do_inference)

    if lang_model:
        if lm_frozen_params:
            lang_model.eval()
        else:
            lang_model.train()

        for p in lang_model.parameters():
            p.requires_grad_(not lm_frozen_params)

    training_steps_per_epoch = len(train_data) // batch_size + (0 if len(train_data) % batch_size == 0 else 1) # number of batches
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to --train-until-patience

    logger.info("Batches per epoch: %d (total for %d epochs: %d)", training_steps_per_epoch, epochs, training_steps)

    if not do_inference:
        model_parameters_data = list(filter(lambda d: d[1].requires_grad, [(k, p) for k, p in model.named_parameters() if not k.startswith("lang_model.")]))
        model_parameters = [d[1] for d in model_parameters_data]
        model_parameters_names = [d[0] for d in model_parameters_data]
        lm_model_parameters = list(filter(lambda p: p.requires_grad, lang_model.parameters())) if lang_model else []
        optimizer_args_params = [{"params": model_parameters, "lr": learning_rate}]

        assert len(model_parameters_data) == len(model_parameters) == len(model_parameters_names)

        if lm_model_parameters:
            optimizer_args_params.append({"params": lm_model_parameters, "lr": lm_learning_rate})

        logger.info("Parameters with requires_grad=True: %d (LM: %d)", len(model_parameters), len(lm_model_parameters))

        optimizer, scheduler = \
                utils.get_lr_scheduler_and_optimizer_using_argparse_values(optimizer_str, scheduler_str, optimizer_args, scheduler_args, optimizer_args_params, learning_rate, training_steps, training_steps_per_epoch, logger)

    # training args
    current_patience = 0
    epoch = 0
    do_training = not do_inference and (epoch < epochs or train_until_patience)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    #loss_function = nn.BCELoss(reduction="none")
    loss_apply_sigmoid = False # Should be True if loss_function = nn.BCELoss()
    #loss_apply_sigmoid = True
    log_steps = 100 # TODO argument
    sum_epoch_loss = np.inf
    early_stopping_best_loss = np.inf
    early_stopping_best_result_dev = -np.inf # accuracy

    while do_training:
        if lang_model is not None:
            s1 = sum([torch.sum(q.data).item() for q in lang_model.parameters()])

            if model.lang_model is not None: # not frozen
                s2 = sum([torch.sum(q.data).item() for q in model.lang_model.parameters()])

                assert np.isclose(s1, s2)

            logger.debug("LM signature: %s", s1)

        epoch_loss = []

        logger.info("Epoch #%d", epoch + 1)

        dev_results = eval(model, translation_model, make_batches(dev_data, batch_size, pt_data=dev_pickle_data, lm_data=dev_lm_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, translation_tokenizer, max_length, max_new_tokens, threshold=threshold, layer=pretrained_model_layer, lang_model=lang_model, lang_model_tokenizer=lang_model_tokenizer, lm_frozen_params=lm_frozen_params, max_length_encoder=max_length_encoder, debug=debug)

        if len(epoch_loss) > 0 and sum_epoch_loss < early_stopping_best_loss:
            logger.info("Better loss result: %s -> %s", early_stopping_best_loss, sum_epoch_loss)

            early_stopping_best_loss = sum_epoch_loss

        logger.info("Dev eval: %s", dev_results)

        early_stopping_metric_dev = dev_results[patience_metric]

        if early_stopping_metric_dev > early_stopping_best_result_dev:
            logger.info("Patience better dev result (metric: %s): %s -> %s", patience_metric, early_stopping_best_result_dev, early_stopping_metric_dev)

            current_patience = 0
            early_stopping_best_result_dev = early_stopping_metric_dev

            if save_model_path:
                logger.info("Saving best model: %s", save_model_path)

                torch.save(model.state_dict(), save_model_path)
        elif patience > 0:
            current_patience += 1

            logger.info("Exhausting patience... %d / %d", current_patience, patience)

        if patience > 0 and current_patience >= patience:
            logger.info("Patience is over ...")

            do_training = False

            break # we need to force the break to avoid the training of the current epoch

        model.zero_grad()
        final_loss = None
        loss_elements = 0

        for batch_idx, (batch, batch_pt, batch_lm) in enumerate(make_batches(train_data, batch_size, pt_data=train_pickle_data, lm_data=train_lm_data), 1):
            src, trg, labels = zip(*batch)
            data_lm = None

            if direction == "trg2src":
                src, trg = trg, src

            assert len(src) == len(trg) == len(labels)

            if batch_idx == training_steps_per_epoch:
                assert len(src) <= batch_size, f"{len(src)} vs {batch_size}"
            else:
                assert len(src) == batch_size, f"{len(src)} vs {batch_size}"

            if batch_pt is None:
#            if batch_pt is None or debug:
#                if debug:
#                    translation_model = translation_model.to(device)

                src_translation_model = [f"{source_lang_token} {_src}{eos_token_token}" for _src in src]
                trg_translation_model = [f"{decoder_start_token_token}{target_lang_token} {_trg}{eos_token_token}" for _trg in trg]
                src_inputs = preprocess(src_translation_model, translation_tokenizer, device, max_length)
                trg_inputs = preprocess(trg_translation_model, translation_tokenizer, device, max_new_tokens)

#                if debug:
#                    assert len(trg_inputs["input_ids"].shape) == 2
#
#                    for debug_idx in range(trg_inputs["input_ids"].shape[0]):
#                        debug_idx2 = np.random.randint(torch.sum(trg_inputs["attention_mask"][debug_idx]).cpu().detach().item())
#                        trg_inputs["input_ids"][debug_idx][debug_idx2] = 123 # random token in random but valid (i.e., non-padding) idx

                translation_output = get_model_last_hidden_state(translation_model, src_inputs, trg_inputs, skip_modules=("encoder",), to_cpu=False, layer=pretrained_model_layer)
                data = translation_output["decoder_last_hidden_state"]

#                if debug and batch_pt is not None:
#                    logger.warning("Debug: 1 %s", data.shape)
#                    logger.warning("Debug: 2 %s", batch_pt.shape)
#                    logger.warning("Debug: 3 %s", torch.sum(data.cpu() - batch_pt.cpu()).cpu().detach().item())
#                    logger.warning("Debug: 4 %s", torch.isclose(data.cpu(), batch_pt.cpu()).cpu().detach().all().item())
#                    logger.warning("Debug: 5 %s %s %s", data.numel(), torch.sum(torch.isclose(data.cpu(), batch_pt.cpu()).cpu().detach()).item(), torch.sum(torch.isclose(data.cpu(), batch_pt.cpu()).cpu().detach()).item() * 100 / data.numel())
            else:
                data = batch_pt.to(device)

            if batch_lm is not None:
                assert lang_model is not None
                assert lm_frozen_params

                data_lm = batch_lm.to(device)
            else:
                if lang_model:
                    assert not lm_frozen_params

                    batch = [f"{_src}{lang_model_tokenizer.sep_token}{_trg}" for _src, _trg in zip(src, trg)]
                    classifier_token = get_lang_model_cls_token(batch, lang_model, lang_model_tokenizer, device, max_length_encoder, to_cpu=False, detach=False)
                    data_lm = classifier_token.to(device)

            target = torch.tensor(labels).to(device)
            result = apply_inference(model, data, target=target, loss_function=loss_function, loss_apply_sigmoid=loss_apply_sigmoid, threshold=threshold, data_lm=data_lm) #, debug_labels=target if debug else None)
            _loss = result["loss"]

            assert len(_loss.shape) == 1, _loss.shape

            loss_elements += _loss.numel()

            if final_loss is None:
                final_loss = torch.sum(_loss)
            else:
                final_loss += torch.sum(_loss)

            # loss
            if batch_idx % gradient_accumulation == 0 or batch_idx == training_steps_per_epoch:
                assert final_loss is not None

                loss = final_loss / loss_elements
                final_loss = None
                loss_elements = 0

                epoch_loss.append(loss.cpu().detach().item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if debug and batch_idx % 50 == 0:
                    # Weights classifier
                    classifier_weight = model.classifier.weight
                    classifier_bias = model.classifier.bias

                    assert len(classifier_weight.shape) == 2
                    assert classifier_weight.shape[0] == 1, classifier_weight.shape
                    assert classifier_weight.shape[1] > 10, classifier_weight.shape # logging purposes
                    assert len(classifier_bias.shape) == 1
                    assert classifier_bias.shape[0] == 1

                    logger.debug("Classifier weight (first and last 5) and bias: %s ... %s (abs sum: %s) | %s", classifier_weight[0,:5].cpu().detach().tolist(), classifier_weight[0,-5:].cpu().detach().tolist(), torch.sum(classifier_weight.abs()).cpu().detach().item(), classifier_bias[0].cpu().detach().tolist())

                    # Grad
                    _model_grad_sum = sum([p.grad.sum().item() for p in model.parameters() if p.grad is not None])
                    _model_lm_grad_sum = sum([p.grad.sum().item() for p in lang_model.parameters() if p.grad is not None and lang_model is not None])
                    _model_non_lm_grad_sum = sum([p.grad.sum().item() for n, p in model.named_parameters() if not n.startswith("lang_model.") and p.grad is not None])
                    _model_projection_grad_sum = sum([p.grad.sum().item() for p in model.projection.parameters() if model.projection is not None and p.grad is not None])
                    #_model_projection_lm_grad_sum = sum([p.grad.sum().item() for p in model.lm_projection.parameters() if model.lm_projection is not None and p.grad is not None])
                    _model_projection_lm_grad_sum = 0.0

                    logger.debug("Grad sum (model, lm, model without lm, projection, projection lm): %s %s %s %s %s", _model_grad_sum, _model_lm_grad_sum, _model_non_lm_grad_sum, _model_projection_grad_sum, _model_projection_lm_grad_sum)

                optimizer.step()
                scheduler.step()

                model.zero_grad()

            if (batch_idx % (log_steps * gradient_accumulation)) == 0:
                sum_partial_loss = sum(epoch_loss[-1 * log_steps:]) # no: -1 * log_steps * gradient_accumulation!
                sum_loss = sum(epoch_loss)

                logger.info("Batch #%d: %s (last %d steps: %s)", batch_idx, sum_loss, log_steps * gradient_accumulation, sum_partial_loss)

                sys.stdout.flush()

        assert batch_idx == training_steps_per_epoch, f"{batch_idx} vs {training_steps_per_epoch}"

        sum_epoch_loss = sum(epoch_loss)

        logger.info("Epoch loss: %s", sum_epoch_loss)

        assert str(sum_epoch_loss) != "nan", "Some values in the input data are NaN"

        sys.stdout.flush()

        epoch += 1
        do_training = epoch < epochs or train_until_patience

    if not do_inference and save_model_path:
        logger.info("Loading best model: %s", save_model_path)

        model_state_dict = torch.load(save_model_path, weights_only=True, map_location=device)

        assert model.state_dict().keys() == model_state_dict.keys()

        model.load_state_dict(model_state_dict)

        model = model.to(device)

    if not model_inference_skip_train:
        train_results = eval(model, translation_model, make_batches(train_data, batch_size, pt_data=train_pickle_data, lm_data=train_lm_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, translation_tokenizer, max_length, max_new_tokens, threshold=threshold, layer=pretrained_model_layer, lang_model=lang_model, lang_model_tokenizer=lang_model_tokenizer, lm_frozen_params=lm_frozen_params, max_length_encoder=max_length_encoder, debug=debug)

        logger.info("Final train eval: %s", train_results)
    else:
        logger.info("Final train eval: skip")

    dev_results = eval(model, translation_model, make_batches(dev_data, batch_size, pt_data=dev_pickle_data, lm_data=dev_lm_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, translation_tokenizer, max_length, max_new_tokens, threshold=threshold, layer=pretrained_model_layer, lang_model=lang_model, lang_model_tokenizer=lang_model_tokenizer, lm_frozen_params=lm_frozen_params, max_length_encoder=max_length_encoder, debug=debug)

    logger.info("Final dev eval: %s", dev_results)

    if not skip_test_eval:
        test_results = eval(model, translation_model, make_batches(test_data, batch_size, pt_data=test_pickle_data, lm_data=test_lm_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, translation_tokenizer, max_length, max_new_tokens, threshold=threshold, layer=pretrained_model_layer, lang_model=lang_model, lang_model_tokenizer=lang_model_tokenizer, lm_frozen_params=lm_frozen_params, max_length_encoder=max_length_encoder, debug=debug)

        logger.info("Final test eval: %s", test_results)
    else:
        logger.info("Final test eval: skip")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="MTDetect classifier using NLLB hidden states (+LM)")

    lr_scheduler_conf = utils.get_options_from_argv("--lr-scheduler", "inverse_sqrt_chichirau_et_al", utils.argparse_pytorch_conf.lr_scheduler_args)
    optimizer_conf = utils.get_options_from_argv("--optimizer", "adamw_no_wd", utils.argparse_pytorch_conf.optimizer_args)

    # Mandatory
    parser.add_argument('dataset_train_filename', type=str, help="Filename with train data (TSV format). Format: original text (OT), machine (MT) or human (HT) translation, 0 if MT or 1 if HT")
    parser.add_argument('dataset_dev_filename', type=str, help="Filename with dev data (TSV format)")
    parser.add_argument('dataset_test_filename', type=str, help="Filename with test data (TSV format)")

    # Optional params
    parser.add_argument('--pickle-train-filename', type=str, default='', help="Pickle filename with train data. The order and batch size is expected to match with the provided flags")
    parser.add_argument('--pickle-dev-filename', type=str, default='', help="Pickle filename with dev data. The order and batch size is expected to match with the provided flags")
    parser.add_argument('--pickle-test-filename', type=str, default='', help="Pickle filename with test data. The order and batch size is expected to match with the provided flags")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size. Elements which will be processed before proceed to train")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs")
    parser.add_argument('--pretrained-model', default="facebook/nllb-200-distilled-600M", help="Pretrained translation model to calculate hidden states (not used if pickle files are provided)")
    parser.add_argument('--pretrained-model-target-layer', type=int, default=None,
                        help="Pretrained translation model hidden state layer to use in order to train the classifier (not used if pickle files are provided)")
    parser.add_argument('--lm-pretrained-model', help="Pretrained language model (encoder-like) to train together with classifier") # default empty: means NO lm
    parser.add_argument('--max-length-tokens', type=int, default=512, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', type=str, default='', help="Classifier input path which will be loaded")
    parser.add_argument('--model-output', type=str, default='', help="Classifier output path where the model will be stored")
    parser.add_argument('--inference', action="store_true", help="Do not train, just apply inference to the train, dev and test files")
    parser.add_argument('--patience', type=int, default=6,
                        help="Patience to stop training. If the specified value is greater than 0, epochs and patience will be taken into account")
    parser.add_argument('--train-until-patience', action="store_true",
                        help="Train until patience value is reached (--epochs will be ignored in order to stop, but will still be "
                             "used for other actions like LR scheduler)")
    parser.add_argument('--lm-model-input', help="Encoder-like model input path where the model will be stored")
    parser.add_argument('--learning-rate', type=float, default=1e-04, help="Classifier learning rate")
    parser.add_argument('--lm-frozen-params', action='store_true', help="Freeze encoder-like model parameters (i.e., do not train)")
    parser.add_argument('--lm-learning-rate', type=float, default=1e-5, help="Encoder-like model learning rate")
    parser.add_argument('--num-layers', type=int, default=3, help="Classifier layers")
    parser.add_argument('--num-attention-heads', type=int, default=4, help="Classifier attention heads")
    parser.add_argument('--source-lang', type=str, required=True, help="NLLB source language (e.g., eng_Latn)")
    parser.add_argument('--target-lang', type=str, required=True, help="NLLB target language")
    parser.add_argument('--direction', type=str, choices=["src2trg", "trg2src"], default="src2trg", help="Translation direction. Providing several values is supported")
    parser.add_argument('--optimizer', choices=optimizer_conf["choices"], default=optimizer_conf["default"], help="Optimizer")
    parser.add_argument('--optimizer-args', **optimizer_conf["options"],
                        help="Args. for the optimizer (in order to see the specific configuration for a optimizer, use -h and set --optimizer)")
    parser.add_argument('--lr-scheduler', choices=lr_scheduler_conf["choices"], default=lr_scheduler_conf["default"], help="LR scheduler")
    parser.add_argument('--lr-scheduler-args', **lr_scheduler_conf["options"],
                        help="Args. for LR scheduler (in order to see the specific configuration for a LR scheduler, "
                             "use -h and set --lr-scheduler)")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Gradient accumulation steps")
#    parser.add_argument('--multiplicative-inverse-temperature-sampling', type=float, default=0.3, help="See https://arxiv.org/pdf/1907.05019 (section 4.2). Default value has been set the one used in the NLLB paper")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout applied to the classifier model (embedding, model, and head)")
    parser.add_argument('--lm-dropout', type=float, default=0.5, help="Dropout applied to the LM")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold to consider a given text to be HT")
    parser.add_argument('--dev-patience-metric', type=str, choices=["acc", "macro_f1"], default="acc", help="Metric to calculate patience using the dev set")
    parser.add_argument('--skip-training-set-during-inference', action="store_true", help="Skip training evaluation during inference to speed up result")
    parser.add_argument('--skip-test-set-eval', action="store_true", help="Skip test evaluation during training or inference")
    parser.add_argument('--data-limit', type=int, default=None, help="Data limit reading in batches (debug purposes)")

    parser.add_argument('--seed', type=int, default=71213,
                        help="Seed in order to have deterministic results (not fully guaranteed). "
                             "Set a negative number in order to disable this feature")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--debug', action="store_true", help="Debug purposes")

    args = parser.parse_args()

    return args

def cli():
    global logger

    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    # Logging
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: %s", str(args)) # First logging message should be the processed arguments

    main(args)

if __name__ == "__main__":
    cli()
