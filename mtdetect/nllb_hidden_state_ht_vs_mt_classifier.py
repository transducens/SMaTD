
import sys
import gzip
import math
import pickle

del sys.path[0] # remove wd path

import mtdetect.inference as inference

import torch
import torch.nn as nn
import transformers
import numpy as np

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

    def __init__(self, d_model, nhead, dim_feedforward, nlayers, projection_in=None, max_seq_len=512, embedding_dropout=0.5, dropout_p=0.5, classifier_dropout_p=0.5, num_labels=1):
        super(TransformerModel, self).__init__()

        initial_dim = d_model if projection_in is None else projection_in
        self.pos_encoder = PositionalEncoding(initial_dim, embedding_dropout, max_seq_len=max_seq_len)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout_p) #, activation="gelu", layer_norm_eps=1e-12)
        self.projection = None if projection_in is None else nn.Linear(projection_in, d_model)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.initializer_range = 0.02
        # classifier
        self.pooler = nn.Linear(d_model, d_model) # https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/models/bert/modeling_bert.py#L737
        self.pooler_activation = nn.Tanh()
        self.classifier_dropout = nn.Dropout(classifier_dropout_p)
        self.classifier = nn.Linear(d_model, num_labels)

        self.init_weights()

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

    def forward(self, src, mask=None):
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        proj = self.projection(src) if self.projection is not None else src
        output = self.transformer_encoder(proj, mask=mask)

        # https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/models/bert/modeling_bert.py#L1680
        output = output[:,0,:] # Only the first token (CLS)
        output = self.classifier_dropout(output)
        output = self.pooler(output) 
        output = self.pooler_activation(output)
        output = self.classifier_dropout(output)
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

def read_pickle(fn, k=None, limit=None):
    data = None
    open_func = gzip.open if fn.endswith(".gz") else open

    print(f"Loading pickle file: {fn} (key: {k})")

    with open_func(fn, "rb") as fd:
        data = pickle.load(fd)

    assert isinstance(data, dict), type(data)

    for _k1 in data.keys():
#        assert isinstance(data[_k1], dict), type(data[_k1])
        assert isinstance(data[_k1], list), type(data[_k1])
        assert isinstance(data[_k1][0], torch.Tensor), type(data[_k1][0])

#        for _k2 in data[_k1].keys():
#            assert isinstance(data[_k1][_k2], list), type(data[_k1][_k2])
#            assert isinstance(data[_k1][_k2][0], torch.Tensor), type(data[_k1][_k2][0])

    if limit is not None:
        for _k1 in data.keys():
            data[_k1] = data[_k1][:limit]
#            for _k2 in data[_k1].keys():
#                data[_k1][_k2] = data[_k1][_k2][:limit]

    if k is not None:
        data = data[k]

    return data

def make_batches(data, bsz, pt_data=None):
    assert bsz > 0

    idx = 0
    batch = []

    for d in data:
        batch.append(d)

        if len(batch) >= bsz:
            if pt_data is None:
                yield (batch, None)
            else:
                assert pt_data[idx].shape[0] == len(batch)

                yield (batch, pt_data[idx])

            idx += 1
            batch = []

    if len(batch) > 0:
        if pt_data is None:
            yield (batch, None)
        else:
            assert pt_data[idx].shape[0] == len(batch)

            yield (batch, pt_data[idx])

        idx += 1
        batch = []

def apply_inference(model, data, mask=None, target=None, loss_function=None, threshold=0.5, loss_apply_sigmoid=False):
    model_outputs = model(data, mask=mask)
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

def eval(model, translation_model, data_generator, direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, print_result=False, print_desc='-', threshold=0.5, layer=-1):
    assert not print_result, "Code not working"

    training = model.training

    model.eval()

    all_outputs = []
    all_labels = []
    print_idx = 0

    for batch, batch_pt in data_generator:
        src, trg, labels = zip(*batch)

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

        target = torch.tensor(labels).to(device)
        results = apply_inference(model, data, target=None, loss_function=None, threshold=threshold, loss_apply_sigmoid=False)
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

    return results

train_fn = sys.argv[1].split(':')
dev_fn = sys.argv[2].split(':')
test_fn = sys.argv[3].split(':')
_src_lang = sys.argv[4]
_trg_lang = sys.argv[5]
direction = sys.argv[6]
batch_size = int(sys.argv[7]) if len(sys.argv) > 7 else 32
save_model_path = sys.argv[8] if len(sys.argv) > 8 else None
learning_rate = float(sys.argv[9]) if len(sys.argv) > 9 else 5e-4
num_layers = int(sys.argv[10]) if len(sys.argv) > 10 else 1
nhead = int(sys.argv[11]) if len(sys.argv) > 11 else 4
load_model_path = sys.argv[12] if len(sys.argv) > 12 else None
do_inference = bool(int(sys.argv[13])) if len(sys.argv) > 13 else False
pretrained_model = sys.argv[14] if len(sys.argv) > 14 and len(sys.argv[14]) > 0 else "facebook/nllb-200-distilled-600M"
pretrained_model_layer = int(sys.argv[15]) if len(sys.argv) > 15 and len(sys.argv[15]) > 0 else -1

train_fn, train_pickle_fn = train_fn[:2] if len(train_fn) > 1 else (train_fn[0], None)
dev_fn, dev_pickle_fn = dev_fn[:2] if len(dev_fn) > 1 else (dev_fn[0], None)
test_fn, test_pickle_fn = test_fn[:2] if len(test_fn) > 1 else (test_fn[0], None)

print(f"Learning rate: {learning_rate}")

if save_model_path:
    print(f"Model save path: {save_model_path}")

if load_model_path:
    print(f"Model load path: {load_model_path}")

print(f"Number of layers: {num_layers}")
print(f"Number of heads: {nhead}")
print(f"Pretrained model: {pretrained_model}")
print(f"Pretrained model layer: {pretrained_model_layer}")

# read data
limit = None
train_data = read(train_fn, limit=None if limit is None else limit * batch_size)
dev_data = read(dev_fn, limit=None if limit is None else limit * batch_size)
test_data = read(test_fn, limit=None if limit is None else limit * batch_size)
train_pickle_data = read_pickle(train_pickle_fn, k="decoder_last_hidden_state", limit=limit) if train_pickle_fn is not None else None
dev_pickle_data = read_pickle(dev_pickle_fn, k="decoder_last_hidden_state", limit=limit) if dev_pickle_fn is not None else None
test_pickle_data = read_pickle(test_pickle_fn, k="decoder_last_hidden_state", limit=limit) if test_pickle_fn is not None else None
all_pickle_data_loaded = train_pickle_fn is not None and dev_pickle_fn is not None and test_pickle_fn is not None

for _p, _d in ((train_pickle_data, train_data), (dev_pickle_data, dev_data), (test_pickle_data, test_data)):
    if _p:
        s = 0

        for d in _p:
            assert isinstance(d, torch.Tensor), type(d)

            s += d.shape[0]

        assert s == len(_d)

for p in (train_pickle_data, dev_pickle_data, test_pickle_data):
    if p:
        assert isinstance(p, list), type(p)
        assert isinstance(p[0], torch.Tensor), type(test_pickle_data[0])

print(f"Train: {len(train_data)}")
print(f"Dev: {len(dev_data)}")
print(f"test: {len(test_data)}")

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
max_length = translation_model.config.max_length
max_new_tokens = translation_model.generation_config.max_length
eos_token_token = translation_tokenizer.convert_ids_to_tokens(translation_model.generation_config.eos_token_id)
decoder_start_token_token = translation_tokenizer.convert_ids_to_tokens(translation_model.generation_config.decoder_start_token_id)
max_seq_len = max_new_tokens

# classifier
#projection_in = translation_model.config.d_model # 1024 for facebook/nllb-200-distilled-600M
projection_in = None
#d_model = 512
#d_model = 128
d_model = translation_model.config.d_model
model = TransformerModel(d_model, nhead, dim_feedforward, num_layers,
                         projection_in=projection_in, max_seq_len=max_seq_len,
                         embedding_dropout=0.1, dropout_p=0.1, classifier_dropout_p=0.1)

if load_model_path:
    print(f"Loading init model: {load_model_path}")

    model_state_dict = torch.load(load_model_path, weights_only=True, map_location=device)

    assert model.state_dict().keys() == model_state_dict.keys()

    model.load_state_dict(model_state_dict)

if do_inference:
    model = model.eval()
else:
    model = model.train()

model = model.to(device)

#print(model)

# classifier args
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(model_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
scheduler = transformers.get_inverse_sqrt_schedule(optimizer, 400)

# training args
patience = 6
current_patience = 0
epochs = -1
epoch = 0
do_training = not do_inference and (epochs < 0 or epoch < epochs)
loss_function = nn.BCEWithLogitsLoss()
loss_apply_sigmoid = False # Should be True if loss_function = nn.BCELoss()
threshold = 0.5
log_steps = 100
sum_epoch_loss = np.inf
early_stopping_best_loss = np.inf
patience_metric = "acc"
early_stopping_best_result_dev = -np.inf # accuracy
early_stopping_best_result_train = -np.inf # accuracy

while do_training:
    epoch_loss = []

    print(f"Epoch #{epoch + 1}")

    #train_results = eval(model, translation_model, make_batches(train_data, batch_size), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, threshold=threshold)
    dev_results = eval(model, translation_model, make_batches(dev_data, batch_size, pt_data=dev_pickle_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, threshold=threshold, layer=pretrained_model_layer)
    better_loss_result = False

    if len(epoch_loss) > 0 and sum_epoch_loss < early_stopping_best_loss:
        print(f"Better loss result: {early_stopping_best_loss} -> {sum_epoch_loss}")

        better_loss_result = True
        early_stopping_best_loss = sum_epoch_loss

    #print(f"Train eval: {train_results}")
    print(f"Dev eval: {dev_results}")

    #early_stopping_metric_train = train_results[patience_metric]
    early_stopping_metric_dev = dev_results[patience_metric]
    better_train_result = False
    patience_dev_equal = np.isclose(early_stopping_metric_dev, early_stopping_best_result_dev)
    #patience_train_equal = np.isclose(early_stopping_metric_train, early_stopping_best_result_train)

    #if early_stopping_metric_train > early_stopping_best_result_train:
    #    print(f"Better train result (metric: {patience_metric}): {early_stopping_best_result_train} -> {early_stopping_metric_train}")

    #    better_train_result = True
    #    early_stopping_best_result_train = early_stopping_metric_train

    #if early_stopping_metric_dev > early_stopping_best_result_dev or ((patience_dev_equal and better_train_result) or (patience_dev_equal and patience_train_equal and better_loss_result)):
    if early_stopping_metric_dev > early_stopping_best_result_dev:
        print(f"Patience better dev result (metric: {patience_metric}): {early_stopping_best_result_dev} -> {early_stopping_metric_dev}")

        current_patience = 0
        early_stopping_best_result_dev = early_stopping_metric_dev

        if save_model_path:
            print(f"Saving best model: {save_model_path}")

            torch.save(model.state_dict(), save_model_path)
    elif patience > 0:
        current_patience += 1

        print(f"Exhausting patience... {current_patience} / {patience}")

    if patience > 0 and current_patience >= patience:
        print("Patience is over ...")

        do_training = False

        break # we need to force the break to avoid the training of the current epoch

    model.zero_grad()

    for bsz_idx, (batch, batch_pt) in enumerate(make_batches(train_data, batch_size, pt_data=train_pickle_data), 1):
        src, trg, labels = zip(*batch)

        if direction == "trg2src":
            src, trg = trg, src

        if batch_pt is None:
            src = [f"{source_lang_token} {_src}{eos_token_token}" for _src in src]
            trg = [f"{decoder_start_token_token}{target_lang_token} {_trg}{eos_token_token}" for _trg in trg]
            src_inputs = preprocess(src, translation_tokenizer, device, max_length)
            trg_inputs = preprocess(trg, translation_tokenizer, device, max_new_tokens)
            translation_output = get_model_last_hidden_state(translation_model, src_inputs, trg_inputs, skip_modules=("encoder",), to_cpu=False, layer=pretrained_model_layer)
            data = translation_output["decoder_last_hidden_state"]
        else:
            data = batch_pt.to(device)

        target = torch.tensor(labels).to(device)

        # loss
        model.zero_grad()

        result = apply_inference(model, data, target=target, loss_function=loss_function, loss_apply_sigmoid=loss_apply_sigmoid, threshold=threshold)
        loss = result["loss"]

        epoch_loss.append(loss.cpu().detach().item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (bsz_idx % log_steps) == 0:
            sum_partial_loss = sum(epoch_loss[-100:])
            sum_loss = sum(epoch_loss)

            print(f"Batch #{bsz_idx}: {data.shape} -> {result['outputs'].shape}: {sum_loss} (last {log_steps} steps: {sum_partial_loss})")

            sys.stdout.flush()

    sum_epoch_loss = sum(epoch_loss)

    print(f"Epoch loss: {sum_epoch_loss}")

    assert str(sum_epoch_loss) != "nan", "Some values in the input data are NaN"

    sys.stdout.flush()

    epoch += 1
    do_training = epochs < 0 or epoch < epochs

model_inference_skip_train = False
skip_test = False

if not do_inference and save_model_path:
    print(f"Loading best model: {save_model_path}")

    model_state_dict = torch.load(save_model_path, weights_only=True, map_location=device)

    assert model.state_dict().keys() == model_state_dict.keys()

    model.load_state_dict(model_state_dict)

    model = model.to(device)

if not model_inference_skip_train:
    train_results = eval(model, translation_model, make_batches(train_data, batch_size, pt_data=train_pickle_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, threshold=threshold, layer=pretrained_model_layer)

    print(f"Final train eval: {train_results}")
else:
    print("Final train eval: skip")

dev_results = eval(model, translation_model, make_batches(dev_data, batch_size, pt_data=dev_pickle_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, threshold=threshold, layer=pretrained_model_layer)

print(f"Final dev eval: {dev_results}")

if not skip_test:
    test_results = eval(model, translation_model, make_batches(test_data, batch_size, pt_data=test_pickle_data), direction, device, source_lang_token, target_lang_token, decoder_start_token_token, eos_token_token, threshold=threshold, layer=pretrained_model_layer)

    print(f"Final test eval: {test_results}")
else:
    print(f"Final test eval: skip")

#custom_embeddings = torch.rand(8, 128, d_model)  # Example shape
#custom_embeddings = custom_embeddings.to(device)
#output = model(custom_embeddings)

#print(output.shape)
