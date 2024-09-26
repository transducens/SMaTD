
import os
import sys
import json
import copy
import pickle

import mtdetect.transformer_mm_explainability.example_translation_nllb as example_translation_nllb
import mtdetect.inference as inference
import mtdetect.utils.utils as utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers

print(f"Provided args: {sys.argv}")

seed = int(os.environ["MTDETECT_RANDOM_SEED"]) if "MTDETECT_RANDOM_SEED" in os.environ else np.random.randint(2 ** 32 - 1)

print(f"Random seed: {seed}")

utils.init_random_with_seed(seed)

force_pickle_file = True # Change manually
skip_test = False # Change manually
disable_cnn = "MTDETECT_DISABLE_CNN" in os.environ and bool(int(os.environ["MTDETECT_DISABLE_CNN"]))
model_inference_skip_train = "MTDETECT_MODEL_INFERENCE_SKIP_TRAIN" in os.environ and bool(int(os.environ["MTDETECT_MODEL_INFERENCE_SKIP_TRAIN"]))

def default_sys_argv(n, default, f=str):
    return f(sys.argv[n]) if len(sys.argv) > n else default

train_filename = sys.argv[1]
dev_filename = sys.argv[2]
test_filename = sys.argv[3]
batch_size = default_sys_argv(4, 32, f=int)
batch_size = batch_size if batch_size > 0 else 32
cnn_max_width = default_sys_argv(5, 64, f=int)
cnn_max_width = cnn_max_width if cnn_max_width > 0 else 64
cnn_max_height = default_sys_argv(6, 64, f=int)
cnn_max_height = cnn_max_height if cnn_max_height > 0 else 64
source_lang = default_sys_argv(7, "eng_Latn")
source_lang = source_lang if source_lang != '' else "eng_Latn"
target_lang = default_sys_argv(8, "spa_Latn")
target_lang = target_lang if target_lang != '' else "spa_Latn"
direction = default_sys_argv(9, ["src2trg"], f=lambda q: ["src2trg"] if q == '' else [_q for _q in q.split('+')])
attention_matrix = default_sys_argv(10, "encoder+decoder+cross")
attention_matrix = attention_matrix.split('+') # The order is not important (i.e., cross+decoder should be equivalent to decoder+cross) -> sorted
explainability_normalization = default_sys_argv(11, "none")
self_attention_remove_diagonal = default_sys_argv(12, True, f=lambda q: bool(int(q)))
cnn_pooling = default_sys_argv(13, "avg+max+avg")
cnn_pooling = cnn_pooling.split('+')
save_model_path = default_sys_argv(14, '')
learning_rate = default_sys_argv(15, 5e-3, f=float)
multichannel = default_sys_argv(16, True, f=lambda q: bool(int(q)))
pretrained_model = default_sys_argv(17, '')
teacher_forcing = default_sys_argv(18, [False], f=lambda q: [False] if q == '' else [True if _q == "yes" else (False if _q == "no" else bool(int(_q))) for _q in q.split('+')])
ignore_attention = default_sys_argv(19, [False], f=lambda q: [False] if q == '' else [True if _q == "yes" else (False if _q == "no" else bool(int(_q))) for _q in q.split('+')])
lm_pretrained_model = default_sys_argv(20, None)
lm_model_input = default_sys_argv(21, None, f=lambda q: None if q == '' else q)
lm_frozen_params = default_sys_argv(22, True, f=lambda q: True if q == '' else bool(int(q)))
lm_learning_rate = default_sys_argv(23, 1e-5, f=float)
cnn_model_input = default_sys_argv(24, '')
model_inference = default_sys_argv(25, False, f=lambda q: False if q == '' else bool(int(q)))
gradient_accumulation = default_sys_argv(26, 1, f=lambda q: 1 if q == '' else int(q))

assert gradient_accumulation > 0, gradient_accumulation

print(f"Gradient accumulation: {gradient_accumulation} (if enabled, i.e. > 1, the same results would be obtained if dropout is disabled for both CNN and LM, train shuffle is disabled, and if float precision errors are ignored)")

model_inference_skip_train = model_inference_skip_train and model_inference

if model_inference:
    lm_frozen_params = True

if lm_pretrained_model:
    print(f"LM is going to be used: {lm_pretrained_model} (local file: {lm_model_input})")

if lm_pretrained_model and not lm_model_input and lm_frozen_params and not model_inference:
    print(f"warning: LM provided but it is not a fine-tuned model and its parameters are frozen: the format is src<sep>trg, and the output is the first output token, which might not be the expected behaviour for the model")

def load_model(model_input, pretrained_model, device, classifier_dropout=0.0):
    local_model = model_input is not None
    config = transformers.AutoConfig.from_pretrained(pretrained_model, num_labels=1, classifier_dropout=classifier_dropout)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
    tokenizer = utils.get_tokenizer(pretrained_model)

    if local_model:
        state_dict = torch.load(model_input, weights_only=True, map_location=device) # weights_only: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
                                                                                     # map_location: avoid creating a new process and using additional and useless memory

        model.load_state_dict(state_dict)

    model = model.to(device)

    return model, tokenizer

if skip_test:
    print(f"warning: test set evaluation is disabled")

for _attention_matrix in attention_matrix:
    assert _attention_matrix in ("encoder", "decoder", "cross"), attention_matrix

for _cnn_pooling in cnn_pooling:
    assert _cnn_pooling in ("max", "avg"), cnn_pooling

for _direction in direction:
    assert _direction in ("src2trg", "trg2src"), _direction

assert len(cnn_pooling) in (1, len(attention_matrix)), cnn_pooling
assert explainability_normalization in ("none", "absolute", "relative"), explainability_normalization
assert cnn_max_width > 0, cnn_max_width
assert cnn_max_height > 0, cnn_max_height

attention_matrix = [f"explainability_{_attention_matrix}" for _attention_matrix in attention_matrix]
translation_model_conf = {
    "source_lang": source_lang,
    "target_lang": target_lang,
    "direction": direction,
    "attention_matrix": attention_matrix,
    "explainability_normalization": explainability_normalization,
    "self_attention_remove_diagonal": self_attention_remove_diagonal,
    "teacher_forcing": teacher_forcing,
    "ignore_attention": ignore_attention,
}
translation_model_conf = json.dumps(translation_model_conf, indent=4)

print(f"NLLB conf:\n{translation_model_conf}")

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier_dropout = 0.1
cnn_dropout = 0.5
shuffle_training = True and not model_inference
lang_model, tokenizer = load_model(lm_model_input, lm_pretrained_model, None, classifier_dropout=classifier_dropout) if lm_pretrained_model else (None, None)

def extend_tensor_with_zeros_and_truncate(t, max_width, max_height, device):
    assert len(t.shape) == 2

    result = torch.zeros((max_width, max_height)).to(device)
    result[:t.shape[0], :t.shape[1]] = t[:max_width, :max_height]

    return result

def read(filename, direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
         focus=["explainability_cross"], store_explainability_arrays=True, load_explainability_arrays=True,
         device=None, pretrained_model=None, pickle_template=None, pickle_check_env=True, teacher_forcing=False,
         ignore_attention=False, force_pickle_file=False):
    cnn_width = -np.inf
    cnn_height = -np.inf
    loaded_samples = 0
    source_text = []
    target_text = []
    labels = []
    groups = [] # If more than 1 entry belongs to the same group, they will be randomly selected dynamically
    uniq_groups = set()
    explainability_ee = []
    explainability_dd = []
    explainability_de = []
    fd = open(filename)
    fn_pickle_array = None
    limit_data = np.inf if "MTDETECT_LIMIT_DATA" not in os.environ else int(os.environ["MTDETECT_LIMIT_DATA"])
    limit_data = np.inf if limit_data <= 0 else limit_data

    if force_pickle_file:
        assert load_explainability_arrays

    if pickle_check_env:
        envvar_prefix = "MTDETECT_PICKLE_FN"
        envvar = envvar_prefix

        if pickle_template:
            envvar = f"{envvar_prefix}_{pickle_template}"

            if envvar not in os.environ:
                envvar = envvar_prefix

        if envvar in os.environ:
            fn_pickle_array = os.environ[envvar]

            if envvar == envvar_prefix:
                if pickle_template:
                    fn_pickle_array = fn_pickle_array.replace("{template}", pickle_template)

                fn_pickle_array = fn_pickle_array.replace("{direction}", direction)
                fn_pickle_array = fn_pickle_array.replace("{teacher_forcing}", "yes" if teacher_forcing else "no")
                fn_pickle_array = fn_pickle_array.replace("{ignore_attention}", "yes" if ignore_attention else "no")

    fn_pickle_array_exists = os.path.isfile(fn_pickle_array)

    if not fn_pickle_array or not fn_pickle_array_exists:
        if fn_pickle_array:
            print(f"warning: provided envvar pickle path, but we could not find it: {fn_pickle_array}")

        teacher_forcing_str = "yes" if teacher_forcing else "no"
        ignore_attention_str = "yes" if ignore_attention else "no"
        fn_pickle_array = f"{filename}.{direction}.{source_lang}.{target_lang}.teacher_forcing_{teacher_forcing_str}.ignore_attention_{ignore_attention_str}.pickle"

    fn_pickle_array_exists = os.path.isfile(fn_pickle_array)

    assert fn_pickle_array_exists if force_pickle_file else True, f"Pickle file not found: {fn_pickle_array}"

    if load_explainability_arrays and fn_pickle_array_exists:
        print(f"Loading explainability arrays: {fn_pickle_array}")

        with open(fn_pickle_array, "rb") as pickle_fd:
            pickle_data = pickle.load(pickle_fd)
            explainability_ee = pickle_data["explainability_encoder"][:None if limit_data == np.inf else limit_data]
            explainability_dd = pickle_data["explainability_decoder"][:None if limit_data == np.inf else limit_data]
            explainability_de = pickle_data["explainability_cross"][:None if limit_data == np.inf else limit_data]

    for idx, l in enumerate(fd):
        l = l.rstrip("\r\n").split('\t')
        source = l[0]
        target = l[1]
        label = l[2]
        group = l[3] if len(l) > 3 else str(idx)

        assert label in ('0', '1'), label # 0 is NMT; 1 is HT

        label = float(label)

        source_text.append(source)
        target_text.append(target)
        labels.append(label)
        groups.append(group)
        uniq_groups.add(group)

        if not fn_pickle_array_exists:
            input_tokens, output_tokens, output, r_ee, r_dd, r_de = \
                example_translation_nllb.explainability(source, target_text=target, source_lang=source_lang, target_lang=target_lang,
                                                        debug=False, apply_normalization=True, self_attention_remove_diagonal=False,
                                                        explainability_normalization="none", device=device, pretrained_model=pretrained_model,
                                                        teacher_forcing=teacher_forcing, ignore_attention=ignore_attention, direction=direction,
                                                        print_sentences_info=first_msg)
            first_msg = False

            explainability_ee.append(r_ee)
            explainability_dd.append(r_dd)
            explainability_de.append(r_de)
        else:
            r_ee = explainability_ee[idx]
            r_dd = explainability_dd[idx]
            r_de = explainability_de[idx]

        ##### code from example_translation_nllb.py #####
        if self_attention_remove_diagonal:
            np.fill_diagonal(r_ee, sys.float_info.epsilon)
            np.fill_diagonal(r_dd, sys.float_info.epsilon)

        if explainability_normalization == "none":
            pass
        elif explainability_normalization == "absolute":
            r_ee = (r_ee - r_ee.min()) / (r_ee.max() - r_ee.min())
            r_dd = (r_dd - r_dd.min()) / (r_dd.max() - r_dd.min())
            r_de = (r_de - r_de.min()) / (r_de.max() - r_de.min()) # (target_text_seq_len, source_text_seq_len)
        elif explainability_normalization == "relative":
            # "Relative" normalization (easier to analize per translated token)
            r_ee = np.array([(r_ee[i] - r_ee[i].min()) / (r_ee[i].max() - r_ee[i].min()) for i in range(len(r_ee))])
            r_dd = np.array([(r_dd[i] - r_dd[i].min()) / (r_dd[i].max() - r_dd[i].min()) for i in range(len(r_dd))])
            r_de = np.array([(r_de[i] - r_de[i].min()) / (r_de[i].max() - r_de[i].min()) for i in range(len(r_de))])
        ##### code from example_translation_nllb.py #####

        #print(f"{loaded_samples + 1} pairs loaded! {r_de.shape}")

        width = -np.inf
        height = -np.inf

        if "explainability_encoder" in focus:
            width = max(width, r_ee.shape[0])
            height = max(height, r_ee.shape[1])
        if "explainability_decoder" in focus:
            width = max(width, r_dd.shape[0])
            height = max(height, r_dd.shape[1])
        if "explainability_cross" in focus:
            width = max(width, r_de.shape[0])
            height = max(height, r_de.shape[1])

        assert width != -np.inf
        assert height != -np.inf

        cnn_width = cnn_max_width if cnn_max_width > 0 else max(cnn_width, width)
        cnn_height = cnn_max_height if cnn_max_height > 0 else max(cnn_height, height)
        loaded_samples += 1

        if loaded_samples % 100 == 0:
            print(f"{loaded_samples} samples loaded: {filename}")

            sys.stdout.flush()

        if idx + 1 >= limit_data:
            break

    fd.close()

    if load_explainability_arrays and fn_pickle_array_exists:
        # Explainability arrays were loaded
        expected_len = min(len(source_text), limit_data)

        assert expected_len == len(explainability_ee), f"{expected_len} != {len(explainability_ee)}"
        assert expected_len == len(explainability_dd), f"{expected_len} != {len(explainability_dd)}"
        assert expected_len == len(explainability_de), f"{expected_len} != {len(explainability_de)}"

    if store_explainability_arrays and not fn_pickle_array_exists:
        print(f"Storing explainability arrays: {fn_pickle_array}")

        with open(fn_pickle_array, "wb") as pickle_fd:
            pickle_data = {
                "explainability_encoder": explainability_ee,
                "explainability_decoder": explainability_dd,
                "explainability_cross": explainability_de,
            }

            pickle.dump(pickle_data, pickle_fd)

    print(f"Samples: {len(groups)} (limit: {limit_data}); Groups: {len(uniq_groups)}")

    return {
        "cnn_width": cnn_width,
        "cnn_height": cnn_height,
        "loaded_samples": loaded_samples,
        "source_text": source_text,
        "target_text": target_text,
        "labels": labels,
        "groups": groups,
        "explainability_encoder": explainability_ee,
        "explainability_decoder": explainability_dd,
        "explainability_cross": explainability_de,
    }

def get_data(explainability_matrix, labels, loaded_samples, cnn_width, cnn_height, device, convert_labels_to_tensor=True):
    inputs = []

    for _input in explainability_matrix:
        _input = torch.from_numpy(_input)

        assert len(_input.shape) == 2

        _input = extend_tensor_with_zeros_and_truncate(_input, cnn_width, cnn_height, None)
        _input = _input.tolist()

        inputs.append(_input)

    inputs = torch.tensor(inputs)
    inputs = inputs.unsqueeze(1).to(device) # channel dim

    if convert_labels_to_tensor:
        labels = torch.tensor(labels).to(device)

    inputs_expected_shape = (loaded_samples, 1, cnn_width, cnn_height)
    labels_expected_shape = (loaded_samples,)

    assert inputs.shape == inputs_expected_shape, inputs.shape
    assert labels.shape == labels_expected_shape, labels.shape

    return inputs, labels

channels_factor_len_set = set([len(direction), len(teacher_forcing), len(ignore_attention)])

assert len(channels_factor_len_set) in (1, 2), channels_factor_len_set

if len(channels_factor_len_set) == 2:
    assert 1 in channels_factor_len_set, channels_factor_len_set

if disable_cnn:
    assert lang_model, "disable_cnn does not support not providing lang_model"
    assert not cnn_model_input

    print("CNN disabled")

if multichannel:
    channels = 1
    channels_factor = 1
    cnn_pooling *= 1 if len(cnn_pooling) > 1 else len(attention_matrix)
    cnn_pooling *= max(channels_factor_len_set)

    if lang_model:
        cnn_pooling.append(cnn_pooling[0]) # fake value (it will be ignored) -> easier for further processing
else:
    # Expected: for each value provided to direction, teacher_forcing, and ignore_attention, we will have an extra set of len(attention_matrix) channels
    # Example: {direction: src2trg+trg2src, teacher_forcing: True+False, ignore_attention: False} -> [(src2trg, True, False), (trg2src, False, False)] # ignore_attention is expanded
    # Example: {direction: src2trg+src2trg+trg2src+trg2src, teacher_forcing: True+False+True+False, ignore_attention: False+True+False+True} -> [(src2trg, True, False), (src2trg, False, True), (trg2src, True, False), (trg2src, False, True)]
    channels = len(attention_matrix)
    channels_factor = max(channels_factor_len_set)

direction *= 1 if len(direction) > 1 else max(channels_factor_len_set)
teacher_forcing *= 1 if len(teacher_forcing) > 1 else max(channels_factor_len_set)
ignore_attention *= 1 if len(ignore_attention) > 1 else max(channels_factor_len_set)
channels *= channels_factor

print(f"Total channels: {channels} (factor: {channels_factor})")

if channels_factor > 1 and not force_pickle_file:
    print(f"warning: channels_factor={channels_factor} > 1, and force_pickle_file=False: it may be very slow to create all the pickle files if they do not exist (they may exist)")

cnn_width = -np.inf
cnn_height = -np.inf
data_input_all_keys = []
train_data = {}
dev_data = {}
test_data = {}

assert len(direction) == len(teacher_forcing) == len(ignore_attention)

for _direction, _teacher_forcing, _ignore_attention in zip(direction, teacher_forcing, ignore_attention):
    # TODO we are reading the files several times...

    _train_data = read(train_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                      focus=attention_matrix, device=device, pretrained_model=pretrained_model, pickle_template="train",
                      teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention, force_pickle_file=force_pickle_file)
    _dev_data = read(dev_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                     focus=attention_matrix, device=device, pretrained_model=pretrained_model, pickle_template="dev",
                     teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention, force_pickle_file=force_pickle_file)

    if skip_test:
        _test_data = {"cnn_width": -np.inf, "cnn_height": -np.inf}
    else:
        _test_data = read(test_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                          focus=attention_matrix, device=device, pretrained_model=pretrained_model, pickle_template="test",
                          teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention, force_pickle_file=force_pickle_file)

    train_data.update(_train_data)
    dev_data.update(_dev_data)
    test_data.update(_test_data)

    cnn_width = max(train_data["cnn_width"], dev_data["cnn_width"], test_data["cnn_width"], cnn_width)
    cnn_height = max(train_data["cnn_height"], dev_data["cnn_height"], test_data["cnn_height"], cnn_height)
    first_time = True
    _teacher_forcing_str = "yes" if _teacher_forcing else "no"
    _ignore_attention_str = "yes" if _ignore_attention else "no"

    for _attention_matrix in attention_matrix:
        inputs = f"{_attention_matrix}_{_direction}_{_teacher_forcing_str}_{_ignore_attention_str}"
        train_data[f"inputs_{inputs}"], train_data["labels"] = get_data(train_data[_attention_matrix], train_data["labels"], train_data["loaded_samples"], cnn_width, cnn_height, None, convert_labels_to_tensor=first_time)
        dev_data[f"inputs_{inputs}"], dev_data["labels"] = get_data(dev_data[_attention_matrix], dev_data["labels"], dev_data["loaded_samples"], cnn_width, cnn_height, None, convert_labels_to_tensor=first_time)

        if not skip_test:
            test_data[f"inputs_{inputs}"], test_data["labels"] = get_data(test_data[_attention_matrix], test_data["labels"], test_data["loaded_samples"], cnn_width, cnn_height, None, convert_labels_to_tensor=first_time)

        first_time = False

        assert inputs not in data_input_all_keys

        data_input_all_keys.append(inputs)

if lang_model:
    # Add LM data to inputs

    _max_length_encoder = 512 # TODO add user argument?
    max_length_encoder = utils.get_encoder_max_length(lang_model, tokenizer, max_length_tokens=_max_length_encoder)
    max_length_encoder = min(max_length_encoder, _max_length_encoder)
    data_input_all_keys.append("lm_inputs")

    print(f"Max length: {max_length_encoder}")

    for d, desc in ((train_data, "train"), (dev_data, "dev"), (test_data, "test")):
        if desc == "test" and skip_test:
            continue

        assert "inputs_lm_inputs" not in d, desc

        d["inputs_lm_inputs"] = []
        all_texts = []

        for source_text, target_text in zip(d["source_text"], d["target_text"]):
            all_texts.append(f"{source_text}{tokenizer.sep_token}{target_text}")

        inputs = tokenizer.batch_encode_plus(all_texts, return_tensors=None, add_special_tokens=True, max_length=max_length_encoder,
                                             return_attention_mask=False, truncation=True, padding="longest")
        d["inputs_lm_inputs"] = inputs["input_ids"]
        inputs = torch.tensor(d["inputs_lm_inputs"])

        assert inputs.shape == (len(d["source_text"]), min(max_length_encoder, inputs.shape[1])), inputs.shape

        d["inputs_lm_inputs"] = inputs

if not multichannel:
    assert len(cnn_pooling) == 1, cnn_pooling

    cnn_pooling *= len(data_input_all_keys)

len_data = len(data_input_all_keys)

assert len(data_input_all_keys) == len_data, f"{data_input_all_keys} len is not {len_data}"
assert len(cnn_pooling) == len_data, f"{cnn_pooling} len is not {len_data}"

data_input_all_keys, cnn_pooling = \
    zip(*map(lambda s: s.split('|'), sorted([f"{d}|{c}" for d, c in zip(data_input_all_keys, cnn_pooling)]))) # We sort to get the same results when the order is different
data_input_all_keys = list(data_input_all_keys)
cnn_pooling = list(cnn_pooling)

assert len(data_input_all_keys) == len_data, f"{data_input_all_keys} len is not {len_data}"
assert len(cnn_pooling) == len_data, f"{cnn_pooling} len is not {len_data}"

print(f"CNN width and height: {cnn_width} {cnn_height}")
print(f"All channels (keys): {' '.join(data_input_all_keys)}")

assert len(set([k[7:] for k in train_data.keys() if k.startswith("inputs_")]).intersection(set(data_input_all_keys))) == len(data_input_all_keys), f"{[k for k in train_data.keys() if k.startswith('inputs_')]} vs keys"
assert len(set([k[7:] for k in dev_data.keys() if k.startswith("inputs_")]).intersection(set(data_input_all_keys))) == len(data_input_all_keys), f"{[k for k in dev_data.keys() if k.startswith('inputs_')]} vs keys"

if not skip_test:
    assert len(set([k[7:] for k in test_data.keys() if k.startswith("inputs_")]).intersection(set(data_input_all_keys))) == len(data_input_all_keys), f"{[k for k in test_data.keys() if k.startswith('inputs_')]} vs keys"

class MyDataset(Dataset):
    def __init__(self, data, all_keys, create_groups=False, return_group=False, add_text=False):
        self.create_groups = create_groups
        self.return_group = return_group
        self.data = {}
        self.uniq_groups = []
        self.all_keys = all_keys
        self.add_text = add_text

        if create_groups:
            self.groups = data["groups"]
        else:
            self.groups = list(range(len(data["labels"])))

            if "groups" in data:
                assert len(data["groups"]) == len(self.groups)

                _set_groups = set(data["groups"])

                if len(_set_groups) < len(self.groups):
                    print("warning: create_groups=False, but groups were provided, and there are groups with >1 element: {len(self.groups) - len(_set_groups)} groups with >1 element")

        for i in range(len(self.all_keys) - 1):
            k1 = self.all_keys[i]
            k2 = self.all_keys[i + 1]

            assert len(data[f"inputs_{k1}"]) == len(data[f"inputs_{k2}"])
            assert len(data[f"inputs_{k1}"]) == len(data["labels"])

        assert len(data["labels"]) == len(self.groups)

        for idx in range(len(self.groups)):
            group = self.groups[idx]

            if group not in self.data:
                self.uniq_groups.append(group)
                self.data[group] = {
                    'x': {k: [] for k in self.all_keys},
                    'y': [],
                }

                if self.add_text:
                    self.data[group]["source_text"] = []
                    self.data[group]["target_text"] = []

            for k in self.all_keys:
                self.data[group]['x'][k].append(data[f"inputs_{k}"][idx])

            self.data[group]['y'].append(data["labels"][idx])

            if self.add_text:
                self.data[group]["source_text"].append(data["source_text"][idx])
                self.data[group]["target_text"].append(data["target_text"][idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group = self.uniq_groups[idx]
        x = {k: [v.clone().detach().type(torch.float32) if k != "lm_inputs" else v.clone().detach() for v in l] for k, l in self.data[group]['x'].items()}
        y = [_y.clone().detach().type(torch.float32) for _y in self.data[group]['y']]

        if not self.create_groups:
            assert len(y) == 1

            for k in self.all_keys:
                assert len(x[k]) == 1

        result = {"x": x, "y": y}

        if self.return_group:
            result["group"] = group

        if self.add_text:
            result["source_text"] = self.data[group]["source_text"]
            result["target_text"] = self.data[group]["target_text"]

        return result

def wrapper_select_random_group_collate_fn(tokenizer=None, remove_padding=True, return_text=False):
    remove_padding = False if tokenizer is None else remove_padding
    padding_id = None if tokenizer is None else tokenizer.pad_token_id

    def collate_fn(batch, remove_padding=True, padding_id=None, return_text=False):
        data, target = [], []
        source_text, target_text = [], []

        for idx in range(len(batch)):
            x = batch[idx]["x"]
            y = batch[idx]["y"]
            _source_text = batch[idx]["source_text"] if return_text else None
            _target_text = batch[idx]["target_text"] if return_text else None

            assert len(y) > 0

            for _x in x.values():
                assert len(_x) == len(y)

            group_idx = np.random.randint(len(y))
            output_x = {k: v[group_idx] for k, v in x.items()}
            output_y = y[group_idx]

            data.append(output_x)
            target.append(output_y)

            if return_text:
                assert len(_source_text) == len(y)
                assert len(_target_text) == len(y)

                source_text.append(_source_text[group_idx])
                target_text.append(_target_text[group_idx])

        target = torch.stack(target, dim=0)
        data = {k: torch.stack([v[k] for v in data], dim=0) for k in data[0].keys()}

        if remove_padding and padding_id is not None and "lm_inputs" in data.keys():
            lm_len = data["lm_inputs"].shape[1]

            assert data["lm_inputs"].shape == (len(batch), lm_len), data["lm_inputs"].shape

            mask = ~(torch.all(data["lm_inputs"] == padding_id, dim=0))

            assert mask.shape == (lm_len,), mask.shape

            uc, uc_counts = torch.unique_consecutive(mask, return_counts=True)
            uc = uc.tolist()
            uc_counts = uc_counts.tolist()

            assert uc in ([True, False], [True]), mask

            data["lm_inputs"] = data["lm_inputs"][:, mask]

            assert data["lm_inputs"].shape == (len(batch), uc_counts[0]), data["lm_inputs"].shape

        if return_text:
            assert "source_text" not in data.keys()
            assert "target_text" not in data.keys()

            data["source_text"] = source_text
            data["target_text"] = target_text

        return data, target

    return lambda batch: collate_fn(batch, remove_padding=remove_padding, padding_id=padding_id, return_text=return_text)

class SimpleCNN(nn.Module):
    def __init__(self, c, w, h, num_classes, all_keys, pooling="max", only_conv=True, lang_model=None, disable_cnn=False, dropout_p=0.5):
        super(SimpleCNN, self).__init__()

        if disable_cnn:
            self.channels = 1
        else:
            self.channels = c

        self.only_conv = only_conv
        self.all_keys = list(all_keys)
        self.dimensions = (w, h)
        self.lang_model_hidden_size = 0
        self.disable_cnn = disable_cnn

        if lang_model is not None:
            assert "lm_inputs" in self.all_keys

            self.lang_model_hidden_size = lang_model.config.hidden_size

        if "lm_inputs" in self.all_keys:
            assert lang_model is not None

        if lang_model:
            self.all_keys.remove("lm_inputs")

        # First convolutional layer
        self.kernel_size = 3
        self.padding = 1
        self.layer_size = 32
        self.conv_layers = 2
        self.in_channels = [self.channels, *[self.layer_size * (2 ** i) for i in range(self.conv_layers - 1)]]
        self.out_channels = self.in_channels[1:] + [self.layer_size * (2 ** len(self.in_channels[1:]))]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.out_channels[i],
                                              kernel_size=self.kernel_size, stride=1, padding=self.padding,
                                              padding_mode="zeros") for i in range(self.conv_layers)])

        assert len(self.in_channels) == self.conv_layers
        assert len(self.out_channels) == self.conv_layers
        assert len(self.convs) == self.conv_layers

        # Second convolutional layer
        if pooling == "max":
            pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        elif pooling == "avg":
            pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            raise Exception(f"Unexpected pooling: {pooling}")

        self.pool = pool

        # Calculate the size of the feature map after the convolutional layers and pooling
        if self.disable_cnn:
            self._to_linear = 0
        else:
            self._to_linear = self.linear(torch.rand(1, self.channels, *self.dimensions)).numel()

        self._to_linear += self.lang_model_hidden_size

        # Fully connected layers
        self.hidden = 128
        self.fc1 = nn.Linear(self._to_linear, self.hidden)
        self.fc2 = nn.Linear(self.hidden, num_classes)

        self.dropout = nn.Dropout(p=dropout_p)

        self._initialize_weights()

        # Store lang model after weights initialization to avoid problems......
        self.lang_model = lang_model

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def linear(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))

        return x

    def forward(self, x):
        lm_input_ids = None

        if isinstance(x, dict):
            if self.lang_model:
                lm_input_ids = x["lm_inputs"]

            x = torch.cat([x[k] for k in self.all_keys], dim=1)

            assert x.shape[1:] == (self.channels, *self.dimensions), x.shape

        x = self.linear(x)

        if self.only_conv:
            return x

        bs = x.shape[0]
        x = x.view(bs, self._to_linear - self.lang_model_hidden_size)

        if self.lang_model:
            lm_attention_mask = utils.get_attention_mask(tokenizer, lm_input_ids)
            output = self.lang_model(input_ids=lm_input_ids, attention_mask=lm_attention_mask, output_hidden_states=True)
            last_hidden_state = output["hidden_states"][-1]
            classifier_token = last_hidden_state[:,0,:]

            assert classifier_token.shape == (bs, self.lang_model_hidden_size)

            x = torch.cat([x, classifier_token], dim=1)

        assert x.shape == (bs, self._to_linear)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, simple_cnns, all_keys, lang_model=None, disable_cnn=False, dropout_p=0.5):
        super(MultiChannelCNN, self).__init__()

        self.all_keys = list(all_keys)
        self.lang_model_hidden_size = 0
        self.disable_cnn = disable_cnn

        if lang_model is not None:
            assert "lm_inputs" in self.all_keys

            self.lang_model_hidden_size = lang_model.config.hidden_size

        if "lm_inputs" in self.all_keys:
            assert lang_model is not None

        if lang_model:
            self.all_keys.remove("lm_inputs")

        for k, simple_cnn in simple_cnns.items():
            assert k in self.all_keys
            assert isinstance(simple_cnn, SimpleCNN), type(simple_cnn)

        assert len(self.all_keys) == len(simple_cnns)

        if self.disable_cnn:
            self.simple_cnns = None
            self._to_linear = {k: 0 for k in simple_cnns.keys()}
        else:
            self.simple_cnns = nn.ModuleDict({k: v for k, v in simple_cnns.items()})
            self._to_linear = {k: simple_cnns[k]._to_linear for k in simple_cnns.keys()}

        self._to_linear_sum = sum([self._to_linear[k] for k in self._to_linear.keys()])
        self._to_linear_sum += self.lang_model_hidden_size

        # Fully connected layers
        self.hidden = 128
        self.fc1 = nn.Linear(self._to_linear_sum, self.hidden)
        self.fc2 = nn.Linear(self.hidden, num_classes)

        self.dropout = nn.Dropout(p=dropout_p)

        self._initialize_weights()

        # Store lang model after weights initialization to avoid problems......
        self.lang_model = lang_model

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, dict), type(x)

        lm_input_ids = x["lm_inputs"] if self.lang_model else None

        if not self.disable_cnn:
            x = [self.simple_cnns[k](x[k]) for k in self.all_keys]
            x = [_x.view(-1, self._to_linear[k]) for _x, k in zip(x, self.all_keys)]
            x = torch.cat(x, dim=1)
            bs = x.shape[0]

        if self.lang_model:
            lm_attention_mask = utils.get_attention_mask(tokenizer, lm_input_ids)

            assert lm_attention_mask.shape == lm_input_ids.shape

            output = self.lang_model(input_ids=lm_input_ids, attention_mask=lm_attention_mask, output_hidden_states=True)
            last_hidden_state = output["hidden_states"][-1]
            classifier_token = last_hidden_state[:,0,:]

            if not self.disable_cnn:
                assert classifier_token.shape == (bs, self.lang_model_hidden_size)

                x = torch.cat([x, classifier_token], dim=1)
            else:
                x = classifier_token
                bs = x.shape[0]

        assert x.shape == (bs, self._to_linear_sum)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def apply_inference(model, data, target=None, loss_function=None, threshold=0.5, loss_apply_sigmoid=False):
    model_outputs = model(data)
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

def eval(model, dataloader, all_keys, device, print_result=False, print_desc='-', threshold=0.5):
    training = model.training
    training_lm = False if not model.lang_model else model.lang_model.training

    model.eval()

    if model.lang_model:
        model.lang_model.eval()

    all_outputs = []
    all_labels = []
    print_idx = 0

    for data, target in dataloader:
        _data = {k: data[k].to(device) for k in all_keys}
        results = apply_inference(model, _data, target=None, loss_function=None, loss_apply_sigmoid=False)
        outputs_classification = results["outputs_classification_detach_list"]
        outputs = results["outputs"]
        outputs = torch.sigmoid(outputs).cpu().detach().tolist()
        labels = target.cpu()
        labels = torch.round(labels).type(torch.long)

        all_outputs.extend(outputs_classification)
        all_labels.extend(labels.tolist())

        if print_result:
            assert len(data["source_text"]) == len(outputs)
            assert len(data["target_text"]) == len(outputs)
            assert len(labels) == len(outputs)

            for source_text, target_text, output, label in zip(data["source_text"], data["target_text"], outputs, labels):
                output_classification = int(output >= threshold)

                assert output_classification in (0, 1), output_classification
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

        if not training_lm and model.lang_model:
            # Previous .train() might have enabled the language model training...
            model.lang_model.eval()

    if training_lm:
        model.lang_model.train()

    return results

# Load data
num_workers = 0
collate_fn = wrapper_select_random_group_collate_fn(tokenizer=tokenizer, remove_padding=True, return_text=model_inference)
train_dataset = MyDataset(train_data, data_input_all_keys, create_groups=True, return_group=True, add_text=model_inference)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_training, num_workers=num_workers, collate_fn=collate_fn)
dev_dataset = MyDataset(dev_data, data_input_all_keys, add_text=model_inference)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

# Model
num_classes = 1
epochs = 100
patience = 20

if multichannel:
    _data_input_all_keys = list(data_input_all_keys)
    _cnn_pooling = list(cnn_pooling)

    if lang_model:
        i = _data_input_all_keys.index("lm_inputs")

        del _cnn_pooling[i]
        del _data_input_all_keys[i]

    simple_cnns = {k: SimpleCNN(channels, cnn_width, cnn_height, num_classes, _data_input_all_keys, pooling=pooling, only_conv=True, disable_cnn=False, dropout_p=cnn_dropout) for k, pooling in zip(_data_input_all_keys, _cnn_pooling)}
    model = MultiChannelCNN(num_classes, simple_cnns, data_input_all_keys, lang_model=lang_model, disable_cnn=disable_cnn, dropout_p=cnn_dropout)
else:
    model = SimpleCNN(channels, cnn_width, cnn_height, num_classes, data_input_all_keys, pooling=cnn_pooling[0], only_conv=False, lang_model=lang_model, disable_cnn=disable_cnn, dropout_p=cnn_dropout)

if cnn_model_input:
    print(f"Loading CNN model (and optionally LM): {cnn_model_input}")

    assert not disable_cnn

    old_model_state_dict_keys = set(model.state_dict().keys())
    cnn_state_dict = torch.load(cnn_model_input, weights_only=True, map_location=device)

    if lang_model and model.state_dict()["fc1.weight"].shape[1] - cnn_state_dict["fc1.weight"].shape[1] == lang_model.config.hidden_size:
        # Our model has a longer feed-forward layer because of lang_model
        print("Fixing shape of layers...")

        fc1_weight = copy.deepcopy(cnn_state_dict["fc1.weight"])
        fc1_bias = copy.deepcopy(cnn_state_dict["fc1.bias"])

        assert model.hidden == fc1_weight.shape[0] == fc1_bias.shape[0]

        cnn_state_dict["fc1.weight"] = nn.Parameter(torch.rand(model.hidden, model.state_dict()["fc1.weight"].shape[1]), requires_grad=False)

        nn.init.xavier_normal_(cnn_state_dict["fc1.weight"])

        cnn_state_dict["fc1.bias"] = nn.Parameter(fc1_bias, requires_grad=True)
        cnn_state_dict["fc1.weight"][:,:fc1_weight.shape[1]] = fc1_weight.clone().detach()

        cnn_state_dict["fc1.weight"].requires_grad_(not model_inference)

    new_model_state_dict_keys = set(cnn_state_dict.keys())

    model_state_dict_keys_intersection = set.intersection(old_model_state_dict_keys, new_model_state_dict_keys)
    model_state_dict_keys_new_old_diff = set.difference(new_model_state_dict_keys, old_model_state_dict_keys)
    model_state_dict_keys_old_new_diff = set.difference(old_model_state_dict_keys, new_model_state_dict_keys)

    print(f"CNN model keys (old: {len(old_model_state_dict_keys)}; new: {len(new_model_state_dict_keys)}; intersection: {len(model_state_dict_keys_intersection)}): new - old: {model_state_dict_keys_new_old_diff}: old - new: {model_state_dict_keys_old_new_diff}")

    model.load_state_dict(cnn_state_dict, strict=False)

model = model.to(device)

if model_inference:
    model.eval()
else:
    model.train()

for p in model.parameters():
    p.requires_grad_(not model_inference)

if lang_model:
    if lm_frozen_params:
        lang_model.eval()
    else:
        lang_model.train()

    for p in lang_model.parameters():
        p.requires_grad_(not lm_frozen_params)

if not model_inference:
    loss_function = nn.BCELoss(reduction="none") # BCELoss vs BCEWithLogitsLoss: check https://github.com/pytorch/pytorch/issues/49844
    loss_apply_sigmoid = True # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    lm_model_parameters = list(filter(lambda p: p.requires_grad, lang_model.parameters())) if lang_model else []
    model_parameters_data = list(filter(lambda d: d[1].requires_grad, [(k, p) for k, p in model.named_parameters() if not k.startswith("lang_model.")]))
    model_parameters = [d[1] for d in model_parameters_data]
    model_parameters_names = [d[0] for d in model_parameters_data]
    optimizer_args = [{"params": model_parameters, "lr": learning_rate}]

    assert len(model_parameters_data) == len(model_parameters) == len(model_parameters_names)

    if lm_model_parameters:
        optimizer_args.append({"params": lm_model_parameters, "lr": lm_learning_rate})

    print(f"Parameters with requires_grad=True: {len(model_parameters)} (LM: {len(lm_model_parameters)})")
    #print(f"CNN parameters with requires_grad=True: {' '.join(model_parameters_names)}")

    optimizer = torch.optim.AdamW(optimizer_args, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    lr_scheduler_str = "linear"
    warmup_steps = 400

    print(f"Info: {epochs} epochs, {patience} patience, {lr_scheduler_str} LR scheduler ({warmup_steps} warmup, if applicable), {learning_rate} learning rate, {lm_learning_rate} LM learning rate")

    if lr_scheduler_str == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 400, len(train_dataloader) * epochs)
    elif lr_scheduler_str == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate * 1, steps_per_epoch=len(train_dataloader), epochs=epochs)
    else:
        raise Exception(f"Unexpected LR scheduler: {lr_scheduler_str}")

    early_stopping_best_result_dev = -np.inf # accuracy
    early_stopping_best_result_train = -np.inf # accuracy
    early_stopping_best_loss = np.inf
    current_patience = 0
    epoch_loss = []
    sum_epoch_loss = np.inf

if model_inference:
    print("Inference!")

    epochs = 0
else:
    print("Training!")

sys.stdout.flush()

for epoch in range(epochs):
    print(f"Epoch {epoch}")

    train_results = eval(model, train_dataloader, data_input_all_keys, device)
    dev_results = eval(model, dev_dataloader, data_input_all_keys, device)
    better_loss_result = False

    if len(epoch_loss) > 0 and sum_epoch_loss < early_stopping_best_loss:
        print(f"Better loss result: {early_stopping_best_loss} -> {sum_epoch_loss}")

        better_loss_result = True
        early_stopping_best_loss = sum_epoch_loss

    print(f"Train eval: {train_results}")
    print(f"Dev eval: {dev_results}")

    epoch_loss = []
    early_stopping_metric_train = train_results["acc"]
    early_stopping_metric_dev = dev_results["acc"]
    better_train_result = False
    patience_dev_equal = np.isclose(early_stopping_metric_dev, early_stopping_best_result_dev)
    patience_train_equal = np.isclose(early_stopping_metric_train, early_stopping_best_result_train)

    if early_stopping_metric_train > early_stopping_best_result_train:
        print(f"Better train result: {early_stopping_best_result_train} -> {early_stopping_metric_train}")

        better_train_result = True
        early_stopping_best_result_train = early_stopping_metric_train

    if early_stopping_metric_dev > early_stopping_best_result_dev or ((patience_dev_equal and better_train_result) or (patience_dev_equal and patience_train_equal and better_loss_result)):
        print(f"Patience better dev result: {early_stopping_best_result_dev} -> {early_stopping_metric_dev}")

        current_patience = 0
        early_stopping_best_result_dev = early_stopping_metric_dev

        if save_model_path:
            print(f"Saving best model: {save_model_path}")

            torch.save(model.state_dict(), save_model_path)
    else:
        current_patience += 1

        print(f"Exhausting patience... {current_patience} / {patience}")

    if current_patience >= patience:
        print("Patience is over ...")

        break

    model.zero_grad()
    final_loss = None
    loss_elements = 0

    for batch_idx, (data, target) in enumerate(train_dataloader, 1):
        data = {k: data[k].to(device) for k in data_input_all_keys}
        target = target.to(device)

        result = apply_inference(model, data, target=target, loss_function=loss_function, loss_apply_sigmoid=loss_apply_sigmoid)
        _loss = result["loss"]

        assert len(_loss.shape) == 1, _loss.shape

        loss_elements += _loss.numel()

        if final_loss is None:
            final_loss = torch.sum(_loss)
        else:
            final_loss += torch.sum(_loss)

        if batch_idx % gradient_accumulation == 0 or batch_idx == len(train_dataloader):
            assert final_loss is not None

            loss = final_loss / loss_elements
            final_loss = None
            loss_elements = 0

            epoch_loss.append(loss.cpu().detach().item())

            loss.backward()

            optimizer.step()
            scheduler.step()

            model.zero_grad()

        if batch_idx % (100 * gradient_accumulation) == 0:
            sum_partial_loss = sum(epoch_loss[-100:])

            print(f"Epoch loss (sum last 100 steps): step {batch_idx}: {sum_partial_loss}")

            sys.stdout.flush()

    sum_epoch_loss = sum(epoch_loss)

    print(f"Epoch loss: {sum_epoch_loss}")

    assert str(sum_epoch_loss) != "nan", "Some values in the input data are NaN"

    sys.stdout.flush()

if save_model_path and not model_inference:
    print(f"Loading best model: {save_model_path}")

    model_state_dict = torch.load(save_model_path, weights_only=True, map_location=device)

    assert model.state_dict().keys() == model_state_dict.keys()

    model.load_state_dict(model_state_dict)

    model = model.to(device)

if not model_inference_skip_train:
    train_results = eval(model, train_dataloader, data_input_all_keys, device, print_result=model_inference, print_desc="train")

    print(f"Final train eval: {train_results}")
else:
    print("Final train eval: skip (inference)")

del train_dataset
del train_dataloader

torch.cuda.empty_cache()

dev_results = eval(model, dev_dataloader, data_input_all_keys, device, print_result=model_inference, print_desc="dev")

print(f"Final dev eval: {dev_results}")

if not skip_test:
    del dev_dataset
    del dev_dataloader

    torch.cuda.empty_cache()

    test_dataset = MyDataset(test_data, data_input_all_keys, add_text=model_inference)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    test_results = eval(model, test_dataloader, data_input_all_keys, device, print_result=model_inference, print_desc="test")

    print(f"Final test eval: {test_results}")
