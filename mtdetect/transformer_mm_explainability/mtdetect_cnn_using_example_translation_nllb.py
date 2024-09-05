
import os
import sys
import json
import pickle

import mtdetect.transformer_mm_explainability.example_translation_nllb as example_translation_nllb
import mtdetect.inference as inference

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print(f"Provided args: {sys.argv}")

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
teacher_forcing = default_sys_argv(18, [False], f=lambda q: [False] if q == '' else [bool(int(_q)) for _q in q.split('+')])
ignore_attention = default_sys_argv(19, [False], f=lambda q: [False] if q == '' else [bool(int(_q)) for _q in q.split('+')])
force_pickle_file = True # Change manually
skip_test = False # Change manually

if skip_test:
    print(f"warning: test set evaluation is disabled")

for _attention_matrix in attention_matrix:
    assert _attention_matrix in ("encoder", "decoder", "cross"), attention_matrix

for _cnn_pooling in cnn_pooling:
    assert _cnn_pooling in ("max", "avg"), cnn_pooling

assert len(cnn_pooling) <= len(attention_matrix)

if multichannel and len(cnn_pooling) < len(attention_matrix):
    assert len(cnn_pooling) == 1, cnn_pooling

    cnn_pooling = [cnn_pooling[0]] * len(attention_matrix)

if not multichannel:
    assert len(cnn_pooling) == 1, cnn_pooling

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
patience = 100

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
            explainability_ee = pickle_data["explainability_encoder"]
            explainability_dd = pickle_data["explainability_decoder"]
            explainability_de = pickle_data["explainability_cross"]

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

    fd.close()

    if load_explainability_arrays and fn_pickle_array_exists:
        # Explainability arrays were loaded

        assert len(source_text) == len(explainability_ee), f"{len(source_text)} != {len(explainability_ee)}"
        assert len(source_text) == len(explainability_dd), f"{len(source_text)} != {len(explainability_dd)}"
        assert len(source_text) == len(explainability_de), f"{len(source_text)} != {len(explainability_de)}"

    if store_explainability_arrays and not fn_pickle_array_exists:
        print(f"Storing explainability arrays: {fn_pickle_array}")

        with open(fn_pickle_array, "wb") as pickle_fd:
            pickle_data = {
                "explainability_encoder": explainability_ee,
                "explainability_decoder": explainability_dd,
                "explainability_cross": explainability_de,
            }

            pickle.dump(pickle_data, pickle_fd)

    print(f"Samples: {len(groups)}; Groups: {len(uniq_groups)}")

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

        _input = extend_tensor_with_zeros_and_truncate(_input, cnn_width, cnn_height, device)
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

for d in direction:
    assert d in ("src2trg", "trg2src"), d

channels_factor_len_set = set([len(direction), len(teacher_forcing), len(ignore_attention)])

assert len(channels_factor_len_set) in (1, 2), channels_factor_len_set

if len(channels_factor_len_set) == 2:
    assert 1 in channels_factor_len_set, channels_factor_len_set

if multichannel:
    channels = 1
    channels_factor = 1
else:
    # Expected: for each value provided to direction, teacher_forcing, and ignore_attention, we will have an extra set of len(attention_matrix) channels
    # Example: {direction: src2trg+trg2src, teacher_forcing: True+False, ignore_attention: False} -> [(src2trg, True, False), (trg2src, False, False)] # ignore_attention is expanded
    # Example: {direction: src2trg+src2trg+trg2src+trg2src, teacher_forcing: True+False+True+False, ignore_attention: False+True+False+True} -> [(src2trg, True, False), (src2trg, False, True), (trg2src, True, False), (trg2src, False, True)]
    channels = len(attention_matrix)

    channels_factor = max(channels_factor_len_set)
    direction *= 1 if len(direction) else channels_factor
    teacher_forcing *= 1 if len(teacher_forcing) else channels_factor
    ignore_attention *= 1 if len(ignore_attention) else channels_factor

channels *= channels_factor

print(f"Total channels: {channels} (factor: {channels_factor})")

if channels_factor > 1 and not force_pickle_file:
    print(f"warning: channels_factor={channels_factor} > 1, and force_pickle_file=False: it may be very slow to create all the pickle files if they do not exist (they may exist)")

cnn_width = -np.inf
cnn_height = -np.inf
data_input_all_keys = []
first_time = True

for _direction, _teacher_forcing, _ignore_attention in zip(direction, teacher_forcing, ignore_attention):
    # TODO we are reading the files several times...

    train_data = read(train_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                      focus=attention_matrix, device=device, pretrained_model=pretrained_model, pickle_template="train",
                      teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention, force_pickle_file=force_pickle_file)
    dev_data = read(dev_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                    focus=attention_matrix, device=device, pretrained_model=pretrained_model, pickle_template="dev",
                    teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention, force_pickle_file=force_pickle_file)

    if skip_test:
        test_data = {"cnn_width": -np.inf, "cnn_height": -np.inf}
    else:
        test_data = read(test_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                         focus=attention_matrix, device=device, pretrained_model=pretrained_model, pickle_template="test",
                         teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention, force_pickle_file=force_pickle_file)

    cnn_width = max(train_data["cnn_width"], dev_data["cnn_width"], test_data["cnn_width"], cnn_width)
    cnn_height = max(train_data["cnn_height"], dev_data["cnn_height"], test_data["cnn_height"], cnn_height)

    for _attention_matrix in attention_matrix:
        inputs = f"{_attention_matrix}_{_direction}_{_teacher_forcing}_{_ignore_attention}"
        train_data[f"inputs_{inputs}"], train_data["labels"] = get_data(train_data[_attention_matrix], train_data["labels"], train_data["loaded_samples"], cnn_width, cnn_height, device, convert_labels_to_tensor=first_time)
        dev_data[f"inputs_{inputs}"], dev_data["labels"] = get_data(dev_data[_attention_matrix], dev_data["labels"], dev_data["loaded_samples"], cnn_width, cnn_height, device, convert_labels_to_tensor=first_time)

        if not skip_test:
            test_data[f"inputs_{inputs}"], test_data["labels"] = get_data(test_data[_attention_matrix], test_data["labels"], test_data["loaded_samples"], cnn_width, cnn_height, device, convert_labels_to_tensor=first_time)

        first_time = False

        data_input_all_keys.append(inputs)

data_input_all_keys = sorted(data_input_all_keys)

print(f"CNN width and height: {cnn_width} {cnn_height}")

class MyDataset(Dataset):
    def __init__(self, data, all_keys, create_groups=False, return_group=False):
        self.create_groups = create_groups
        self.return_group = return_group
        self.data = {}
        self.uniq_groups = []
        self.all_keys = all_keys

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

            for k in self.all_keys:
                self.data[group]['x'][k].append(data[f"inputs_{k}"][idx])

            self.data[group]['y'].append(data["labels"][idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group = self.uniq_groups[idx]
        x = {k: [v.clone().detach().type(torch.float32) for v in l] for k, l in self.data[group]['x'].items()}
        y = [_y.clone().detach().type(torch.float32) for _y in self.data[group]['y']]

        if not self.create_groups:
            assert len(y) == 1

            y = y[0]

            for k in self.all_keys:
                assert len(x[k]) == 1

                x[k] = x[k][0]

        if self.return_group:
            return x, y, group

        return x, y

def select_random_group_collate_fn(batch):
    data, target = [], []

    for idx in range(len(batch)):
        x, y, group = batch[idx]

        assert len(y) > 0

        for _x in x.values():
            assert len(_x) == len(y)

        group_idx = np.random.randint(len(y))
        output_x = {k: v[group_idx] for k, v in x.items()}
        output_y = y[group_idx]

        data.append(output_x)
        target.append(output_y)

    target = torch.stack(target, dim=0)
    data = {k: torch.stack([v[k] for v in data], dim=0) for k in data[0].keys()}

    return data, target

class SimpleCNN(nn.Module):
    def __init__(self, c, w, h, num_classes, all_keys, pooling="max", only_conv=True):
        super(SimpleCNN, self).__init__()

        self.only_conv = only_conv
        self.all_keys = all_keys
        self.channels = c
        self.dimensions = (w, h)

        # First convolutional layer
        self.kernel_size = 3
        self.padding = 1
        self.layer_size = 32
        self.conv_layers = 2
        self.in_channels = [c, *[self.layer_size * (2 ** i) for i in range(self.conv_layers - 1)]]
        self.out_channels = self.in_channels[1:] + [self.layer_size * (2 ** len(self.in_channels[1:]))]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=self.kernel_size, stride=1, padding=self.padding, padding_mode="zeros") for i in range(self.conv_layers)])

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
        self._to_linear = self.linear(torch.rand(1, self.channels, *self.dimensions)).numel()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self._initialize_weights()

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
        if isinstance(x, dict):
            x = torch.cat([x[k] for k in self.all_keys], dim=1)

            assert x.shape[1:] == (self.channels, *self.dimensions), x.shape

        x = self.linear(x)

        if self.only_conv:
            return x

        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes, simple_cnns, all_keys):
        super(MultiChannelCNN, self).__init__()

        self.all_keys = all_keys

        for k, simple_cnn in simple_cnns.items():
            assert k in self.all_keys
            assert isinstance(simple_cnn, SimpleCNN), type(simple_cnn)

        assert len(self.all_keys) == len(simple_cnns)

        self.simple_cnns = nn.ModuleDict({k: v for k, v in simple_cnns.items()})
        self._to_linear = {k: simple_cnns[k]._to_linear for k in simple_cnns.keys()}
        self._to_linear_sum = sum([self._to_linear[k] for k in self._to_linear.keys()])

        # Fully connected layers
        self.hidden = 128
        self.fc1 = nn.Linear(self._to_linear_sum, self.hidden)
        self.fc2 = nn.Linear(self.hidden, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self._initialize_weights()

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
        x = [self.simple_cnns[k](x[k]) for k in self.all_keys]
        x = [_x.view(-1, self._to_linear[k]) for _x, k in zip(x, self.all_keys)]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def apply_inference(model, data, target=None, loss_function=None, threshold=0.5):
    model_outputs = model(data)
    outputs = model_outputs
    outputs = outputs.squeeze(1)
    loss = None

    if loss_function is not None and target is not None:
        loss = loss_function(outputs, target)

    outputs_classification = torch.sigmoid(outputs).cpu().detach().tolist()
    outputs_classification = list(map(lambda n: int(n >= threshold), outputs_classification))

    results = {
        "outputs": outputs,
        "outputs_classification_detach_list": outputs_classification,
        "loss": loss,
    }

    return results

def eval(model, dataloader, all_keys, device):
    training = model.training

    model.eval()

    all_outputs = []
    all_labels = []

    for data, target in dataloader:
        data = {k: data[k].to(device) for k in all_keys}
        results = apply_inference(model, data, target=None, loss_function=None)
        outputs_classification = results["outputs_classification_detach_list"]
        labels = target.cpu()
        labels = torch.round(labels).type(torch.long)

        all_outputs.extend(outputs_classification)
        all_labels.extend(labels.tolist())

    all_outputs = torch.as_tensor(all_outputs)
    all_labels = torch.as_tensor(all_labels)
    results = inference.get_metrics(all_outputs, all_labels)

    if training:
        model.train()

    return results

# Load data
num_workers = 0
train_dataset = MyDataset(train_data, data_input_all_keys, create_groups=True, return_group=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=select_random_group_collate_fn)
dev_dataset = MyDataset(dev_data, data_input_all_keys)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model
num_classes = 1
epochs = 500

if multichannel:
    simple_cnns = {k: SimpleCNN(channels, cnn_width, cnn_height, num_classes, data_input_all_keys, pooling=pooling, only_conv=True) for k, pooling in zip(data_input_all_keys, cnn_pooling)}
    model = MultiChannelCNN(num_classes, simple_cnns, data_input_all_keys)
else:
    model = SimpleCNN(channels, cnn_width, cnn_height, num_classes, data_input_all_keys, pooling=cnn_pooling[0], only_conv=False)

model = model.to(device)

model.train()

loss_function = nn.BCEWithLogitsLoss(reduction="mean")
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate * 1, steps_per_epoch=len(train_dataloader), epochs=epochs)
early_stopping_best_result_dev = -np.inf # accuracy
early_stopping_best_result_train = -np.inf # accuracy
early_stopping_best_loss = np.inf
current_patience = 0
epoch_loss = None

print("Training...")

sys.stdout.flush()

for epoch in range(epochs):
    print(f"Epoch {epoch}")

    train_results = eval(model, train_dataloader, data_input_all_keys, device)
    dev_results = eval(model, dev_dataloader, data_input_all_keys, device)
    better_loss_result = False

    if epoch_loss is not None and epoch_loss < early_stopping_best_loss:
        print(f"Better loss result: {early_stopping_best_loss} -> {epoch_loss}")

        better_loss_result = True
        early_stopping_best_loss = epoch_loss

    print(f"Train eval: {train_results}")
    print(f"Dev eval: {dev_results}")

    epoch_loss = 0.0
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

            torch.save(model, save_model_path)
    else:
        current_patience += 1

        print(f"Exhausting patience... {current_patience} / {patience}")

    if current_patience >= patience:
        print("Patience is over ...")

        break

    for batch_idx, (data, target) in enumerate(train_dataloader, 1):
        data = {k: data[k].to(device) for k in data_input_all_keys}
        target = target.to(device)

        model.zero_grad()

        result = apply_inference(model, data, target=target, loss_function=loss_function)
        loss = result["loss"]

        epoch_loss += loss

        loss.backward()

        optimizer.step()
        scheduler.step()

    print(f"Loss: {epoch_loss}")

    assert str(epoch_loss) != "nan", "Some values in the input data are NaN"

    sys.stdout.flush()

if save_model_path:
    print(f"Loading best model: {save_model_path}")

    model = torch.load(save_model_path, weights_only=False, map_location=device)

train_results = eval(model, train_dataloader, data_input_all_keys, device)

print(f"Final train eval: {train_results}")

del train_dataset
del train_dataloader

torch.cuda.empty_cache()

dev_results = eval(model, dev_dataloader, data_input_all_keys, device)

print(f"Final dev eval: {dev_results}")

if not skip_test:
    del dev_dataset
    del dev_dataloader

    torch.cuda.empty_cache()

    test_dataset = MyDataset(test_data, data_input_all_keys)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_results = eval(model, test_dataloader, data_input_all_keys, device)

    print(f"Final test eval: {test_results}")
