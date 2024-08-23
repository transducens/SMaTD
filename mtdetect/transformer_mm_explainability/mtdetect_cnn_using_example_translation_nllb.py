
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
from torch.optim import AdamW
from transformers import get_inverse_sqrt_schedule, get_linear_schedule_with_warmup

print(f"Provided args: {sys.argv}")

def default_sys_argv(n, default, f=str):
    return f(sys.argv[n]) if len(sys.argv) > n else default

train_filename = sys.argv[1]
dev_filename = sys.argv[2]
test_filename = sys.argv[3]
batch_size = default_sys_argv(4, 16, f=int)
batch_size = batch_size if batch_size > 0 else 16
cnn_max_width = default_sys_argv(5, -np.inf, f=int)
cnn_max_width = cnn_max_width if cnn_max_width > 0 else -np.inf
cnn_max_height = default_sys_argv(6, -np.inf, f=int)
cnn_max_height = cnn_max_height if cnn_max_height > 0 else -np.inf
source_lang = default_sys_argv(7, "eng_Latn")
source_lang = source_lang if source_lang != '' else "eng_Latn"
target_lang = default_sys_argv(8, "spa_Latn")
target_lang = target_lang if target_lang != '' else "spa_Latn"
direction = default_sys_argv(9, "src2trg")
attention_matrix = default_sys_argv(10, "cross")
explainability_normalization = default_sys_argv(11, "relative")
self_attention_remove_diagonal = default_sys_argv(12, True, f=lambda q: bool(int(q)))
cnn_pooling = default_sys_argv(13, "max")

assert direction in ("src2trg", "trg2src", "only_src", "only_trg"), direction
assert attention_matrix in ("encoder", "decoder", "cross"), attention_matrix
assert explainability_normalization in ("none", "absolute", "relative"), explainability_normalization
assert cnn_pooling in ("max", "avg"), cnn_pooling

attention_matrix = f"explainability_{attention_matrix}"
translation_model_conf = {
    "source_lang": source_lang,
    "target_lang": target_lang,
    "direction": direction,
    "attention_matrix": attention_matrix,
    "explainability_normalization": explainability_normalization,
    "self_attention_remove_diagonal": self_attention_remove_diagonal,
}
translation_model_conf = json.dumps(translation_model_conf, indent=4)

print(f"NLLB conf:\n{translation_model_conf}")

channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
patience = 3

def extend_tensor_with_zeros(t, max_width, max_height, device):
    assert len(t.shape) == 2

    result = torch.zeros((max_width, max_height)).to(device)
    result[:t.shape[0], :t.shape[1]] = t[:max_width, :max_height]

    return result

def read(filename, direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
         focus="cross", store_explainability_arrays=True, load_explainability_arrays=True):
    cnn_width = -np.inf
    cnn_height = -np.inf
    loaded_samples = 0
    source_text = []
    target_text = []
    labels = []
    explainability_ee = []
    explainability_dd = []
    explainability_de = []
    fd = open(filename)
    first_msg = False
    fn_pickle_array = f"{filename}.{direction}.{source_lang}.{target_lang}.pickle"
    fn_pickle_array_exists = os.path.isfile(fn_pickle_array)

    if load_explainability_arrays and fn_pickle_array_exists:
        print(f"Loading explainability arrays: {fn_pickle_array}")

        with open(fn_pickle_array, "rb") as pickle_fd:
            pickle_data = pickle.load(pickle_fd)
            explainability_ee = pickle_data["explainability_encoder"]
            explainability_dd = pickle_data["explainability_decoder"]
            explainability_de = pickle_data["explainability_cross"]

    for idx, l in enumerate(fd):
        s, t, l = l.rstrip("\r\n").split('\t')

        if direction == "src2trg":
            pass
        elif direction == "trg2src":
            s, t = t, s # swap
        elif direction == "only_src": # disable teacher forcing
            t = ''
        elif direction == "only_trg": # disable teacher forcing
            s, t = t, ''
        else:
            raise Exception(f"Unexpected direction: {direction}")

        if not first_msg:
            print(f"The next sentence (source) is expected to be {source_lang}: {s}")

            if t != '':
                print(f"The next sentence (target) is expected to be {target_lang}: {t}")

            first_msg = True

        source_text.append(s)
        target_text.append(t)
        labels.append(float(l))

        if not fn_pickle_array_exists:
            input_tokens, output_tokens, r_ee, r_dd, r_de = \
                example_translation_nllb.explainability(s, target_text=t, source_lang=source_lang, target_lang=target_lang, debug=False,
                                                        apply_normalization=True, self_attention_remove_diagonal=False,
                                                        explainability_normalization="none")

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

        if focus == "explainability_encoder":
            width, height = r_ee.shape
        elif focus == "explainability_decoder":
            width, height = r_dd.shape
        elif focus == "explainability_cross":
            width, height = r_de.shape
        else:
            raise Exception(f"Unexpected focus: {focus}")

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

    return {
        "cnn_width": cnn_width,
        "cnn_height": cnn_height,
        "loaded_samples": loaded_samples,
        "source_text": source_text,
        "target_text": target_text,
        "labels": labels,
        "explainability_encoder": explainability_ee,
        "explainability_decoder": explainability_dd,
        "explainability_cross": explainability_de,
    }

def get_data(explainability_matrix, labels, loaded_samples, cnn_width, cnn_height, device):
    inputs = []

    for _input in explainability_matrix:
        _input = torch.from_numpy(_input)

        assert len(_input.shape) == 2

        _input = extend_tensor_with_zeros(_input, cnn_width, cnn_height, device)
        _input = _input.tolist()

        inputs.append(_input)

    inputs = torch.tensor(inputs)
    inputs = inputs.unsqueeze(1).to(device) # channel dim
    labels = torch.tensor(labels).to(device)

    inputs_expected_shape = (loaded_samples, channels, cnn_width, cnn_height)
    labels_expected_shape = (loaded_samples,)

    assert inputs.shape == inputs_expected_shape, inputs.shape
    assert labels.shape == labels_expected_shape, labels.shape

    return inputs, labels

train_data = read(train_filename, direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization, focus=attention_matrix)
dev_data = read(dev_filename, direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization, focus=attention_matrix)
test_data = read(test_filename, direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization, focus=attention_matrix)
cnn_width = max(train_data["cnn_width"], dev_data["cnn_width"], test_data["cnn_width"])
cnn_height = max(train_data["cnn_height"], dev_data["cnn_height"], test_data["cnn_height"])
train_data_data = get_data(train_data[attention_matrix], train_data["labels"], train_data["loaded_samples"], cnn_width, cnn_height, device)
dev_data_data = get_data(dev_data[attention_matrix], dev_data["labels"], dev_data["loaded_samples"], cnn_width, cnn_height, device)
test_data_data = get_data(test_data[attention_matrix], test_data["labels"], test_data["loaded_samples"], cnn_width, cnn_height, device)
train_data["inputs"], train_data["labels"] = train_data_data
dev_data["inputs"], dev_data["labels"] = dev_data_data
test_data["inputs"], test_data["labels"] = test_data_data

print(f"CNN width and height: {cnn_width} {cnn_height}")

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].clone().detach().type(torch.float32)
        y = self.y[idx].clone().detach().type(torch.float32)

        return x, y

class SimpleCNN(nn.Module):
    def __init__(self, c, w, h, num_classes, pooling="max"):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)

        # Second convolutional layer
        if pooling == "max":
            pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        elif pooling == "avg":
            pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            raise Exception(f"Unexpected pooling: {pooling}")

        self.pool = pool

        # Calculate the size of the feature map after the convolutional layers and pooling
        self._to_linear = self._calculate_linear_input(c, w, h)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_linear_input(self, c, w, h):
        x = torch.rand(1, c, w, h)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def apply_inference(model, data, target=None, loss_function=None, threshold=0.5):
    model_outputs = model(data)
    outputs = model_outputs
    outputs = outputs.squeeze(1)
    loss = None

    if loss_function is not None:
        loss = loss_function(outputs, target)

    outputs_classification = torch.sigmoid(outputs).cpu().detach().tolist()
    outputs_classification = list(map(lambda n: int(n >= threshold), outputs_classification))

    results = {
        "outputs": outputs,
        "outputs_classification_detach_list": outputs_classification,
        "loss": loss,
    }

    return results

def eval(model, dataloader, device):
    training = model.training

    model.eval()

    all_outputs = []
    all_labels = []

    for data, target in dataloader:
        data = data.to(device)
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
train_dataset = MyDataset(train_data["inputs"], train_data["labels"])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataset = MyDataset(dev_data["inputs"], dev_data["labels"])
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# Model
num_classes = 1
epochs = 10
model = SimpleCNN(channels, cnn_width, cnn_height, num_classes, pooling=cnn_pooling).to(device)

model.train()

loss_function = nn.BCEWithLogitsLoss(reduction="mean")
learning_rate = 1e-5
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(model_parameters, lr=learning_rate, weight_decay=0.0)
warmup_steps = 400
#scheduler = get_inverse_sqrt_schedule(optimizer, warmup_steps)
#scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0) # Disable LR scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, len(train_dataloader) * epochs)
early_stopping_best_result = -np.inf # accuracy
current_patience = 0

for epoch in range(epochs):
    print(f"Epoch {epoch}")

    train_results = eval(model, train_dataloader, device)
    dev_results = eval(model, dev_dataloader, device)
    epoch_loss = 0.0

    print(f"Train eval: {train_results}")
    print(f"Dev eval: {dev_results}")

    early_stopping_metric = dev_results["acc"]

    if early_stopping_metric > early_stopping_best_result:
        print(f"Patience better result: {early_stopping_best_result} -> {early_stopping_metric}")

        current_patience = 0
        early_stopping_best_result = early_stopping_metric
    else:
        current_patience += 1

        print(f"Exhausting patience... {current_patience} / {patience}")

    if current_patience >= patience:
        print("Patience is over ...")

        break

    for batch_idx, (data, target) in enumerate(train_dataloader, 1):
        data = data.to(device)

        model.zero_grad()

        result = apply_inference(model, data, target=target, loss_function=loss_function)
        loss = result["loss"]

        epoch_loss += loss

        loss.backward()

        optimizer.step()
        scheduler.step()

    print(f"Loss: {epoch_loss}")

train_results = eval(model, train_dataloader, device)

print(f"Final train eval: {train_results}")

del train_dataset
del train_dataloader

torch.cuda.empty_cache()

dev_results = eval(model, dev_dataloader, device)

print(f"Final dev eval: {dev_results}")

del dev_dataset
del dev_dataloader

torch.cuda.empty_cache()

test_dataset = MyDataset(test_data["inputs"], test_data["labels"])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_results = eval(model, test_dataloader, device)

print(f"Final test eval: {test_results}")
