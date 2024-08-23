
import sys

import mtdetect.transformer_mm_explainability.example_translation_nllb as example_translation_nllb
import mtdetect.inference as inference

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_inverse_sqrt_schedule

train_filename = sys.argv[1]
dev_filename = sys.argv[2]
test_filename = sys.argv[3]
batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 8
batch_size = batch_size if batch_size > 0 else 8
cnn_max_width = int(sys.argv[5]) if len(sys.argv) > 5 else np.inf
cnn_max_width = cnn_max_width if cnn_max_width > 0 else np.inf
cnn_max_height = int(sys.argv[6]) if len(sys.argv) > 6 else np.inf
cnn_max_height = cnn_max_height if cnn_max_height > 0 else np.inf
source_lang = sys.argv[7] if len(sys.argv) > 7 else "eng_Latn" # e.g., eng_Latn
target_lang = sys.argv[8] if len(sys.argv) > 8 else "spa_Latn" # e.g., spa_Latn

print(f"NLLB: from {source_lang} to {target_lang}")

self_attention_remove_diagonal = True
explainability_normalization = "relative" # allowed: none, absolute, relative
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

def extend_tensor_with_zeros(t, max_width, max_height, device):
    assert len(t.shape) == 2

    result = torch.zeros((max_width, max_height)).to(device)
    result[:t.shape[0], :t.shape[1]] = t[:max_width, :max_height]

    return result

def read(filename):
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

    for l in fd:
        s, t, l = l.rstrip("\r\n").split('\t')

        source_text.append(s)
        target_text.append(t)
        labels.append(float(l))

        input_tokens, output_tokens, r_ee, r_dd, r_de = \
            example_translation_nllb.explainability(s, target_text=t, source_lang=source_lang, target_lang=target_lang, debug=False,
                                                    apply_normalization=True, self_attention_remove_diagonal=self_attention_remove_diagonal,
                                                    explainability_normalization=explainability_normalization)

        explainability_ee.append(r_ee)
        explainability_dd.append(r_dd)
        explainability_de.append(r_de)

        #print(f"{loaded_samples + 1} pairs loaded! {r_de.shape}")

        width, height = r_de.shape
        cnn_width = min(max(cnn_width, width), cnn_max_width)
        cnn_height = min(max(cnn_height, height), cnn_max_height)
        loaded_samples += 1

        if loaded_samples % 100 == 0:
            print(f"{loaded_samples} samples loaded: {filename}")

            sys.stdout.flush()

    fd.close()

    return {
        "cnn_width": cnn_width,
        "cnn_height": cnn_height,
        "loaded_samples": loaded_samples,
        "source_text": source_text,
        "target_text": target_text,
        "labels": labels,
        "explainability_ee": explainability_ee,
        "explainability_dd": explainability_dd,
        "explainability_de": explainability_de,
    }

def get_data(explainability_de, labels, loaded_samples, cnn_width, cnn_height, device):
    inputs = []

    for _input in explainability_de:
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

train_data = read(train_filename)
dev_data = read(dev_filename)
test_data = read(test_filename)
cnn_width = max(train_data["cnn_width"], dev_data["cnn_width"], test_data["cnn_width"])
cnn_height = max(train_data["cnn_height"], dev_data["cnn_height"], test_data["cnn_height"])
train_data_data = get_data(train_data["explainability_de"], train_data["labels"], train_data["loaded_samples"], cnn_width, cnn_height, device)
dev_data_data = get_data(dev_data["explainability_de"], dev_data["labels"], dev_data["loaded_samples"], cnn_width, cnn_height, device)
test_data_data = get_data(test_data["explainability_de"], test_data["labels"], test_data["loaded_samples"], cnn_width, cnn_height, device)
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
        #x = torch.tensor(self.x[idx], dtype=torch.float32)
        #y = torch.tensor(self.y[idx], dtype=torch.float32)
        x = self.x[idx].clone().detach().type(torch.float32)
        y = self.y[idx].clone().detach().type(torch.float32)

        return x, y

class SimpleCNN(nn.Module):
    def __init__(self, c, w, h, num_classes):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

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
        x = x.view(-1, self._to_linear)  # Flatten the tensor
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
epochs = 100
model = SimpleCNN(channels, cnn_width, cnn_height, num_classes).to(device)

model.train()

loss_function = nn.BCEWithLogitsLoss(reduction="mean")
#learning_rate = 1e-5
learning_rate = 1e-3
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(model_parameters, lr=learning_rate)
linear_steps = 0
scheduler = get_inverse_sqrt_schedule(optimizer, linear_steps)

for epoch in range(epochs):
    print(f"Epoch {epoch}")

    train_results = eval(model, train_dataloader, device)
    dev_results = eval(model, dev_dataloader, device)
    epoch_loss = 0.0

    print(f"Train eval: {train_results}")
    print(f"Dev eval: {dev_results}")

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
