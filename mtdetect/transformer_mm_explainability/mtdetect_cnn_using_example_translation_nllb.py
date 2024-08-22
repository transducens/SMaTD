
import sys

import mtdetect.transformer_mm_explainability.example_translation_nllb as example_translation_nllb

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
source_lang = sys.argv[4] if len(sys.argv) > 4 else "eng_Latn" # e.g., eng_Latn
target_lang = sys.argv[5] if len(sys.argv) > 5 else "spa_Latn" # e.g., spa_Latn

print(f"NLLB: from {source_lang} to {target_lang}")

self_attention_remove_diagonal = True
explainability_normalization = "relative" # allowed: none, absolute, relative
cnn_max_width = np.inf # TODO user value
cnn_max_height = np.inf # TODO user value
batch_size = 3 # TODO user value
channels = 1

def extend_tensor_with_zeros(t, max_width, max_height):
    assert len(t.shape) == 2
    assert t.shape[0] <= max_width
    assert t.shape[1] <= max_height

    result = torch.zeros((max_width, max_height))
    result[:t.shape[0], :t.shape[1]] = t

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

def get_data(explainability_de, labels, loaded_samples, cnn_width, cnn_height):
    inputs = []

    for _input in explainability_de:
        _input = torch.from_numpy(_input)

        assert len(_input.shape) == 2

        _input = extend_tensor_with_zeros(_input, cnn_width, cnn_height)
        _input = _input.tolist()

        inputs.append(_input)

    inputs = torch.tensor(inputs)
    inputs = inputs.unsqueeze(1) # channel dim
    labels = torch.tensor(labels)

    inputs_expected_shape = (loaded_samples, channels, cnn_width, cnn_height)
    labels_expected_shape = (loaded_samples,)

    assert inputs.shape == inputs_expected_shape, inputs.shape
    assert labels.shape == labels_expected_shape, labels.shape

    return inputs

train_data = read(train_filename)
dev_data = read(dev_filename)
test_data = read(test_filename)
cnn_width = max(train_data["cnn_width"], dev_data["cnn_width"], test_data["cnn_width"])
cnn_height = max(train_data["cnn_height"], dev_data["cnn_height"], test_data["cnn_height"])
train_data_data = get_data(train_data["explainability_de"], train_data["labels"], train_data["loaded_samples"], cnn_width, cnn_height)
dev_data_data  =   get_data(dev_data["explainability_de"], dev_data["labels"], dev_data["loaded_samples"], cnn_width, cnn_height)
test_data_data =  get_data(test_data["explainability_de"], test_data["labels"], test_data["loaded_samples"], cnn_width, cnn_height)
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
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

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

sys.exit(0)

num_classes = 1
epochs = 10
train_dataset = MyDataset(inputs, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = SimpleCNN(channels, cnn_width, cnn_height, num_classes)
loss_function = nn.BCEWithLogitsLoss(reduction="mean")
learning_rate = 1e-5
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(model_parameters, lr=learning_rate)
linear_steps = 0
scheduler = get_inverse_sqrt_schedule(optimizer, linear_steps)

for epoch in range(epochs):
    print(f"Epoch {epoch}")

    for batch_idx, (data, target) in enumerate(train_dataloader, 1):
        print(f"Batch {batch_idx}")

        model.zero_grad()

        model_outputs = model(data)
        outputs = model_outputs
        outputs = outputs.squeeze(1)
        loss = loss_function(outputs, target)

        loss.backward()

        optimizer.step()
        scheduler.step()
