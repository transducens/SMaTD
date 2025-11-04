
import re
import sys

import torch
import numpy as np
import transformers
from PIL import Image, ImageFont, ImageDraw

import mtdetect.utils.utils as utils

def load_model(model_input, pretrained_model, device):
    local_model = model_input is not None
    config = transformers.AutoConfig.from_pretrained(pretrained_model, num_labels=1)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
    loaded_model = f"{pretrained_model}:{model_input}" if local_model else pretrained_model

    if local_model:
        state_dict = torch.load(model_input, weights_only=True, map_location=device) # weights_only: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
                                                                                      # map_location: avoid creating a new process and using additional and useless memory

        model.load_state_dict(state_dict)

    model = model.to(device)

    return model

def colorize_background(segments, intensities, font_size=40, output_image="output.png"):
    segments = [s.replace('▁', '_') for s in segments]

    if len(segments) != len(intensities):
        raise ValueError("Segments and intensities lists must have the same length")

    # Initialize font and image parameters
    font = ImageFont.load_default(font_size)
    width, height = 0, 0

    # Calculate total image width and maximum height
    for segment in segments:
        left, top, right, bottom = font.getbbox(segment)
        text_width = right - left
        text_height = bottom - top
        width += text_width
        height = max(height, text_height)

    height = int(height * 1.3)

    # Create an image with a white background
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Start drawing text at the leftmost position
    x_position = 0

    # Saturation?
    satured = (intensities < 0.0).any() or (intensities > 1.0).any()

    if satured:
        print(f"warning: intensity values are not in [0, 1] and color will be saturated: {output_image}")

    for segment, intensity in zip(segments, intensities):
        # Map the intensity to a red color (intensity 0 -> white, intensity 1 -> red)
        intensity = min(max(intensity, 0.0), 1.0)
        green_blue_value = int(255 * (1 - intensity))
        background_color = (255, green_blue_value, green_blue_value)

        # Calculate the size of the text segment
        left, top, right, bottom = font.getbbox(segment)
        text_width = right - left
        text_height = bottom - top

        # Draw the background rectangle
        draw.rectangle([x_position, 0, x_position + text_width, height], fill=background_color)

        # Draw the text segment on top of the background
        draw.text((x_position, 0), segment, font=font, fill="black")

        # Update the x_position for the next segment
        x_position += text_width

    # Save the image
    img.save(output_image)

src_sentence = sys.argv[1]
trg_sentence = sys.argv[2]
target_class = sys.argv[3].lower() if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else None # e.g., mt
model_input = sys.argv[4] if len(sys.argv) > 4 and len(sys.argv[4]) > 0 else None
colorize_output = sys.argv[5] if len(sys.argv) > 5 and len(sys.argv[5]) > 0 else ''

assert target_class in ("mt", "ht", None), target_class

pretrained_model = "xlm-roberta-base"

print(f"Pretrained model: {pretrained_model} (input model: {model_input})")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = utils.get_tokenizer(pretrained_model)
model = load_model(model_input, pretrained_model, device)
attention_components = {}

model.eval()

for name, module in model.named_modules():
    # Model specific
    once = False

    for attention_component in ("query", "key", "value"):
        if name.endswith(f"attention.self.{attention_component}"):
            assert not once

            once = True
            layer = int(name.split('.')[3])

            if layer not in attention_components:
                attention_components[layer] = {}

            module.register_forward_hook((lambda _layer, _attention_component: lambda module, input, output: attention_components[_layer].update({_attention_component: output}))(layer, attention_component))

assert len(attention_components) > 0

# prepare input
max_length_tokens = utils.get_encoder_max_length(model, tokenizer, max_length_tokens=512, pretrained_model=pretrained_model, logger=None)
sentences = f"{src_sentence}{tokenizer.sep_token}{trg_sentence}"
encoded_input = tokenizer(sentences, add_special_tokens=True, truncation=True, padding="do_not_pad",
                          return_attention_mask=True, return_tensors="pt", max_length=max_length_tokens).to(device)

# forward pass
output = model(encoded_input["input_ids"], encoded_input["attention_mask"], output_attentions=True)

# obtain probability
probability = torch.sigmoid(output["logits"]).cpu().squeeze().item()
threshold = 0.5
ht_vs_mt_class = 0 if probability < threshold else 1
ht_vs_mt_str = "MT" if probability < threshold else "HT"

print(f"HT vs MT (probability): {ht_vs_mt_str} ({probability})")

##########################################

if target_class is None:
    target_class = ht_vs_mt_class
else:
    target_class = 0 if target_class == "mt" else 1 if target_class == "ht" else None

    assert target_class is not None

target = torch.eye(1)
batch_size = output["logits"].shape[0]

assert output["logits"].shape, target.shape
assert output["logits"].requires_grad

target = target.to(device)
loss = torch.sum(output["logits"] * target)
input_tensor = encoded_input["input_ids"]

assert input_tensor.shape[0] == batch_size, input_tensor
assert batch_size == 1, batch_size

input_tensor = input_tensor.squeeze(0).cpu().detach().tolist()
input_tensor_decoded = tokenizer.convert_ids_to_tokens(input_tensor)
seq_len = len(input_tensor)

assert seq_len == len(input_tensor_decoded)

print(f"Tokens: {input_tensor_decoded}")
print(f"Length: {seq_len}")

# Get attention layers (both result and gradients)

attentions = output["attentions"]
num_hidden_layers = model.config.num_hidden_layers # Model layers
attentions_grad_store = {}

def capture_attention_grads(layer):
    def _f(grad):
        assert layer not in attentions_grad_store

        attentions_grad_store[layer] = grad

    return _f

for layer in range(num_hidden_layers):
    assert attentions[layer].requires_grad

    attentions[layer].retain_grad()
    attentions[layer].register_hook(capture_attention_grads(layer))

num_attention_heads = model.config.num_attention_heads # Heads each attention layer has
hidden_size = model.config.hidden_size
attention_expected_shape = (batch_size, num_attention_heads, seq_len, seq_len)
attention_components_expected_shape = (batch_size, seq_len, hidden_size)

assert len(attentions) == num_hidden_layers, f"{len(attentions)} != {num_hidden_layers}"
assert len(attention_components) == num_hidden_layers, f"{len(attention_components)} != {num_hidden_layers}"
assert hidden_size % num_attention_heads == 0, f"{hidden_size} % {num_attention_heads} != 0"

attention_dim = hidden_size // num_attention_heads
attention_components_reshape_expected_shape = (batch_size, num_attention_heads, seq_len, attention_dim)

for i in range(1, num_hidden_layers):
    assert attentions[i].shape == attentions[i - 1].shape, f"{i - 1}: {attentions[i].shape} != {attentions[i - 1].shape}"

    for l in ("query", "key", "value"):
        assert attention_components[i][l].shape == attention_components[i - 1][l].shape, f"{i - 1}: {attention_components[i][l].shape} != {attention_components[i - 1][l].shape}"

for i in range(num_hidden_layers):
    assert attentions[i].shape == attention_expected_shape, f"{attentions[0].shape} != {attention_expected_shape}"
    assert len(attention_components[i]) == 3, f"{i} {attention_components[i]}"

    for l in ("query", "key", "value"):
        assert l in attention_components[i], attention_components[i]
        assert attention_components[i][l].shape == attention_components_expected_shape, f"{attention_components[i][l].shape} != {attention_components_expected_shape}"

        attention_components[i][l] = attention_components[i][l].view(batch_size, seq_len, num_attention_heads, attention_dim).permute(0, 2, 1, 3)

        assert attention_components[i][l].shape == attention_components_reshape_expected_shape, f"{attention_components[i][l].shape} != {attention_components_reshape_expected_shape}"

# Sanity check attentions

for layer in range(num_hidden_layers):
    q, k, v = [attention_components[layer][l] for l in ("query", "key", "value")]
    a = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(attention_dim)
    a = torch.softmax(a, dim=-1)

    assert a.shape == attention_expected_shape
    assert a.shape == attentions[layer].shape
    assert (a == attentions[layer]).all().item(), f"{(a - attentions[layer]).sum().item()}: {a - attentions[layer]}"

    #o = torch.matmul(a, v)

assert len(attentions_grad_store) == 0, f"{len(attentions_grad_store)} != 0"

model.zero_grad()
loss.backward(retain_graph=True)

assert len(attentions_grad_store) == num_hidden_layers, f"{len(attentions_grad_store)} != {num_hidden_layers}"
assert attentions_grad_store.keys() == attention_components.keys(), f"{attentions_grad_store.keys()} != {attention_components.keys()}"

##########################################

a_line = []

for layer in range(num_hidden_layers):
    gradient = attentions_grad_store[layer]
    attention = attentions[layer]

    assert gradient.shape == attention.shape, f"{gradient.shape} != {attention.shape}"

    a_line_aux = gradient * attention

    assert a_line_aux.shape == attention.shape, f"{a_line_aux.shape} != {attention.shape}"
    assert a_line_aux.shape == attention_expected_shape, f"{a_line_aux.shape} != {attention_expected_shape}"

    a_line_aux = torch.max(a_line_aux, torch.zeros_like(a_line_aux)).cpu().detach()
    a_line_aux = torch.mean(a_line_aux, dim=1)

    a_line.append(a_line_aux)

assert len(a_line) == num_hidden_layers
assert a_line[0].shape == (batch_size, seq_len, seq_len), a_line[0].shape

# a_line shape: (num_hidden_layers, batch_size, seq_len, seq_len)

# Update
r_tt = np.identity(seq_len) # text

assert r_tt.shape == (seq_len, seq_len), r_tt.shape

for layer in range(num_hidden_layers):
    a_line_aux = a_line[layer][0].numpy() # current layer, and batch size is 0

    assert a_line_aux.shape == (seq_len, seq_len), a_line_aux.shape
    assert a_line_aux.shape == r_tt.shape

    r_tt = r_tt + np.matmul(a_line_aux, r_tt)

assert r_tt.shape == (seq_len, seq_len), r_tt.shape

r_tt = r_tt[0] # Classification: according to the paper, get first token (i.e., CLS token)
r_tt = r_tt[1:-1] # Remove special tokens
input_tensor_decoded = input_tensor_decoded[1:-1] # Remove special tokens
r_tt = (r_tt - r_tt.min()) / (r_tt.max() - r_tt.min()) # min-max normalization

for priority, token in zip(r_tt, input_tensor_decoded):
    print(f"{token}\t{priority}")

if colorize_output:
    colorize_background(input_tensor_decoded, r_tt, output_image=colorize_output)

    print(f"Image with tokens and intensities stored: {colorize_output}")
