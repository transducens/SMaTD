
import re
import sys
import urllib.parse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk.tokenize
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw

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

    for segment, intensity in zip(segments, intensities):
        # Map the intensity to a red color (intensity 0 -> white, intensity 1 -> red)
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

url = sys.argv[1] # e.g., https://es.wikipedia.org/wiki/Halo_3#Matchmaking
target_class = sys.argv[2] if len(sys.argv) > 2 and len(sys.argv[2]) > 0 else None # e.g., spa
colorize_output = sys.argv[3] if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else ''
preprocess_tokenizer_regex = r'[^\W_0-9]+|[^\w\s]+|_+|\s+|[0-9]+' # Similar to wordpunct_tokenize
preprocess_tokenizer = nltk.tokenize.RegexpTokenizer(preprocess_tokenizer_regex).tokenize

def preprocess_url(url):
    protocol_idx = url.find("://")
    protocol_idx = (protocol_idx + 3) if protocol_idx != -1 else 0
    url = url.rstrip('/')[protocol_idx:]
    url = urllib.parse.unquote(url, errors="backslashreplace")

    # Remove blanks
    url = re.sub(r'\s+', ' ', url)
    url = re.sub(r'^\s+|\s+$', '', url)

    # Tokenize
    url = ' '.join(preprocess_tokenizer(url))

    return url

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Transducens/xlm-roberta-base-url2lang")
model = AutoModelForSequenceClassification.from_pretrained("Transducens/xlm-roberta-base-url2lang").to(device)
attention_components = {}

#model.train()
model.eval()

#for name, param in model.named_parameters(): # TODO remove
#    print(name)
#
#for name, param in model.named_modules(): # TODO remove
#    print(name)

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
url = preprocess_url(url)
encoded_input = tokenizer(url, add_special_tokens=True, truncation=True, padding="longest",
                          return_attention_mask=True, return_tensors="pt", max_length=256).to(device)

# forward pass
output = model(encoded_input["input_ids"], encoded_input["attention_mask"], output_attentions=True)

# obtain lang
probabilities = torch.softmax(output["logits"], dim=1).cpu().squeeze(0)
lang_idx = torch.argmax(probabilities, dim=0).item()
probability = probabilities[lang_idx].item()
lang = model.config.id2lang[str(lang_idx)]

print(f"Language (probability): {lang} ({probability})")

##########################################

lang2id = {lang: int(_id) for _id, lang in model.config.id2lang.items()}

if target_class is None:
    target_class = lang_idx
else:
    assert target_class in lang2id, f"{target_class} not supported: {lang2id.keys()}"

    target_class = lang2id[target_class]

target_lang = model.config.id2lang[str(target_class)]

print(f"Target lang: {target_lang}")

criterion = torch.nn.BCEWithLogitsLoss()
target = torch.eye(len(model.config.id2lang))[target_class].unsqueeze(0)
batch_size = output["logits"].shape[0]

assert output["logits"].shape, target.shape
assert output["logits"].requires_grad

target = target.to(device)
#loss = criterion(output["logits"], target)
loss = torch.sum(output["logits"] * target) # Why is this the loss?????? Taken from: https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_explainability_ViT.ipynb
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

##########################################
# Results for https://es.wikipedia.org/wiki/Halo_3#Matchmaking
########################################## spa (it makes sense that "▁es" is highlighted, as it is the main indication that the URL links to a Spanish document)
# ▁es     1.0
# ▁       0.655019430343012
# .       0.7637193536636468
# ▁       0.49272859413346826
# wikipedia       0.4773227022720526
# ▁       0.5286190934140631
# .       0.6222616652878866
# ▁org    0.2968295510435883
# ▁/      0.5423566155537004
# ▁wiki   0.4229652649238778
# ▁/      0.7234402508753892
# ▁Halo   0.48151405250875334
# ▁_      0.7452003770495972
# ▁3      0.04804429453348593
# ▁#      0.2542532853981647
# ▁Match  0.0
# making  0.1237987899177173
########################################## eng (it makes sense that the only English word has the most relevance: "▁Match" and "making")
# ▁es     0.43261021987432907
# ▁       0.10096244200177303
# .       0.14804454251464105
# ▁       0.0446340194062123
# wikipedia       0.1581167806711341
# ▁       0.07818782228045079
# .       0.11523778021161747
# ▁org    0.0
# ▁/      0.1164210304358272
# ▁wiki   0.19257041756315163
# ▁/      0.2127387219243301
# ▁Halo   0.5380063625371798
# ▁_      0.3879838679373022
# ▁3      0.5102079610685366
# ▁#      0.29790781221112145
# ▁Match  0.9626208961637784
# making  1.0
##########################################
