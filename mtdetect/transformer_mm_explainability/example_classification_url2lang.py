
import re
import sys
import urllib.parse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk.tokenize
import torch
import numpy as np

url = sys.argv[1] # e.g., https://es.wikipedia.org/wiki/Halo_3#Matchmaking
target_class = None if len(sys.argv) < 3 else sys.argv[2]
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

# Get gradients

#gradients_components = {}
#
#for name, param in model.named_parameters():
#    gradient = param.grad
#
#    if gradient is None:
#        continue
#
#    # Model specific
#    once = False
#
##    for attention_component in ("query", "key", "value"):
##        for gradient_component in ("weight", "bias"):
##            if name.endswith(f"attention.self.{attention_component}.{gradient_component}"):
##                assert not once
##
##                once = True
##                layer = int(name.split('.')[3])
##
##                if layer not in gradients_components:
##                    gradients_components[layer] = {}
##                if attention_component not in gradients_components[layer]:
##                    gradients_components[layer][attention_component] = {}
##
##                assert gradient_component not in gradients_components[layer][attention_component], f"{layer}: {gradient_component}: {gradients_components[layer].keys()}: {gradients_components[layer]}"
##
##                gradients_components[layer][attention_component][gradient_component] = gradient
#
#    for gradient_component in ("weight", "bias"):
#        if name.endswith(f"attention.output.LayerNorm.{gradient_component}"): # TODO "dense" or "LayerNorm"?
#            assert not once
#
#            once = True
#            layer = int(name.split('.')[3])
#
#            if layer not in gradients_components:
#                gradients_components[layer] = {}
#
#            assert gradient_component not in gradients_components[layer], f"{layer}: {gradient_component}: {gradients_components[layer].keys()}: {gradients_components[layer]}"
#
#            gradients_components[layer][gradient_component] = gradient
#
#assert len(gradients_components) == num_hidden_layers, f"{len(gradients_components)} != {num_hidden_layers}"
#assert gradients_components.keys() == attention_components.keys(), f"{gradients_components.keys()} != {attention_components.keys()}"

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
#r_tt = np.zeros((seq_len, seq_len)) # text

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
#r_tt = torch.softmax(torch.tensor(r_tt), dim=0).numpy() # Easily interpreted
r_tt = (r_tt - r_tt.min()) / (r_tt.max() - r_tt.min()) # min-max normalization

for priority, token in zip(r_tt, input_tensor_decoded):
    print(f"{token}\t{priority}")
