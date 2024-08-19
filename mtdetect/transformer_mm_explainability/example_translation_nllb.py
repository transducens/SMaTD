
import sys

import torch
import transformers
import numpy as np

# NLLB supported languages: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

source_text = sys.argv[1]
target_text = sys.argv[2] if len(sys.argv) > 2 else '' # Teacher forcing
source_lang = sys.argv[3] if len(sys.argv) > 3 else "eng_Latn" # e.g., eng_Latn
target_lang = sys.argv[4] if len(sys.argv) > 4 else "spa_Latn" # e.g., spa_Latn
debug = True # Change manually

print(f"Translating from {source_lang} to {target_lang}")
print(f"Source text: {source_text}")
print(f"Target text: {target_text}")

# Load NLLB

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=source_lang, tgt_lang=target_lang)
max_length = tokenizer.model_max_length
translator_pipeline = None
num_hidden_layers = model.config.num_hidden_layers # Model layers
num_attention_heads = model.config.num_attention_heads # Heads each attention layer has

model.eval()

def get_lang_token(lang):
    token = tokenizer.convert_tokens_to_ids(lang)
    aux_lang = tokenizer.convert_ids_to_tokens(token)

    assert lang == aux_lang, f"{lang} != {aux_lang}"

    return token

target_lang_id = get_lang_token(target_lang)
source_lang_id = get_lang_token(source_lang) # sanity check

# Translate

def decode(translated_tokens):
    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    return output

def translate_from_generate():
    translated_tokens = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_length=max_length)
    translation = decode(translated_tokens)

    return translation

def translate_from_pipeline():
    global translator_pipeline

    if translator_pipeline is None:
        translator_pipeline = transformers.pipeline("translation", model=model, tokenizer=tokenizer, batch_size=batch_size, src_lang=source_lang,
                                                    tgt_lang=target_lang, max_length=max_length, truncation=True, device=device)

    output = translator_pipeline(source_text)
    translation = [_output["translation_text"] for _output in output]

    return translation

inputs = tokenizer(source_text, return_tensors="pt", add_special_tokens=True).to(device)

# We can't use model.generate because we need to apply teacher forcing and need the attention...
# Generation strategy: following https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/generation_config.json
#  ... and default model.generate parameters: https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/text_generation#transformers.GenerationConfig

# https://huggingface.co/facebook/nllb-200-distilled-600M/commit/716b434935682cabef30af23dc1128d84b9003d2:
#
#{
#  "_from_model_config": true,
#  "bos_token_id": 0,
#  "decoder_start_token_id": 2,
#  "eos_token_id": 2,
#  "max_length": 200,
#  "pad_token_id": 1,
#  "transformers_version": "4.27.0.dev0"
#}

decoder_start_token_id = tokenizer.eos_token_id

# _from_model_config : https://github.com/huggingface/transformers/blob/52cb4034ada381fe1ffe8d428a1076e5411a8026/src/transformers/trainer_seq2seq.py#L315
#  ... it handles the generation configuration override by the user, so we do not need to worry about it

teacher_forcing = bool(target_text)
outputs = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist() if teacher_forcing else []
generated_tokens = [decoder_start_token_id, target_lang_id] + outputs # NLLB starts with these two tokens
initial_tokens = len(generated_tokens)

assert batch_size == 1, batch_size

vocab_size = None

# Greedy generation
for i in range(max_length):
    decoder_input_ids = torch.tensor([generated_tokens]).to(device)

    assert decoder_input_ids.shape == (batch_size, i + initial_tokens), decoder_input_ids.shape

    model_output = model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True) # odict_keys(['logits', 'past_key_values', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_attentions'])
    logits = model_output.logits
    vocab_size = logits.shape[-1] if vocab_size is None else vocab_size

    assert logits.shape == (batch_size, i + initial_tokens, vocab_size), f"{logits.shape} ... {vocab_size}"

    logits = logits[:, -1, :] # Get last token logits
    translated_tokens = torch.argmax(logits, dim=-1)

    assert translated_tokens.shape == (batch_size,), translated_tokens.shape

    next_token_id = translated_tokens[-1].cpu().detach().item()

    if teacher_forcing:
        # Next token might be different of EoS -> force
        if next_token_id != tokenizer.eos_token_id:
            if debug:
                print(f"DEBUG: last generated token is not EoS but {tokenizer.convert_ids_to_tokens(next_token_id)} (id: {next_token_id}). This is expected due to teacher forcing")

            next_token_id = tokenizer.eos_token_id

    if next_token_id == tokenizer.eos_token_id:
        break

    generated_tokens.append(next_token_id) # We do not insert the token if it's the last because it's not provided to the model in the next iteration

# Attention, and gradients hook

attentions_grad_store = {}

def capture_attention_grads(layer, component):
    def _f(grad):
        if layer not in attentions_grad_store:
            attentions_grad_store[layer] = {}

        assert component not in attentions_grad_store[layer]

        attentions_grad_store[layer][component] = grad

    return _f

assert len(model_output.encoder_attentions) == num_hidden_layers, f"{len(model_output.encoder_attentions)} != {num_hidden_layers}"
assert len(model_output.decoder_attentions) == num_hidden_layers, f"{len(model_output.decoder_attentions)} != {num_hidden_layers}"
assert len(model_output.cross_attentions) == num_hidden_layers, f"{len(model_output.cross_attentions)} != {num_hidden_layers}"

attentions = {l: {
    "encoder": model_output.encoder_attentions[l],
    "decoder": model_output.decoder_attentions[l],
    "cross": model_output.cross_attentions[l],
    } for l in range(num_hidden_layers)}

for layer in range(num_hidden_layers):
    for component in ("encoder", "decoder", "cross"):
        attentions[layer][component].requires_grad

        attentions[layer][component].retain_grad()
        attentions[layer][component].register_hook(capture_attention_grads(layer, component))

source_text_seq_len = len(inputs.input_ids[0])
target_text_seq_len = len(generated_tokens)
attention_expected_shape = {
    "encoder": (batch_size, num_attention_heads, source_text_seq_len, source_text_seq_len),
    "decoder": (batch_size, num_attention_heads, target_text_seq_len, target_text_seq_len),
    "cross": (batch_size, num_attention_heads, target_text_seq_len, source_text_seq_len),
    }

for l in range(num_hidden_layers):
    for component in ("encoder", "decoder", "cross"):
        assert attentions[l][component].shape == attention_expected_shape[component], f"{l}: {attentions[l][component].shape} != {attention_expected_shape[component]}"

translated_tokens = torch.tensor([generated_tokens]).to(device)

# Calculate gradients

logits = model_output.logits
logits_expected_shape = (batch_size, len(generated_tokens), vocab_size)

assert logits.shape == logits_expected_shape, f"{logits.shape} != {logits_expected_shape}"

target = torch.zeros(logits_expected_shape)

for i, token in enumerate(generated_tokens):
    target[:, i, token] = 1.0

target = target.to(device)
loss = torch.sum(logits * target)

assert len(attentions_grad_store) == 0, f"{len(attentions_grad_store)} != 0"

model.zero_grad()
loss.backward(retain_graph=True)

assert len(attentions_grad_store) == num_hidden_layers, f"{len(attentions_grad_store)} != {num_hidden_layers}"

# Decode

output = decode(translated_tokens)

for translated_text in output:
    print(translated_text)

if debug:
    from_generate = translate_from_generate()

    assert len(from_generate) == len(output)

    for o1, o2 in zip(output, from_generate):
        if o1 != o2:
            print(f"DEBUG: warning: different result than using generate (this is expected if teacher forcing is enabled): {o1} vs {o2}")

    from_pipeline = translate_from_pipeline()

    assert len(from_pipeline) == len(output)

    for o1, o2 in zip(output, from_pipeline):
        if o1 != o2:
            print(f"DEBUG: warning: different result than using pipeline (this is expected if teacher forcing is enabled): {o1} vs {o2}")

# Apply explainability
# i -> source, t -> target (following paper nomenclature)

r_ii = np.identity(source_text_seq_len)
r_tt = np.identity(target_text_seq_len)
r_it = np.zeros((source_text_seq_len, target_text_seq_len)) # influence of the input text on the translated text
#r_ti = np.zeros((target_text_seq_len, source_text_seq_len)) # We only have co-attention in the decoder (i.e., r_it)
a_line = {}

for layer in range(num_hidden_layers):
    a_line[layer] = {}

    for component in ("encoder", "decoder", "cross"):
        assert component not in a_line[layer]

        gradient = attentions_grad_store[layer][component]
        attention = attentions[layer][component]

        assert gradient.shape == attention.shape, f"{gradient.shape} != {attention.shape}"

        a_line_aux = gradient * attention

        assert a_line_aux.shape == attention.shape, f"{a_line_aux.shape} != {attention.shape}"
        assert a_line_aux.shape == attention_expected_shape[component], f"{a_line_aux.shape} != {attention_expected_shape[component]}"

        a_line_aux = torch.max(a_line_aux, torch.zeros_like(a_line_aux)).cpu().detach()
        a_line_aux = torch.mean(a_line_aux, dim=1)

        a_line[layer][component] = a_line_aux

        assert a_line[layer][component].shape == (batch_size, *attention_expected_shape[component][-2:]), a_line[layer].shape

assert len(a_line) == num_hidden_layers

# Update attention layers

for layer in range(num_hidden_layers):

    # equation 6
    for component, r in (("encoder", r_ii), ("decoder", r_tt)):
        a_line_aux = a_line[layer][component][0].numpy() # current layer, and batch size is 0

        assert a_line_aux.shape == attention_expected_shape[component][-2:], a_line_aux.shape
        assert a_line_aux.shape == r.shape

        r += np.matmul(a_line_aux, r)

    # equation 7
    for component, r in (("encoder", r_it),): # is this equation useful for anything...??? (result is always 0...) (for cross attention we use the encoder A variable according
                                              #  ... to https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/lxmert/lxmert/src/ExplanationGenerator.py#L169 )
        a_line_aux = a_line[layer][component][0].numpy()

        assert a_line_aux.shape == attention_expected_shape[component][-2:], a_line_aux.shape

        r += np.matmul(a_line_aux, r)

        assert (r == np.zeros_like(r)).all()

    # TODO equation 8, 9, 10, and 11
