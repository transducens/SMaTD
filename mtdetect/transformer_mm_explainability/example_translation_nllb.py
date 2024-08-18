
import sys

import torch
import transformers

# NLLB supported languages: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

source_text = sys.argv[1]
source_lang = sys.argv[2] if len(sys.argv) > 2 else "eng_Latn" # e.g., eng_Latn
target_lang = sys.argv[3] if len(sys.argv) > 3 else "spa_Latn" # e.g., spa_Latn
debug = True # Change manually

print(f"Translating from {source_lang} to {target_lang}: {source_text}")

# Load NLLB

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=source_lang, tgt_lang=target_lang)
max_length = tokenizer.model_max_length
translator_pipeline = None

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

inputs = tokenizer(source_text, return_tensors="pt").to(device)

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

generated_tokens = [decoder_start_token_id, target_lang_id] # NLLB starts with these two tokens
initial_tokens = len(generated_tokens)

assert batch_size == 1, batch_size

vocab_size = None

# Greedy generation
for i in range(max_length):
    decoder_input_ids = torch.tensor([generated_tokens]).to(device)

    assert decoder_input_ids.shape == (batch_size, i + initial_tokens), decoder_input_ids.shape

    model_output = model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)
    logits = model_output.logits
    vocab_size = logits.shape[-1] if vocab_size is None else vocab_size

    assert logits.shape == (batch_size, i + initial_tokens, vocab_size), f"{logits.shape} ... {vocab_size}"

    logits = logits[:, -1, :] # Get last token logits
    translated_tokens = torch.argmax(logits, dim=-1)

    assert translated_tokens.shape == (batch_size,), translated_tokens.shape

    next_token_id = translated_tokens[-1].cpu().detach().item()

    generated_tokens.append(next_token_id)

    if next_token_id == tokenizer.eos_token_id:
        break

translated_tokens = torch.tensor([generated_tokens]).to(device)

# Decode

output = decode(translated_tokens)

for translated_text in output:
    print(translated_text)

if debug:
    from_generate = translate_from_generate()

    print(f"DEBUG: from generate(): {from_generate}")

    assert len(from_generate) == len(output)

    for o1, o2 in zip(output, from_generate):
        assert o1 == o2, f"{o1} vs {o2}"

    from_pipeline = translate_from_pipeline()

    print(f"DEBUG: from pipeline(): {from_pipeline}")

    assert len(from_pipeline) == len(output)

    for o1, o2 in zip(output, from_pipeline):
        assert o1 == o2, f"{o1} vs {o2}"
