
import sys

import mtdetect.translation.util as util

import torch
import transformers

source_lang = sys.argv[1] # e.g., eng_Latn (check https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
target_lang = sys.argv[2]
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 16
pretrained_model = sys.argv[4] if len(sys.argv) > 4 else "facebook/nllb-200-600M"
beam_size = int(sys.argv[5]) if len(sys.argv) > 5 and len(sys.argv[5]) > 0 else 4 # 4 is the value used in the paper to train the destilled models

assert batch_size > 0, batch_size
assert beam_size > 0, beam_size

sys.stderr.write(f"Translating from {source_lang} to {target_lang} (pretrained_model={pretrained_model} ; beam_size={beam_size})\n")

def get_lang_token(tokenizer, lang):
    token = tokenizer.convert_tokens_to_ids(lang)
    aux_lang = tokenizer.convert_ids_to_tokens(token)

    assert lang == aux_lang, f"{lang} != {aux_lang}"

    return token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=source_lang, tgt_lang=target_lang)
target_lang_id = get_lang_token(tokenizer, target_lang)

def translate(batch):
    inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True).to(device)
    result = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_new_tokens=model.generation_config.max_length, num_beams=beam_size)
    output = tokenizer.batch_decode(result, skip_special_tokens=True)

    return output

batch = []

for l in sys.stdin:
    l = l.rstrip("\r\n")

    batch.append(l)

    if len(batch) >= batch_size:
        translations = util.translate_oom_aware(batch, translate)
        batch = util.print_translation(translations, batch)

if len(batch) > 0:
    translations = util.translate_oom_aware(batch, translate)
    batch = util.print_translation(translations, batch)
