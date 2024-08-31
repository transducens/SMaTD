
import sys

import mtdetect.translation.util as util

import torch
import transformers

source_lang = sys.argv[1] # e.g., en (check https://huggingface.co/facebook/m2m100_1.2B)
target_lang = sys.argv[2]
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 16
pretrained_model = sys.argv[4] if len(sys.argv) > 4 else "facebook/m2m100_418M"

assert batch_size > 0, batch_size

sys.stderr.write(f"Translating from {source_lang} to {target_lang} (pretrained_model={pretrained_model})\n")

model, device = util.load_model(pretrained_model, transformers.AutoModelForSeq2SeqLM)
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=source_lang, tgt_lang=target_lang)
target_lang_id = tokenizer.get_lang_id(target_lang)
beam_size = model.generation_config.num_beams
max_new_tokens = model.generation_config.max_length
early_stopping = True if beam_size > 1 else False

def translate(batch):
    inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True).to(device)
    result = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_new_tokens=max_new_tokens, num_beams=beam_size, early_stopping=early_stopping)
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

