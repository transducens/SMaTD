
import sys

import mtdetect.translation.util as util

import torch
import transformers

source_lang = sys.argv[1] # e.g., en (check https://huggingface.co/facebook/m2m100_1.2B)
target_lang = sys.argv[2]
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 16
pretrained_model = sys.argv[4] if len(sys.argv) > 4 else "facebook/m2m100_1.2B"
beam_size = int(sys.argv[5]) if len(sys.argv) > 5 and len(sys.argv[5]) > 0 else 5 # Paper: "Unless otherwise specified: [...], use beam search with beam 5"

assert batch_size > 0, batch_size
assert beam_size > 0, beam_size

sys.stderr.write(f"Translating from {source_lang} to {target_lang} (pretrained_model={pretrained_model} ; beam_size={beam_size})\n")

model, device = util.load_model(pretrained_model, transformers.AutoModelForSeq2SeqLM)
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=source_lang, tgt_lang=target_lang)
target_lang_id = tokenizer.get_lang_id(target_lang)
max_new_tokens = 1024 # not available: model.generation_config.max_length

def translate(batch, device):
    _model = model

    if _model.device != device:
        _model = model.to(device)

    inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, max_length=max_new_tokens, truncation=True, padding=True).to(device)
    result = _model.generate(**inputs, forced_bos_token_id=target_lang_id, max_new_tokens=max_new_tokens, num_beams=beam_size)
    output = tokenizer.batch_decode(result, skip_special_tokens=True)

    return output

batch = []
current_patience = 0

for l in sys.stdin:
    l = l.rstrip("\r\n")

    batch.append(l)

    if len(batch) >= batch_size:
        translations, current_patience = util.translate_oom_aware(batch, translate, device, current_patience=current_patience)
        batch = util.print_translation(translations, batch)

if len(batch) > 0:
    translations, current_patience = util.translate_oom_aware(batch, translate, device, current_patience=current_patience)
    batch = util.print_translation(translations, batch)

