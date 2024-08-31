
import sys

import mtdetect.translation.util as util

import torch
import transformers

pretrained_model = sys.argv[1] # e.g., Helsinki-NLP/opus-mt-en-ha (check https://huggingface.co/Helsinki-NLP)
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 16

assert batch_size > 0, batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
source_lang = tokenizer.source_lang
target_lang = tokenizer.target_lang
forced_bos_token_id = model.generation_config.decoder_start_token_id
beam_size = model.generation_config.num_beams
max_new_tokens = model.generation_config.max_length
early_stopping = True if beam_size > 1 else False

sys.stderr.write(f"Translating from {source_lang} to {target_lang} (pretrained_model={pretrained_model})\n")

def translate(batch):
    inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
    result = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=max_new_tokens, num_beams=beam_size, early_stopping=early_stopping)
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

