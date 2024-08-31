
import sys

import mtdetect.translation.util as util

import transformers

# Source lang can't be specified

target_lang = sys.argv[1] # e.g., ha (check https://huggingface.co/google/madlad400-10b-mt/discussions/2)
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 and len(sys.argv[2]) > 0 else 16
pretrained_model = sys.argv[3] if len(sys.argv) > 3 else "google/madlad400-3b-mt"
beam_size = int(sys.argv[4]) if len(sys.argv) > 4 and len(sys.argv[4]) > 0 else 1

assert batch_size > 0, batch_size
assert beam_size > 0, beam_size

sys.stderr.write(f"Translating from - to {target_lang} (pretrained_model={pretrained_model} ; beam_size={beam_size})\n")

model, device = util.load_model(pretrained_model, transformers.AutoModelForSeq2SeqLM)
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
max_new_tokens = 1024 # not available: model.generation_config.max_length
early_stopping = True if beam_size > 1 else False

def translate(batch):
    inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, max_length=max_new_tokens, truncation=True, padding=True).to(device)
    result = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beam_size, early_stopping=early_stopping)
    output = tokenizer.batch_decode(result, skip_special_tokens=True)

    return output

batch = []
original_text = []

for l in sys.stdin:
    l = l.rstrip("\r\n")

    original_text.append(l)

    l = f"<2{target_lang}> {l}"

    batch.append(l)

    if len(batch) >= batch_size:
        translations = util.translate_oom_aware(batch, translate)
        original_text = util.print_translation(translations, original_text)
        batch = []

if len(batch) > 0:
    translations = util.translate_oom_aware(batch, translate)
    original_text = util.print_translation(translations, original_text)
