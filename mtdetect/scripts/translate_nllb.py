
import sys
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

#checkpoint = "facebook/nllb-200-distilled-600M"
# checkpoint = "facebook/nllb-200–1.3B"
# checkpoint = "facebook/nllb-200–3.3B"
#checkpoint = "facebook/nllb-200-distilled-1.3B"
checkpoint = "facebook/nllb-moe-54b"

device = "cuda"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#source_lang = "eng_Latn"
source_lang = sys.argv[1]
#target_lang = "spa_Latn"
target_lang = sys.argv[2]
text = []

for l in sys.stdin:
    text.append(l.strip())

batch_size = 128

def get_translator(batch_size):
    return pipeline("translation", batch_size=batch_size, model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, max_length=tokenizer.model_max_length, device=device, truncation=True)

translator = get_translator(batch_size)

while batch_size > 0:
    try:
        output = translator(text)

        break
    except torch.OutOfMemoryError:
        _batch_size = batch_size // 2

        sys.stderr.write(f"OOM: {batch_size} -> {_batch_size}\n")

        batch_size = _batch_size
        translator = get_translator(batch_size)

if batch_size > 0:

    for _output in output:
        translated_text = _output["translation_text"]

        print(translated_text)
else:
    sys.stderr.write("Batch size <= 0 ...\n")