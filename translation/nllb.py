
import sys

import torch
import transformers

source_lang = sys.argv[1]
target_lang = sys.argv[2]
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 16
beam_size = int(sys.argv[4]) if len(sys.argv) > 4 and len(sys.argv[4]) > 0 else 1
pretrained_model = sys.argv[5] if len(sys.argv) > 5 else "facebook/nllb-200-distilled-600M"

assert batch_size > 0, batch_size
assert beam_size > 0, beam_size

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
    output = []
    bs = len(batch)

    while True:
        _batch = batch[len(output):len(output) + bs]

        try:
            inputs = tokenizer(_batch, return_tensors="pt", add_special_tokens=True, truncation=True).to(device)
            result = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_new_tokens=model.generation_config.max_length, num_beams=beam_size)
            _output = tokenizer.batch_decode(result, skip_special_tokens=True)

            output.extend(_output)
        except torch.OutOfMemoryError as e:
            if len(_batch) == 1:
                raise Exception("torch.OutOfMemoryError even using batch_size=1") from e

            sys.stderr.write(f"torch.OutOfMemoryError error: current batch size is {len(_batch)}: using half batch_size\n")

            bs = len(_batch) // 2

        if len(output) >= len(batch):
            assert len(output) == len(batch)

            break

    return output

def print_translation(translations, batch):
    assert len(translations) == len(batch)

    for sentence, translation in zip(batch, translations):
        print(f"{sentence}\t{translation}")

    sys.stdout.flush()

    return []

batch = []

for l in sys.stdin:
    l = l.rstrip("\r\n")

    batch.append(l)

    if len(batch) >= batch_size:
        translations = translate(batch)
        batch = print_translation(translations, batch)

if len(batch) > 0:
    translations = translate(batch)
    batch = print_translation(translations, batch)
