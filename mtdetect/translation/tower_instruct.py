
import sys

import mtdetect.translation.util as util

import torch
from transformers import pipeline

source_lang = sys.argv[1] # e.g., Portuguese (check example in https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2)
target_lang = sys.argv[2] # e.g., Spanish
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 4
pretrained_model = sys.argv[4] if len(sys.argv) > 4 and len(sys.argv[4]) > 0 else "Unbabel/TowerInstruct-7B-v0.2"

assert batch_size > 0, batch_size

sys.stderr.write(f"Translating from {source_lang} to {target_lang} (pretrained_model={pretrained_model})\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("text-generation", model=pretrained_model, torch_dtype=torch.bfloat16, device_map=device)

def translate(batch, device):
    _pipe = pipe

    if _pipe.device != device:
        _pipe.model = pipe.model.to(device)

    prompt = _pipe.tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
    output = _pipe(prompt, max_new_tokens=256, do_sample=False)
    output = [o["generated_text"] for o in output]

    assert len(output) == len(batch)

    for idx, o in enumerate(output):
        assert isinstance(o, str)

        i = o.find("\n<|im_start|>assistant\n")

        if i < 0:
            i = -23 # -23 + 23 = 0 -> whole sentence

            sys.stderr.write(f"ERROR: unexpected output for the following prompt: {batch[idx]['content']}\n")

        output[idx] = ' '.join(list(filter(lambda s: len(s) > 0, map(lambda s: s.strip(), o[i + 23:].split('\n')))))

    return output

batch = []
sentences = []
current_patience = 0

for l in sys.stdin:
    l = l.rstrip("\r\n")

    sentences.append(l)
    batch.append({"role": "user", "content": f"Translate the following text from {source_lang} into {target_lang}.\n{source_lang}: {sentences[-1]}\n{target_lang}:"})

    if len(batch) >= batch_size:
        translations, current_patience = util.translate_oom_aware(batch, translate, device, current_patience=current_patience)

        util.print_translation(translations, sentences)

        batch = sentences = []

if len(batch) > 0:
    translations, current_patience = util.translate_oom_aware(batch, translate, device, current_patience=current_patience)

    util.print_translation(translations, sentences)

    batch = sentences = []
