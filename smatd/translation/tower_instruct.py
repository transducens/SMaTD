
import sys

import smatd.translation.util as util

import torch
import transformers

source_lang = sys.argv[1] # e.g., Portuguese (check example in https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2)
target_lang = sys.argv[2] # e.g., Spanish
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 and len(sys.argv[3]) > 0 else 16
pretrained_model = sys.argv[4] if len(sys.argv) > 4 and len(sys.argv[4]) > 0 else "Unbabel/TowerInstruct-7B-v0.2"
beam_size = int(sys.argv[5]) if len(sys.argv) > 5 and len(sys.argv[5]) > 0 else 1
mbr = bool(int(sys.argv[6])) if len(sys.argv) > 6 and len(sys.argv[6]) > 0 else False

assert batch_size > 0, batch_size
assert beam_size > 0, beam_size

sys.stderr.write(f"Translating from {source_lang} to {target_lang} (pretrained_model={pretrained_model} ; beam_size={beam_size} ; mbr={mbr})\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_init_kwargs = {}
pipeline_call_kwargs = {}
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)

if not mbr:
    model = transformers.LlamaForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
    pipeline_init_kwargs["num_beams"] = beam_size
    pipeline_call_kwargs["do_sample"] = False
else:
    assert beam_size == 1, "We use the decoding strategy (nucleus sampling p=0.6 with temperature=0.9) used in the Tower paper (Appendix A): https://arxiv.org/abs/2402.17733"

    import mbr
    import evaluate

    model = mbr.MBR(transformers.LlamaForCausalLM).from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
    metric = evaluate.load("comet", "Unbabel/wmt22-comet-da")
    mbr_config = mbr.MBRConfig(
        num_samples=20,
        metric="chrf",
        #metric="comet",
        #metric_output_field="mean_score",
    )
    pipeline_call_kwargs["mbr_config"] = mbr_config
    pipeline_call_kwargs["do_sample"] = True
    pipeline_call_kwargs["temperature"] = 0.9
    pipeline_call_kwargs["top_p"] = 0.6
    pipeline_call_kwargs["tokenizer"] = tokenizer

model = model.to(device)
#pipe = pipeline("text-generation", model=pretrained_model, torch_dtype=torch.bfloat16, device_map=device, num_beams=beam_size)
pipe = transformers.pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device=device, tokenizer=tokenizer, **pipeline_init_kwargs)

def translate(batch, device):
    _pipe = pipe

    if _pipe.device != device:
        _pipe.model = pipe.model.to(device)

    prompt = [_pipe.tokenizer.apply_chat_template([b], tokenize=False, add_generation_prompt=True) for b in batch]
    output = _pipe(prompt, max_new_tokens=512, **pipeline_call_kwargs)

    assert len(output) == len(batch), f"{output} vs {batch}"

    for idx, o in enumerate(output):
        assert len(o) == 1

        o = o[0]["generated_text"]

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

        batch = []
        sentences = []

if len(batch) > 0:
    translations, current_patience = util.translate_oom_aware(batch, translate, device, current_patience=current_patience)

    util.print_translation(translations, sentences)

    batch = []
    sentences = []
