
import sys

import torch

def translate_oom_aware(batch, callback, reset_batch_size_after_oom=False):
    output = []
    bs = len(batch)
    oom = False

    while True:
        _batch = batch[len(output):len(output) + bs]

        try:
            _output = callback(_batch)

            if oom and reset_batch_size_after_oom:
                sys.stderr.write(f"Batch size: {bs} -> {len(batch)}\n")

                bs = len(batch)

            oom = False

            output.extend(_output)
        except torch.OutOfMemoryError as e:
            if len(_batch) == 1:
                raise Exception("torch.OutOfMemoryError even using batch_size=1: you should using CPU device") from e

            sys.stderr.write(f"torch.OutOfMemoryError error: current batch size is {len(_batch)}: using half batch_size\n")

            bs = len(_batch) // 2
            oom = True

        if len(output) >= len(batch):
            assert not oom
            assert len(output) == len(batch)

            break

    return output

def load_model(pretrained_model, transformers_class, device=None, from_pretrained_kwargs={}):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers_class.from_pretrained(pretrained_model, **from_pretrained_kwargs)

    try:
        model = model.to(device)
    except torch.OutOfMemoryError as e1:
        if device != "cpu":
            sys.stderr.write(f"torch.OutOfMemoryError error while loading model: device: {device} -> cpu\n")

            device = "cpu"
        else:
            raise e1

        try:
            model = model.to(device)
        except Exception as e2:
            raise Exception("could not load model using CPU device") from e2

    torch.cuda.empty_cache()

    return model, device

def print_translation(translations, batch):
    assert len(translations) == len(batch)

    for sentence, translation in zip(batch, translations):
        sentence = sentence.replace('\t', ' ')
        translation = translation.replace('\t', ' ')

        print(f"{sentence}\t{translation}")

    sys.stdout.flush()

    return []
