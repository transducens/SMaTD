
import sys

import torch

def translate_oom_aware(batch, callback, device, reset_batch_size_after_oom=False, current_patience=0, patience=8):
    output = []
    bs = len(batch)
    oom = False
    _device = device

    if current_patience >= patience:
        current_patience = patience + 1

        _device = "cpu"

    while True:
        _batch = batch[len(output):len(output) + bs]

        try:
            _output = callback(_batch, _device)

            if oom and reset_batch_size_after_oom and bs != len(batch):
                sys.stderr.write(f"Batch size: {bs} -> {len(batch)}\n")

                bs = len(batch)

            oom = False

            output.extend(_output)
        except torch.OutOfMemoryError as e:
            oom = True

            if len(_batch) == 1:
                _device = "cpu"
                bs = len(batch)
                current_patience += 1

                sys.stderr.write("torch.OutOfMemoryError error: current batch size is 1: "
                                 f"using CPU device and using original batch size: {bs} (current_patience: {current_patience})\n")
            else:
                bs = len(_batch) // 2

                sys.stderr.write(f"torch.OutOfMemoryError error: current batch size is {len(_batch)}: using smaller batch size: {bs}\n")

        if not oom and _device == "cpu" and _device != device and current_patience < patience:
            sys.stderr.write(f"Using original device: {_device} -> {device}\n")

            _device = device
        elif current_patience == patience:
            assert _device == "cpu"

            sys.stderr.write(f"torch.OutOfMemoryError error: patience exhausted ({patience}): previous device ({device}) permanently set to CPU\n")
            torch.cuda.empty_cache()

            current_patience += 1 # avoid enter here again

        if len(output) >= len(batch):
            assert not oom
            assert len(output) == len(batch)

            break

    sys.stderr.write(f"{len(batch)} translations finished!\n")

    return output, current_patience

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

        diff_perc = 100. - len(translation) * 100. / len(sentence)

        if len(translation) == 0:
            sys.stderr.write(f"Empty translation for the following sentence: {sentence}\n")
        if abs(diff_perc) > 20.:
            sys.stderr.write(f"Translated sentence length is {'greater' if diff_perc < 0. else 'less'} than 20% compared to the source ({abs(diff_perc)}%): {sentence}\t{translation}\n")

    sys.stdout.flush()

    return []
