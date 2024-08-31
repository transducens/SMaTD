
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
                raise Exception("torch.OutOfMemoryError even using batch_size=1") from e

            sys.stderr.write(f"torch.OutOfMemoryError error: current batch size is {len(_batch)}: using half batch_size\n")

            bs = len(_batch) // 2
            oom = True

        if len(output) >= len(batch):
            assert not oom
            assert len(output) == len(batch)

            break

    return output

def print_translation(translations, batch):
    assert len(translations) == len(batch)

    for sentence, translation in zip(batch, translations):
        sentence = sentence.replace('\t', ' ')
        translation = translation.replace('\t', ' ')

        print(f"{sentence}\t{translation}")

    sys.stdout.flush()

    return []
