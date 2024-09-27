
import os
import sys

labels_to_merge = sys.argv[1]

assert labels_to_merge in ("pos", "neg", "both", "none"), labels_to_merge

target_label = 1 if labels_to_merge == "pos" else 0
input_fns = {}

for idx, mt_fn in enumerate(sys.argv[2:], 1):
    r = mt_fn.split(':')

    assert len(r) in (1, 2), r

    if len(r) == 1:
        mt = f"mt{idx}"
        fn = r[0]
    else:
        mt = r[0]
        fn = r[1]

    assert os.path.isfile(fn), f"{fn}: does not exist (arg {idx}: {mt_fn})"

    input_fns[mt] = fn

print("NOTE: we assume that the source side is human text, and that the target is machine or human translated text. We use human text to discriminate unique sentences", file=sys.stderr)

assert len(input_fns) > 0, "You need to provide data files"

mt_systems = list(input_fns.keys())

def read(fn, mt):
    src, trg, labels = [], [], []
    labels_count = {0: 0, 1: 0}
    src_count = {} # we assume that the human text is in the source side

    with open(fn, "rt") as fd:
        for l in fd:
            _src, _trg, label = l.rstrip("\r\n").split('\t')
            label = int(label)

            assert label in (0, 1), f"{fn}: {label}"

            src.append(_src)
            trg.append(_trg)
            labels.append(label)

            if _src not in src_count:
                src_count[_src] = 0

            src_count[_src] += 1
            labels_count[label] += 1

    assert len(src) == len(trg) == len(labels)
    assert labels_count[0] == labels_count[1], f"{fn}: same number of positive and negative samples is expected"

    src_to_remove = set()

    for _src, count in src_count.items():
        assert count % 2 == 0, f"{fn}: all source sentences count should be even either because we have the human and machine translated text or becuase there are duplicates"
        assert count >= 2, f"{fn}: {count}: {_src}"

        if count > 2:
            src_to_remove.add(_src)

    print(f"{mt}: sentences to remove: {len(src_to_remove)}", file=sys.stderr)

    if len(src_to_remove) > 0:
        for _idx in range(len(src)):
            idx = (_idx + 1) * -1 # reverse because removing can't be done in ascending order
            _src = src[idx]

            if _src in src_to_remove:
                # remove
                del src[idx]
                del trg[idx]
                del labels[idx]

                src_count[_src] -= 1 # For later sanity check

        # Sanity check
        for _src, count in src_count.items():
            assert count == 2, f"{fn}: {count} != 2 and should be equal to 2: {_src}"

    assert len(src) == len(trg) == len(labels)

    result = [(_src, _trg, label) for _src, _trg, label in zip(src, trg, labels)]

    return result

    #return src, trg, labels

data = {mt: read(fn, mt) for mt, fn in input_fns.items()}
indices = {}
groups = {mt: [idx for idx in range(len(data[mt]))] for mt in mt_systems}
all_src = set()

for mt, results in data.items():
    assert len(results) == len(groups[mt])

    for idx, (src, trg, label) in enumerate(results):
        if src not in indices:
            indices[src] = {}
        if mt not in indices[src]:
            indices[src][mt] = []

        indices[src][mt].append(idx)
        all_src.add(src)

# Update groups using indices
for src in all_src:
    # Replace groups according to labels_to_merge

    assert len(indices[src]) == len(input_fns)

    for mt in mt_systems:
        assert len(indices[src][mt]) == 2

    if labels_to_merge == "none":
        continue
    elif labels_to_merge == "both":
        mt = mt_systems[0]
        any_idx = int(indices[src][mt][0])

        for mt in mt_systems:
            for idx in indices[src][mt]:
                groups[mt][idx] = any_idx
    else:
        # TODO for some reason, this configuration is not working as expected...

        assert labels_to_merge in ("pos", "neg"), f"Unexpected value for labels_to_merge: {labels_to_merge}"

        any_idx = None
        idxs_to_update = {mt: [] for mt in mt_systems}
        idxs_to_ignore = {mt: [] for mt in mt_systems}

        for mt in mt_systems:
            for idx in indices[src][mt]:
                _src, _trg, label = data[mt][idx]

                if label == target_label:
                    if any_idx is None:
                        any_idx = int(idx)

                    idxs_to_update[mt].append(idx)
                else:
                    idxs_to_ignore[mt].append(idx)

        assert any_idx is not None

        # Sanity check
        for mt1 in mt_systems:
            for mt2 in mt_systems:
                assert len(idxs_to_update[mt1]) == 1
                assert len(idxs_to_ignore[mt2]) == 1

                if mt1 != mt2:
                    assert idxs_to_update[mt1][0] != idxs_to_update[mt2][0]

        # Update
        for mt in mt_systems:
            for idx in idxs_to_ignore[mt]:
                _src, _trg, label = data[mt][idx]

                assert label != target_label

            for idx in idxs_to_update[mt]:
                _src, _trg, label = data[mt][idx]

                assert label == target_label

                groups[mt][idx] = int(any_idx)

# Print data with groups
for mt, results in data.items():
    for idx, (src, trg, label) in enumerate(results):
        group = groups[mt][idx]

        print(f"{src}\t{trg}\t{label}\t{group}")

print("Done!", file=sys.stderr)
