
import os
import sys
import random

remove_duplicates = False # Change manually if you'd like different behaviour with the duplicates
shuffle_before_print = True # Change manually
print_positive_labels_once = True # Change manually
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

def read(fn, mt, remove_duplicates=False):
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

    expected_removed_sentences = sum([src_count[s] for s in src_to_remove])

    print(f"{mt}: duplicated sentences: {expected_removed_sentences} (uniq: {len(src_to_remove)})", file=sys.stderr)

    if remove_duplicates and len(src_to_remove) > 0:
        idxs_to_remove = []

        for idx, _src in enumerate(src):
            if _src in src_to_remove:
                idxs_to_remove.insert(0, idx) # reverse order in order to safely remove the elements from the lists

                src_count[_src] -= 1 # For later sanity check

        assert len(idxs_to_remove) == expected_removed_sentences

        for idx in idxs_to_remove:
            # remove
            del src[idx]
            del trg[idx]
            del labels[idx]

        # Sanity check
        for _src, count in src_count.items():
            expected_count = 0 if _src in src_to_remove else 2

            assert count == expected_count, f"{fn}: {count} != {expected_count}: {_src}"

        assert len(src) == len(trg) == len(labels)

    result = [(_src, _trg, label) for _src, _trg, label in zip(src, trg, labels)]

    return result

if not remove_duplicates:
    print("WARNING: since remove_duplicates=False, entries with the same source sentences will not be detected across MT systems. "
          f"Therefore, these cases will be merged under the same group according to labels_to_merge={labels_to_merge}", file=sys.stderr)

data = {mt: read(fn, mt, remove_duplicates=remove_duplicates) for mt, fn in input_fns.items()}

for idx_mt1 in range(len(mt_systems)):
    idx_mt2 = idx_mt1 + 1

    while idx_mt2 < len(mt_systems):
        mt1 = mt_systems[idx_mt1]
        mt2 = mt_systems[idx_mt2]

        # We expect to have the same positive samples
        assert set([f"{src}\t{trg}\t{label}" for src, trg, label in data[mt1] if label == 1]) == set([f"{src}\t{trg}\t{label}" for src, trg, label in data[mt2] if label == 1]), f"{mt1} {mt2}"

        idx_mt2 += 1


indices = {}
groups = {mt: None for mt in mt_systems}
all_src = set()
seen_idx = {mt: set() for mt in mt_systems}
aggregated_groups = 0

for mt in mt_systems:
    groups[mt] = [aggregated_groups + idx for idx in range(len(data[mt]))]
    aggregated_groups += len(groups[mt])

for mt, results in data.items():
    assert len(results) == len(groups[mt])

    for idx, (src, trg, label) in enumerate(results):
        if src not in indices:
            indices[src] = {}
        if mt not in indices[src]:
            indices[src][mt] = []

        assert idx not in seen_idx[mt]

        indices[src][mt].append(idx)
        all_src.add(src)
        seen_idx[mt].add(idx)

# Update groups using indices
for src in all_src:
    # Replace groups according to labels_to_merge

    assert len(indices[src]) == len(input_fns)

    for mt in mt_systems:
        assert len(indices[src][mt]) >= 2

    if labels_to_merge == "none":
        continue
    elif labels_to_merge == "both":
        mt = mt_systems[0]
        any_group = groups[mt][indices[src][mt][0]]

        for mt in mt_systems:
            for idx in indices[src][mt]:
                groups[mt][idx] = any_group
    else:
        assert labels_to_merge in ("pos", "neg"), f"Unexpected value for labels_to_merge: {labels_to_merge}"

        any_group = None
        idxs_to_update = {mt: [] for mt in mt_systems}
        idxs_to_ignore = {mt: [] for mt in mt_systems}

        for mt in mt_systems:
            for idx in indices[src][mt]:
                _src, _trg, label = data[mt][idx]

                assert _src == src, f"{_src} != {src}"

                if label == target_label:
                    if any_group is None:
                        any_group = groups[mt][idx]

                    idxs_to_update[mt].append(idx)
                else:
                    idxs_to_ignore[mt].append(idx)

        assert any_group is not None

        # Update
        for mt in mt_systems:
            assert len(idxs_to_update[mt]) == len(idxs_to_ignore[mt])

            for idx in idxs_to_ignore[mt]:
                _src, _trg, label = data[mt][idx]

                assert _src == src, f"{_src} != {src}"
                assert label != target_label

            for idx in idxs_to_update[mt]:
                _src, _trg, label = data[mt][idx]

                assert _src == src, f"{_src} != {src}"
                assert label == target_label

                groups[mt][idx] = any_group

# Print data with groups
first_mt = True
print_data = []

if print_positive_labels_once:
    print("INFO: positive labels are being printed just once instead of once per MT system", file=sys.stderr)

for mt, results in data.items():
    for idx, (src, trg, label) in enumerate(results):
        group = groups[mt][idx]
        data = f"{src}\t{trg}\t{label}\t{group}"

        if print_positive_labels_once and label == 1 and not first_mt:
            # It has been checked before that the positive labels data were the same across all MT systems, so it is ok to print just once
            continue

        print_data.append(data)

    first_mt = False

if shuffle_before_print:
    print("INFO: results are shuffled", file=sys.stderr)

    random.shuffle(print_data)

for data in print_data:
    print(data)

print("Done!", file=sys.stderr)
