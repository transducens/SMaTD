
import os
import sys
import random
import pickle

import numpy as np

print(f"INFO: args provided: {sys.argv[1:]}", file=sys.stderr)

remove_duplicates = False # Change manually if you'd like different behaviour with the duplicates
shuffle_before_print = True # Change manually
print_positive_labels_once = True # Change manually
labels_to_merge = sys.argv[1]
pickle_output = sys.argv[2].split(':')

assert labels_to_merge in ("pos", "neg", "both", "none"), labels_to_merge

for _pickle_output in pickle_output:
    assert not os.path.isfile(_pickle_output), f"{_pickle_output}: does exist"

target_label = 1 if labels_to_merge == "pos" else 0
input_fns = {}
input_fns_pickle = {}

for idx, mt_fn in enumerate(sys.argv[3:], 1):
    r = mt_fn.split(':')

    assert len(r) >= 2, r

    mt = r[0]
    fn = r[1]
    fn_pickle = []

    for _fn_pickle in r[2:]:
        assert _fn_pickle not in pickle_output
        assert os.path.isfile(_fn_pickle), f"{_fn_pickle}: does not exist (arg {idx}: {mt_fn})"

        fn_pickle.append(_fn_pickle)

    assert os.path.isfile(fn), f"{fn}: does not exist (arg {idx}: {mt_fn})"
    assert mt not in input_fns, "Same MT provided twice or more times"

    input_fns[mt] = fn
    input_fns_pickle[mt] = fn_pickle

assert len(input_fns) > 0, "You need to provide data files"
assert len(input_fns_pickle) == len(input_fns)

mt_systems = list(input_fns.keys())
pickle_len = None

for mt in mt_systems:
    if pickle_len is None:
        pickle_len = len(input_fns_pickle[mt])

    assert pickle_len == len(input_fns_pickle[mt]), f"{mt}: {pickle_len} vs {len(input_fns_pickle[mt])}"

load_pickle_data = pickle_len > 0

if load_pickle_data:
    assert len(pickle_output) == pickle_len, f"{len(pickle_output)} vs {pickle_len}"

print(f"INFO: loading pickle data: {load_pickle_data}", file=sys.stderr)
print("NOTE: we assume that the source side is human text, and that the target is machine or human translated text. We use human text to discriminate unique sentences", file=sys.stderr)

def read(fn, mt, pickle_fn=[], remove_duplicates=False):
    src, trg, labels = [], [], []
    labels_count = {0: 0, 1: 0}
    src_count = {} # we assume that the human text is in the source side

    if remove_duplicates:
        assert len(pickle_fn) == 0, "Pickle processing and remove_duplicates=True is not supported"

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

    # Load pickle files
    pickle_data = []

    if len(pickle_fn) > 0:
        for _pickle_fn in pickle_fn:
            with open(_pickle_fn, "rb") as pickle_fd:
                _pickle_data = pickle.load(pickle_fd)

                for k in _pickle_data.keys():
                    assert len(src) == len(_pickle_data[k]), f"{_pickle_fn}: {k}: {len(src)} vs {len(_pickle_data[k])}"

                pickle_data.append(_pickle_data)

    # Remove if necessary

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

    return {
        "result": result,
        "pickle": pickle_data,
    }

if not remove_duplicates:
    print("WARNING: since remove_duplicates=False, entries with the same source sentences will not be detected across MT systems. "
          f"Therefore, these cases will be merged under the same group according to labels_to_merge={labels_to_merge}", file=sys.stderr)

data = {mt: read(input_fns[mt], mt, pickle_fn=input_fns_pickle[mt], remove_duplicates=remove_duplicates) for mt in mt_systems}
data_results = {mt: data[mt]["result"] for mt in mt_systems}
data_pickle = {mt: data[mt]["pickle"] for mt in mt_systems}

for mt in mt_systems:
    assert isinstance(data_pickle[mt], list), type(data_pickle[mt])

    if load_pickle_data:
        assert len(data_pickle[mt]) > 0
        assert len(data_pickle[mt]) == len(pickle_output)

        for idx in range(len(data_pickle[mt])):
            assert isinstance(data_pickle[mt][idx], dict), type(data_pickle[mt][idx])

            for k in data_pickle[mt][idx].keys():
                assert isinstance(data_pickle[mt][idx][k], list), type(data_pickle[mt][idx][k])
                assert isinstance(data_pickle[mt][idx][k][0], np.ndarray), type(data_pickle[mt][idx][k][0])
                assert len(data_pickle[mt][idx][k][0].shape) == 2, data_pickle[mt][idx][k][0].shape
    else:
        assert len(data_pickle[mt]) == 0

for idx_mt1 in range(len(mt_systems)):
    idx_mt2 = idx_mt1 + 1

    while idx_mt2 < len(mt_systems):
        mt1 = mt_systems[idx_mt1]
        mt2 = mt_systems[idx_mt2]

        # We expect to have the same positive samples
        assert set([f"{src}\t{trg}\t{label}" for src, trg, label in data_results[mt1] if label == 1]) == set([f"{src}\t{trg}\t{label}" for src, trg, label in data_results[mt2] if label == 1]), f"{mt1} {mt2}"

        idx_mt2 += 1

indices = {}
groups = {mt: None for mt in mt_systems}
all_src = set()
seen_idx = {mt: set() for mt in mt_systems}
aggregated_groups = 0

for mt in mt_systems:
    groups[mt] = [aggregated_groups + idx for idx in range(len(data_results[mt]))]
    aggregated_groups += len(groups[mt])

for mt, results in data_results.items():
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
                _src, _trg, label = data_results[mt][idx]

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
                _src, _trg, label = data_results[mt][idx]

                assert _src == src, f"{_src} != {src}"
                assert label != target_label

            for idx in idxs_to_update[mt]:
                _src, _trg, label = data_results[mt][idx]

                assert _src == src, f"{_src} != {src}"
                assert label == target_label

                groups[mt][idx] = any_group

# Print data with groups
first_mt = True
print_data = []
data_pickle_update = {k: [[] for _ in range(len(data_pickle[k]))] for k in data_pickle.keys()}

if print_positive_labels_once:
    print("INFO: positive labels are being printed just once instead of once per MT system", file=sys.stderr)

for mt, results in data_results.items():
    for idx, (src, trg, label) in enumerate(results):
        group = groups[mt][idx]
        d = f"{src}\t{trg}\t{label}\t{group}"

        if print_positive_labels_once and label == 1 and not first_mt:
            # It has been checked before that the positive labels data were the same across all MT systems, so it is ok to print just once
            continue

        print_data.append(d)

        if load_pickle_data:
            for v in data_pickle[mt]:
                for vidx in range(len(v)):
                    # TODO fix
                    data_pickle_update[mt][vidx].append(v[vidx][idx])

                assert len(v) == len(data_pickle_update[k])

                for vidx in range(len(v)):
                    assert len(v[vidx]) == len(data_pickle_update[k][vidx])

                    vidx2 = vidx + 1

                    while vidx2 < len(v):
                        assert len(v[vidx]) == len(data_pickle_update[k][vidx2])

                        vidx2 += 1

    first_mt = False

if load_pickle_data:
    for k in data_pickle_update.keys():
        for idx in range(len(data_pickle_update[k])):
            assert len(print_data) == len(data_pickle_update[k][idx]), f"{k}: {idx}: {len(print_data)} vs {len(data_pickle_update[k][idx])}"

if shuffle_before_print:
    print("INFO: results are shuffled", file=sys.stderr)

    idxs = list(range(len(print_data)))

    random.shuffle(idxs)

    print_data = [print_data[idx] for idx in idxs]

    if load_pickle_data:
        #data_pickle_update = {k: [data_pickle_update[k][idx] for idx in idxs] for k in data_pickle_update.keys()}
        data_pickle_update = {k: [[data_pickle_update[k][vidx][idx] for idx in idxs] for vidx in range(len(v))] for k, v in data_pickle_update.items()}

if load_pickle_data:
    for k in data_pickle_update.keys():
        assert len(pickle_output) == len(data_pickle_update[k])
        assert isinstance(data_pickle_update[k], list), type(data_pickle_update[k])

        for idx in range(len(data_pickle_update[k])):
            assert isinstance(data_pickle_update[k][idx], list), type(data_pickle_update[k][idx])
            assert isinstance(data_pickle_update[k][idx][0], np.ndarray), type(data_pickle_update[k][idx][0])
            assert isinstance(data_pickle_update[k][idx][0][0], np.float64), type(data_pickle_update[k][idx][0][0])

    for idx, _pickle_output in enumerate(pickle_output):
#        _data_pickle_update = {k: v[k][idx] for k, v in data_pickle_update.items()}
#
#        with open(_pickle_output, "wb") as pickle_fd:
#            pickle.dump(_data_pickle_update, pickle_fd)

        print(f"INFO: pickle data dumped: {_pickle_output}", file=sys.stderr)

for d in print_data:
    print(d)

print("Done!", file=sys.stderr)
