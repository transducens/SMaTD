
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
select_random_value_if_same_group = bool(int(sys.argv[3])) # according to labels_to_merge

assert labels_to_merge in ("pos", "neg", "both", "none"), labels_to_merge

for _pickle_output in pickle_output:
    assert not os.path.isfile(_pickle_output), f"{_pickle_output}: does exist"

target_label = 1 if labels_to_merge == "pos" else 0
input_fns = {}
input_fns_pickle = {}

for idx, mt_fn in enumerate(sys.argv[4:], 1):
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
    assert ':' not in mt

    src, trg, labels = [], [], []
    labels_count = {0: 0, 1: 0}
    src_count = {} # we assume that the human text is in the source side

    if remove_duplicates:
        assert len(pickle_fn) == 0, "Pickle processing and remove_duplicates=True is not supported"

    with open(fn, "rt") as fd:
        for l in fd:
            _src, _trg, label = l.rstrip("\r\n").split('\t')
            label = int(label)
            _src = f"no_duplicated:{_src}" # necessary for processing duplicates

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

    result = [(_src, _trg, label) for idx, (_src, _trg, label) in enumerate(zip(src, trg, labels))]

    return {
        "result": result,
        "pickle": pickle_data,
        "src_to_remove": src_to_remove,
    }

data = {mt: read(input_fns[mt], mt, pickle_fn=input_fns_pickle[mt], remove_duplicates=remove_duplicates) for mt in mt_systems}
data_results = {mt: data[mt]["result"] for mt in mt_systems}
data_results_src = {mt: [src for src, trg, label in data_results[mt]] for mt in mt_systems}
data_pickle = {mt: data[mt]["pickle"] for mt in mt_systems}
src_to_remove = {mt: data[mt]["src_to_remove"] for mt in mt_systems}
pickle_mat_keys = []

assert len(set([len(src_to_remove[mt]) for mt in mt_systems])) == 1, "Since we expect the same source sentences for all MT systems, duplicates should be the same"
assert len(set.intersection(*[src_to_remove[mt] for mt in mt_systems])) == len(src_to_remove[mt_systems[0]]), "Same duplicated sentences were expected"

if not remove_duplicates:
    uniq_src_to_remove = set.union(*[src_to_remove[mt] for mt in mt_systems])
    total_remove = len(uniq_src_to_remove)

    if total_remove > 0:
        print(f"INFO: {total_remove} src unique sentences found", file=sys.stderr)

        sorted_duplicated_sentences = sorted(list(uniq_src_to_remove))

        for duplicated_sentence in sorted_duplicated_sentences:
            assert duplicated_sentence.split(':')[0] == "no_duplicated"

            all_found_values = []

            for mt in mt_systems:
                found_idx = 0
                idx_and_trg_and_label_pairs = []

                while True:
                    try:
                        new_idx = data_results_src[mt].index(duplicated_sentence, found_idx)
                    except ValueError:
                        break

                    found_idx = new_idx + 1
                    src, trg, label = data_results[mt][new_idx]

                    assert data_results_src[mt][new_idx] == duplicated_sentence == src

                    idx_and_trg_and_label_pairs.append((new_idx, trg, label))

                all_found_values.append(len(idx_and_trg_and_label_pairs))

                idx_and_trg_and_label_pairs = sorted(idx_and_trg_and_label_pairs, key=lambda e: (e[2], e[1])) # We do not care about which elements are matched with each other, but we do care that the label values are the same and sort by trg
                sum_labels = [sum([1 if l == 0 else 0 for _, _, l in idx_and_trg_and_label_pairs]), sum([1 if l == 1 else 0 for _, _, l in idx_and_trg_and_label_pairs])]

                assert sum_labels[0] == sum_labels[1]
                assert len(idx_and_trg_and_label_pairs) % 2 == 0

                # We need to apply changes to BOTH positive and negative label samples
                for idx1 in range(len(idx_and_trg_and_label_pairs) // 2):
                    idx2 = idx1 + len(idx_and_trg_and_label_pairs) // 2
                    found_idx1 = idx_and_trg_and_label_pairs[idx1][0]
                    found_idx2 = idx_and_trg_and_label_pairs[idx2][0]
                    src1, trg1, label1 = data_results[mt][found_idx1]
                    src2, trg2, label2 = data_results[mt][found_idx2]

                    assert duplicated_sentence == src1 == src2 == data_results_src[mt][found_idx1] == data_results_src[mt][found_idx2]

                    data_results[mt][found_idx1] = (f"duplicated_idx_{idx1}:{':'.join(duplicated_sentence.split(':')[1:])}", *data_results[mt][found_idx1][1:])
                    data_results[mt][found_idx2] = (f"duplicated_idx_{idx1}:{':'.join(duplicated_sentence.split(':')[1:])}", *data_results[mt][found_idx2][1:])
                    data_results_src[mt][found_idx1] = data_results[mt][found_idx1][0]
                    data_results_src[mt][found_idx2] = data_results[mt][found_idx2][0]

                    _src1, _trg1, _label1 = data_results[mt][found_idx1]
                    _src2, _trg2, _label2 = data_results[mt][found_idx2]

                    assert src1.split(':')[0] != _src1.split(':')[0]
                    assert src2.split(':')[0] != _src2.split(':')[0]
                    assert ':'.join(src1.split(':')[1:]) == ':'.join(_src1.split(':')[1:]) == ':'.join(src2.split(':')[1:]) == ':'.join(_src2.split(':')[1:])
                    assert trg1 == _trg1
                    assert trg2 == _trg2
                    assert label1 == _label1
                    assert label2 == _label2
                    assert label1 != label2

            assert len(set(all_found_values)) == 1, all_found_values

if load_pickle_data:
    pickle_mat_keys = list(data_pickle[mt_systems[0]][0].keys())

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
        set1 = set([f"{src}\t{trg}\t{label}" for src, trg, label in data_results[mt1] if label == 1])
        set2 = set([f"{src}\t{trg}\t{label}" for src, trg, label in data_results[mt2] if label == 1])

        # We expect to have the same positive samples
        assert len(set1) == len(set2), f"{mt1} {mt2}"
        assert set1 == set2, f"{mt1} {mt2}: {set.difference(set1, set2)} - {set.difference(set2, set1)}"

        idx_mt2 += 1

indices = {}
groups = {mt: None for mt in mt_systems}
descs = {mt: None for mt in mt_systems}
all_src = set()
seen_idx = {mt: set() for mt in mt_systems}
aggregated_groups = 0

for mt in mt_systems:
    groups[mt] = []
    descs[mt] = []

    for idx in range(len(data_results[mt])):
        src, trg, label = data_results[mt][idx]
        desc = f"mt:system_is_{mt}" if label == 0 else f"human:from_{mt}_file"

        groups[mt].append(f"{aggregated_groups + idx}")
        descs[mt].append(desc)

    aggregated_groups += len(groups[mt])

for mt, results in data_results.items():
    assert len(results) == len(groups[mt])
    assert len(descs[mt]) == len(groups[mt])

    for idx, (src, trg, label) in enumerate(results):
        if src not in indices:
            indices[src] = {}
        if mt not in indices[src]:
            indices[src][mt] = []

        assert idx not in seen_idx[mt]

        indices[src][mt].append(idx)
        all_src.add(src)
        seen_idx[mt].add(idx)

all_groups = set.union(*[set(groups[mt]) for mt in mt_systems])
group2label = {group: set() for group in all_groups}

for mt in mt_systems:
    for idx, (src, trg, label) in enumerate(data_results[mt]):
        group = groups[mt][idx]

        if group not in group2label:
            group2label[group] = set()

        group2label[group].add(label)

assert len(all_groups) == len(group2label.keys())

for group in all_groups:
    assert len(group2label[group]) == 1, group

# Update groups using indices
for src in all_src:
    # Replace groups according to labels_to_merge

    assert len(indices[src]) == len(input_fns)

    for mt in mt_systems:
        assert len(indices[src][mt]) >= 2, f"{mt}: {indices[src][mt]}: {src}"

    if labels_to_merge == "none":
        continue
    elif labels_to_merge == "both":
        mt = mt_systems[0]
        any_group = groups[mt][indices[src][mt][0]]

        for mt in mt_systems:
            for idx in indices[src][mt]:
                _src, _trg, label = data_results[mt][idx]

                assert _src == src, f"{_src} != {src}"

                groups[mt][idx] = None if select_random_value_if_same_group else any_group

        random_mt = None

        for mt in random.sample(mt_systems, len(mt_systems)):
            if len(indices[src][mt]) > 0: # there may not be samples in any MT system
                random_mt = mt

                break

        if random_mt is not None:
            random_idx = random.choice(indices[src][random_mt])
            groups[random_mt][random_idx] = any_group # random value if select_random_value_if_same_group, otherwise harmless
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

                groups[mt][idx] = None if select_random_value_if_same_group else any_group

        random_mt = None

        for mt in random.sample(mt_systems, len(mt_systems)):
            if len(idxs_to_update[mt]) > 0: # there may not be samples in any MT system
                random_mt = mt

                break

        if random_mt is not None:
            random_idx = random.choice(idxs_to_update[random_mt])
            groups[random_mt][random_idx] = any_group # random value if select_random_value_if_same_group, otherwise harmless

# Print data with groups
first_mt = True
print_data = []
data_pickle_update = {mt: [{k: [] for k in data_pickle[mt][idx].keys()} for idx in range(len(data_pickle[mt]))] for mt in mt_systems}
data_pickle_update_skip = {mt: 0 for mt in mt_systems}
data_pickle_update_merge = [{k: [] for k in pickle_mat_keys} for _ in range(len(data_pickle[mt_systems[0]]))]

if print_positive_labels_once:
    print("INFO: positive labels are being printed just once instead of once per MT system", file=sys.stderr)

for mt, results in data_results.items():
    for idx, (src, trg, label) in enumerate(results):
        group = groups[mt][idx]
        desc = descs[mt][idx]

        if group is None:
            data_pickle_update_skip[mt] += 1 if load_pickle_data else 0

            continue

        src_id = src.split(':')[0]

        assert src_id == "no_duplicated" or src_id.startswith("duplicated_idx_"), src.split(':')

        src = ':'.join(src.split(':')[1:])
        d = f"{src}\t{trg}\t{label}\t{group}:{desc}"

        if print_positive_labels_once and label == 1 and not first_mt:
            # It has been checked before that the positive labels data were the same across all MT systems, so it is ok to print just once
            data_pickle_update_skip[mt] += 1 if load_pickle_data else 0

            continue

        print_data.append(d)

        if load_pickle_data:
            for vidx, v in enumerate(data_pickle[mt]):
                for vdk in data_pickle[mt][vidx].keys():
                    assert isinstance(v[vdk], list), type(v[vdk])
                    assert isinstance(v[vdk][idx], np.ndarray), type(v[vdk])
                    assert len(v[vdk][idx].shape) == 2

                    data_pickle_update[mt][vidx][vdk].append(v[vdk][idx])

            assert len(data_pickle_update_merge) == len(pickle_output) == len(data_pickle[mt])

            for output_idx in range(len(data_pickle_update_merge)):
                for k in pickle_mat_keys:
                    data_pickle_update_merge[output_idx][k].append(data_pickle[mt][output_idx][k][idx])

    first_mt = False

if load_pickle_data:
    for output_idx in range(len(data_pickle_update_merge)):
        k = pickle_mat_keys[0]
        expected_elements_to_be_printed = sum([len(data_pickle[mt][output_idx][k]) for mt in mt_systems])

        assert expected_elements_to_be_printed == len(data_pickle_update_merge[output_idx][k]) + sum([data_pickle_update_skip[mt] for mt in mt_systems])

for mt in mt_systems:
    # Sanity checks

    assert isinstance(data_pickle_update[mt], list), type(data_pickle_update[mt])

    if load_pickle_data:
        assert len(data_pickle_update[mt]) > 0
        assert len(data_pickle_update[mt]) == len(pickle_output)
        assert isinstance(data_pickle_update[mt], list), type(data_pickle_update[mt])

        for idx in range(len(data_pickle_update[mt])):
            assert isinstance(data_pickle_update[mt][idx], dict), type(data_pickle_update[mt][idx])

            for k in data_pickle_update[mt][idx].keys():
                assert isinstance(data_pickle_update[mt][idx][k], list), type(data_pickle_update[mt][idx][k])

                for vidx in range(len(data_pickle_update[mt][idx][k])):
                    assert isinstance(data_pickle_update[mt][idx][k][vidx], np.ndarray), type(data_pickle_update[mt][idx][k][vidx])
                    assert len(data_pickle_update[mt][idx][k][vidx].shape) == 2, data_pickle_update[mt][idx][k][vidx].shape

        for vidx, v in enumerate(data_pickle[mt]):
            for vdk in data_pickle[mt][vidx].keys():
                assert len(v[vdk]) == (len(data_pickle_update[mt][vidx][vdk]) + data_pickle_update_skip[mt]), f"{mt}: {vidx}: {len(v[vdk])} vs {len(data_pickle_update[mt][vidx][vdk])} + {data_pickle_update_skip[mt]}"

                vidx2 = vidx + 1

                while vidx2 < len(data_pickle[mt]):
                    assert len(v[vdk]) == (len(data_pickle_update[mt][vidx2][vdk]) + data_pickle_update_skip[mt]), f"{mt}: {len(v[vdk])} vs {len(data_pickle_update[mt][vidx2][vdk])} + {data_pickle_update_skip[mt]}"

                    vidx2 += 1

        assert len(data_pickle[mt]) == len(data_pickle_update[mt])

if load_pickle_data:
    expected_elements_to_be_printed = 0

    for mt in mt_systems:
        for idx in range(len(data_pickle_update[mt])):
            vdk = list(data_pickle_update[mt][idx].keys())[0]
            expected_elements_to_be_printed += len(data_pickle_update[mt][idx][vdk])

            for _vdk in data_pickle_update[mt][idx].keys():
                assert len(data_pickle_update[mt][idx][_vdk]) == len(data_pickle_update[mt][idx][vdk])

    assert len(print_data) * len(pickle_output) == expected_elements_to_be_printed, f"{mt}: {len(print_data)} * {len(pickle_output)} vs {expected_elements_to_be_printed}"

if shuffle_before_print:
    print("INFO: results are shuffled", file=sys.stderr)

    idxs = list(range(len(print_data)))

    random.shuffle(idxs)

    print_data = [print_data[idx] for idx in idxs]

    if load_pickle_data:
        for output_idx in range(len(data_pickle_update_merge)):
            for k in pickle_mat_keys:
                assert len(data_pickle_update_merge[output_idx][k]) == len(print_data)

        data_pickle_update_merge = [{k: [data_pickle_update_merge[output_idx][k][idx] for idx in idxs] for k in pickle_mat_keys} for output_idx in range(len(data_pickle_update_merge))]

if load_pickle_data:
    for output_idx, _pickle_output in enumerate(pickle_output):
        _data_pickle_update_merge = data_pickle_update_merge[output_idx]

        with open(_pickle_output, "wb") as pickle_fd:
            pickle.dump(_data_pickle_update_merge, pickle_fd)

        print(f"INFO: pickle data dumped: {_pickle_output}", file=sys.stderr)

for d in print_data:
    print(d)

print("Done!", file=sys.stderr)
