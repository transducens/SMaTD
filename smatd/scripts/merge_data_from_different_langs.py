
import os
import sys
import pickle
import random

import numpy as np

print(f"INFO: args provided: {sys.argv[1:]}", file=sys.stderr)

shuffle_before_print = True # Change manually
pickle_output = sys.argv[1].split(':')

for _pickle_output in pickle_output:
    assert not os.path.isfile(_pickle_output), f"{_pickle_output}: does exist"

input_fns = {}
input_fns_pickle = {}

for idx, lang_fn in enumerate(sys.argv[2:], 1):
    r = lang_fn.split(':')

    assert len(r) >= 2, r

    lang = r[0]
    fn = r[1]
    fn_pickle = []

    for _fn_pickle in r[2:]:
        assert _fn_pickle not in pickle_output
        assert os.path.isfile(_fn_pickle), f"{_fn_pickle}: does not exist (arg {idx}: {lang_fn})"

        fn_pickle.append(_fn_pickle)

    assert os.path.isfile(fn), f"{fn}: does not exist (arg {idx}: {lang_fn})"
    assert lang not in input_fns, "Same lang provided twice or more times"

    input_fns[lang] = fn
    input_fns_pickle[lang] = fn_pickle

assert len(input_fns) > 0, "You need to provide data files"
assert len(input_fns_pickle) == len(input_fns)

langs = list(input_fns.keys())
pickle_len = None

for lang in langs:
    if pickle_len is None:
        pickle_len = len(input_fns_pickle[lang])

    assert pickle_len == len(input_fns_pickle[lang]), f"{lang}: {pickle_len} vs {len(input_fns_pickle[lang])}"

load_pickle_data = pickle_len > 0

if load_pickle_data:
    assert len(pickle_output) == pickle_len, f"{len(pickle_output)} vs {pickle_len}"

print(f"INFO: loading pickle data: {load_pickle_data}", file=sys.stderr)

def read(fn, lang, pickle_fn=[]):
    assert ':' not in lang
    assert '#' not in lang

    src, trg, labels, groups = [], [], [], []
    src_count = {} # we assume that the human text is in the source side

    with open(fn, "rt") as fd:
        for l in fd:
            _src, _trg, label, group = l.rstrip("\r\n").split('\t') # yes, group is mandatory
            label = int(label)

            assert label in (0, 1), f"{fn}: {label}"

            src.append(_src)
            trg.append(_trg)
            labels.append(label)
            groups.append(group)

            if _src not in src_count:
                src_count[_src] = 0

            src_count[_src] += 1

    assert len(src) == len(trg) == len(labels) == len(groups)

    # Load pickle files
    pickle_data = []

    if len(pickle_fn) > 0:
        for _pickle_fn in pickle_fn:
            with open(_pickle_fn, "rb") as pickle_fd:
                _pickle_data = pickle.load(pickle_fd)

                for k in _pickle_data.keys():
                    assert len(src) == len(_pickle_data[k]), f"{_pickle_fn}: {k}: {len(src)} vs {len(_pickle_data[k])}"

                pickle_data.append(_pickle_data)

    result = [(_src, _trg, label, group) for _src, _trg, label, group in zip(src, trg, labels, groups)]

    return {
        "result": result,
        "pickle": pickle_data,
    }

data = {lang: read(input_fns[lang], lang, pickle_fn=input_fns_pickle[lang]) for lang in langs}
data_results = {lang: data[lang]["result"] for lang in langs}
data_results_src = {lang: [src for src, trg, label, group in data_results[lang]] for lang in langs}
data_pickle = {lang: data[lang]["pickle"] for lang in langs}
pickle_mat_keys = []

if load_pickle_data:
    pickle_mat_keys = list(data_pickle[langs[0]][0].keys())

for lang in langs:
    assert isinstance(data_pickle[lang], list), type(data_pickle[lang])

    if load_pickle_data:
        assert len(data_pickle[lang]) > 0
        assert len(data_pickle[lang]) == len(pickle_output)

        for idx in range(len(data_pickle[lang])):
            assert isinstance(data_pickle[lang][idx], dict), type(data_pickle[lang][idx])

            for k in data_pickle[lang][idx].keys():
                assert isinstance(data_pickle[lang][idx][k], list), type(data_pickle[lang][idx][k])
                assert isinstance(data_pickle[lang][idx][k][0], np.ndarray), type(data_pickle[lang][idx][k][0])
                assert len(data_pickle[lang][idx][k][0].shape) == 2, data_pickle[lang][idx][k][0].shape
    else:
        assert len(data_pickle[lang]) == 0

groups = {lang: None for lang in langs}
descs = {lang: None for lang in langs}
all_src = set()
seen_idx = {lang: set() for lang in langs}
aggregated_groups = 0

for lang in langs:
    groups[lang] = []
    descs[lang] = []

    for idx in range(len(data_results[lang])):
        src, trg, label, group = data_results[lang][idx]
        desc = f"mt:lang_is_{lang}" if label == 0 else f"human:lang_is_{lang}"

        if "mt:" in group or "human:" in group:
            desc = ''

        groups[lang].append(group)
        descs[lang].append(desc)

    aggregated_groups += len(groups[lang])

for lang, results in data_results.items():
    assert len(results) == len(groups[lang])
    assert len(descs[lang]) == len(groups[lang])

    for idx, (src, trg, label, group) in enumerate(results):
        assert groups[lang][idx] == group
        assert idx not in seen_idx[lang]

        all_src.add(src)
        seen_idx[lang].add(idx)

all_groups = set.union(*[set(groups[lang]) for lang in langs])
group2label = {group: set() for group in all_groups}

for lang in langs:
    for idx, (src, trg, label, group) in enumerate(data_results[lang]):
        group = groups[lang][idx]

        if group not in group2label:
            group2label[group] = set()

        group2label[group].add(label)

assert len(all_groups) == len(group2label.keys())

for group in all_groups:
    assert len(group2label[group]) == 1, group

# Print data with groups
print_data = []
data_pickle_update = {lang: [{k: [] for k in data_pickle[lang][idx].keys()} for idx in range(len(data_pickle[lang]))] for lang in langs}
data_pickle_update_skip = {lang: 0 for lang in langs}
data_pickle_update_merge = [{k: [] for k in pickle_mat_keys} for _ in range(len(data_pickle[langs[0]]))]

for lang, results in data_results.items():
    for idx, (src, trg, label, group) in enumerate(results):
        assert group == groups[lang][idx]

        desc = descs[lang][idx]

        if desc != '':
            group += f":{desc}"

        group = f"{lang}_{group}" # to avoid future problems with local group collisions

        d = f"{src}\t{trg}\t{label}\t{group}#{lang}"

        print_data.append(d)

        if load_pickle_data:
            for vidx, v in enumerate(data_pickle[lang]):
                for vdk in data_pickle[lang][vidx].keys():
                    assert isinstance(v[vdk], list), type(v[vdk])
                    assert isinstance(v[vdk][idx], np.ndarray), type(v[vdk])
                    assert len(v[vdk][idx].shape) == 2

                    data_pickle_update[lang][vidx][vdk].append(v[vdk][idx])

            assert len(data_pickle_update_merge) == len(pickle_output) == len(data_pickle[lang])

            for output_idx in range(len(data_pickle_update_merge)):
                for k in pickle_mat_keys:
                    data_pickle_update_merge[output_idx][k].append(data_pickle[lang][output_idx][k][idx])

if load_pickle_data:
    for output_idx in range(len(data_pickle_update_merge)):
        k = pickle_mat_keys[0]
        expected_elements_to_be_printed = sum([len(data_pickle[lang][output_idx][k]) for lang in langs])

        assert expected_elements_to_be_printed == len(data_pickle_update_merge[output_idx][k])

for lang in langs:
    # Sanity checks

    assert isinstance(data_pickle_update[lang], list), type(data_pickle_update[lang])

    if load_pickle_data:
        assert len(data_pickle_update[lang]) > 0
        assert len(data_pickle_update[lang]) == len(pickle_output)
        assert isinstance(data_pickle_update[lang], list), type(data_pickle_update[lang])

        for idx in range(len(data_pickle_update[lang])):
            assert isinstance(data_pickle_update[lang][idx], dict), type(data_pickle_update[lang][idx])

            for k in data_pickle_update[lang][idx].keys():
                assert isinstance(data_pickle_update[lang][idx][k], list), type(data_pickle_update[lang][idx][k])

                for vidx in range(len(data_pickle_update[lang][idx][k])):
                    assert isinstance(data_pickle_update[lang][idx][k][vidx], np.ndarray), type(data_pickle_update[lang][idx][k][vidx])
                    assert len(data_pickle_update[lang][idx][k][vidx].shape) == 2, data_pickle_update[lang][idx][k][vidx].shape

        for vidx, v in enumerate(data_pickle[lang]):
            for vdk in data_pickle[lang][vidx].keys():
                assert len(v[vdk]) == (len(data_pickle_update[lang][vidx][vdk]) + data_pickle_update_skip[lang]), f"{lang}: {vidx}: {len(v[vdk])} vs {len(data_pickle_update[lang][vidx][vdk])} + {data_pickle_update_skip[lang]}"

                vidx2 = vidx + 1

                while vidx2 < len(data_pickle[lang]):
                    assert len(v[vdk]) == (len(data_pickle_update[lang][vidx2][vdk]) + data_pickle_update_skip[lang]), f"{lang}: {len(v[vdk])} vs {len(data_pickle_update[lang][vidx2][vdk])} + {data_pickle_update_skip[lang]}"

                    vidx2 += 1

        assert len(data_pickle[lang]) == len(data_pickle_update[lang])

if load_pickle_data:
    expected_elements_to_be_printed = 0

    for lang in langs:
        for idx in range(len(data_pickle_update[lang])):
            vdk = list(data_pickle_update[lang][idx].keys())[0]
            expected_elements_to_be_printed += len(data_pickle_update[lang][idx][vdk])

            for _vdk in data_pickle_update[lang][idx].keys():
                assert len(data_pickle_update[lang][idx][_vdk]) == len(data_pickle_update[lang][idx][vdk])

    assert len(print_data) * len(pickle_output) == expected_elements_to_be_printed, f"{lang}: {len(print_data)} * {len(pickle_output)} vs {expected_elements_to_be_printed}"

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
