
import sys

import numpy as np
from sklearn.metrics import accuracy_score

# files
baseline_fn = sys.argv[1]
system_fn = sys.argv[2]
reference_fn = sys.argv[3]

# Approximate randomization test
n_samples = 10000

def metric(refs, preds):
    assert isinstance(refs, np.ndarray), type(refs)
    assert isinstance(preds, np.ndarray), type(preds)
    assert refs.shape == preds.shape, f"Shape mismatch: {refs.shape} vs {preds.shape}"
    assert len(refs.shape) == 1
    #assert len(refs.shape) == 2
    #assert refs.shape[0] == 1

    #return accuracy_score(refs[0], preds[0])
    return accuracy_score(refs, preds)

def read(fn):
    data = []

    with open(fn, 'r', encoding='utf-8') as fd:
        for line in fd:
            data.append(int(line.rstrip("\r\n"))) # convert to int

    return np.array(data)
    #return np.array([data])

baseline_scores = read(baseline_fn)
system_scores = read(system_fn)
references_scores = read(reference_fn)

#assert len(baseline_scores.shape) == 2
#assert baseline_scores.shape[0] == 1
assert len(baseline_scores.shape) == 1
assert baseline_scores.shape[-1] > 0
assert baseline_scores.shape == system_scores.shape == references_scores.shape, "Input files must have the same number of lines"

# Code adapted from: https://github.com/mjpost/sacrebleu/blob/a5425381d358b27555641bb0903bf4891775cb7b/sacrebleu/significance.py#L112

seed = 42
rng = np.random.default_rng(seed)
baseline_corpus_score = metric(references_scores, baseline_scores)
sys_corpus_score = metric(references_scores, system_scores)
diff = abs(baseline_corpus_score - sys_corpus_score)
statistics = {
    "n_samples": n_samples,
    "baseline_score": round(baseline_corpus_score * 100.0, 2),
    "system_score": round(sys_corpus_score * 100.0, 2),
}

#print(f"Baseline score: {baseline_corpus_score}")
#print(f"System score: {sys_corpus_score}")

# Create selection matrixes:
#pos_sel = rng.integers(2, size=(n_samples, system_scores.shape[-1]), dtype=bool)
#neg_sel = ~pos_sel
# Get permuted samples:
#scores_a = []
#scores_b = []
#
#for _ in range(n_samples):
#    perm1, perm2 = [], []
#
#    for s1, s2 in zip(system_scores[0], baseline_scores[0]):
#        if rng.random() < 0.5:
#            perm1.append(s1)
#            perm2.append(s2)
#        else:
#            perm1.append(s2)
#            perm2.append(s1)
#
#    scores_a.append(metric(references_scores, np.array([perm1])))
#    scores_b.append(metric(references_scores, np.array([perm2])))
mask = rng.integers(0, 2, size=(n_samples, system_scores.shape[-1]), dtype=bool)
shuf_a = np.where(mask, system_scores, baseline_scores)
shuf_b = np.where(mask, baseline_scores, system_scores)

#shuf_a = pos_sel @ np.tile(baseline_scores[0], (system_scores.shape[-1], 1)) + neg_sel @ np.tile(system_scores[0], (system_scores.shape[-1], 1))
#shuf_b = neg_sel @ np.tile(baseline_scores[0], (system_scores.shape[-1], 1)) + pos_sel @ np.tile(system_scores[0], (system_scores.shape[-1], 1))
# Compute scores for each permutation:
scores_a = np.array([metric(references_scores, permuted_scores) for permuted_scores in shuf_a])#[:, None]])
scores_b = np.array([metric(references_scores, permuted_scores) for permuted_scores in shuf_b])#[:, None]])
#scores_a = np.array(scores_a)
#scores_b = np.array(scores_b)
diff_permuted = np.abs(scores_a - scores_b)

assert diff_permuted.shape == (n_samples,)

c = np.sum(diff_permuted >= diff).item()
p = (c + 1) / (len(diff_permuted) + 1)

statistics["c"] = c
statistics["p_value"] = f"{p:.6f}"

#print(f"Approximate Randomization test: p-value = {p:.6f} {'(significant < 0.05)' if p < 0.05 else '(not significant >= 0.05)'}")
print(f"Result ({'p < 0.05' if p < 0.05 else 'p >= 0.05'}): {statistics}")
