#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'
__credits__ = 'Sebastian Padó'
__license__ = 'MIT'

# This is an MIT-licensed implementation of the sigf toolkit for randomization tests:
# https://nlpado.de/~sebastian/software/sigf.shtml

# Original code: https://gist.github.com/dustalov/e6c3b9d3b5b83c81ecd92976e0281d6c

import random
import sys
from statistics import mean


def input_counts(f):
    return [int(line.strip()) for line in f]

def input_binary(f):
    print("# Expected format: 0 or 1 per line (e.g., accuracy if 1 for either tp or tn, otherwise 0)", file=sys.stderr)

    result = []

    for line in f:
        line = int(line.strip())

        assert line in (0, 1), line

        result.append(line)

    return result

def input_tp_fp_fn(f):
    print("# Expected format: tp, tp+fp, tp+fn", file=sys.stderr)

    result = []

    for line in f:
        line = line.strip()
        if line:
            tp, tp_fp, tp_fn = line.split(' ')

            result.append(tuple(int(tp), int(tp_fp), int(tp_fn)))

    return result

def input_conf_mat(f):
    print("# Expected format: tp, fp, fn, tn", file=sys.stderr)

    result = []

    for line in f:
        line = line.strip()
        if line:
            tp, fp, fn, tn = line.split(' ')

            result.append(tuple(int(tp), int(fp), int(fn), int(tn)))

    return result

def input_numerator_and_denominator(f):
    print("# Expected format: numerator, denominator", file=sys.stderr)

    result = []

    for line in f:
        line = line.strip()
        if line:
            num, den = line.split(' ')

            result.append(tuple(int(num), int(den)))

    return result

def input_prediction_per_line_tp_fp_fn_tn(f):
    print("# Expected format: for each prediction per line, the expected values are 'tp', 'fp', 'fn' or 'tn'", file=sys.stderr)

    result = []

    for line in f:
        line = line.strip()

        assert line in ("tp", "fp", "fn", "tn"), line

        if line:
            result.append(tuple(line,))

    return result

def accuracy_prediction_per_line(model):
    tp_tn = sum([(1 if obs[0] in ("tp", "tn") else 0) for obs in model])
    tp_fp_fn_tn = len(model)
    _model = [(tp_tn, tp_fp_fn_tn)]

    return accuracy(_model)

def f1_score(model):
    tp = sum(obs[0] for obs in model)
    tp_fp = sum(obs[1] for obs in model)
    tp_fn = sum(obs[2] for obs in model)

    if tp == 0 or tp_fp == 0 or tp_fn == 0:
        return 0.

    precision, recall = tp / float(tp_fp), tp / float(tp_fn)
    return 2 * precision * recall / (precision + recall)

def f1_score_conf_mat(model):
    _model = [(m[0], m[0] + m[1], m[0] + m[2]) for m in model] # format that f1_score is expecting

    return f1_score(_model)

def accuracy(model):
    tp_tn = sum(obs[0] for obs in model)
    tp_fp_fn_tn = sum(obs[1] for obs in model)
    acc = tp_tn / tp_fp_fn_tn

    assert 0.0 <= acc <= 1.0, acc

    return acc

def randomized_test(model1, model2, score, trials, getrandbits_func):
    print(f"# score(model1) = {score(model1)}", file=sys.stderr)
    print(f"# score(model2) = {score(model2)}", file=sys.stderr)

    diff = abs(score(model1) - score(model2))
    print(f"# abs(diff) = {diff}", file=sys.stderr)

    uncommon = [i for i in range(len(model1)) if model1[i] != model2[i]]

    print(f"# len(uncommon) = {len(uncommon)} ({len(uncommon) * 100 / len(model1):.2f}%)", file=sys.stderr)

    better = 0

    for _ in range(trials):
        model1_local, model2_local = list(model1), list(model2)

        for i in uncommon:
            if getrandbits_func(1) == 1: # equivalent to random.random() < 0.5
                model1_local[i], model2_local[i] = model2[i], model1[i] # shuffle

        assert len(model1_local) == len(model2_local) == len(model1) == len(model2)

        diff_local = abs(score(model1_local) - score(model2_local))

        if diff_local >= diff:
            better += 1

    p = (better + 1) / (trials + 1)
    return p


# Every element of SCORES is a pair of input-reading function and scoring function.
SCORES = {
    'mean': (input_counts, mean),
    'accuracy_mean': (input_binary, mean),
    'f1': (input_tp_fp_fn, f1_score),
    # input_conf_mat is not used for 'accuracy' because it is symmetric with respect to
    # tp and tn, i.e., it is possible that tp + tn = tp' + tn', and tp = tn', tn = tp'
    # This would lead to updating these cases when uncommon is initialized
    'accuracy': (input_numerator_and_denominator, accuracy),
    'f1_conf_mat': (input_conf_mat, f1_score_conf_mat),
    # here it is not necessary to differentiate between tp and tn, because it's not possible
    # to have tp on one side and tn on the other for the same line being evaluated
    'accuracy_prediction_per_line': (input_prediction_per_line_tp_fp_fn_tn, accuracy_prediction_per_line),
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int)
    parser.add_argument('--score', choices=SCORES.keys(), default='accuracy_mean')
    parser.add_argument('--trials', '-n', type=int, default=10 ** 5)
    parser.add_argument('model1', type=argparse.FileType('r'))
    parser.add_argument('model2', type=argparse.FileType('r'))
    args = parser.parse_args()

    if args.seed is None:
        getrandbits_func = random.getrandbits
    else:
        rng = random.Random(args.seed)
        getrandbits_func = rng.getrandbits

    reader, score = SCORES[args.score]

    print("# INFO: 'This program reads in two files containing frequencies of stratified predictions/observations"
          " (e.g. predictions per sentence, per item, etc)'", file=sys.stderr)
    if args.score in ("accuracy_mean", "accuracy_prediction_per_line",):
        print(f"# WARNING: score is '{args.score}': note the previous message: since the data is expected to be"
              " *stratified*, with the chosen score approach, each data entry is treated as a subset of results,"
              " 'as its own stratum'", file=sys.stderr)

    model1, model2 = reader(args.model1), reader(args.model2)
    assert len(model1) == len(model2)

    p = randomized_test(model1, model2, score, args.trials, getrandbits_func)
    print(f"p-value = {p:.4f}")


if '__main__' == __name__:
    main()
