
# https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-4-confidence-intervals-from-retraining-models-with-different-random-seeds

import sys

import numpy as np
import scipy

test_accuracies = list(map(float, sys.argv[1:]))
test_mean = np.mean(test_accuracies)
rounds = len(test_accuracies) # number of runs
confidence = 0.95  # Change to your desired confidence level
t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=rounds - 1)
sd = np.std(test_accuracies, ddof=1)
se = sd / np.sqrt(rounds)
ci_length = t_value * se
ci_lower = test_mean - ci_length
ci_upper = test_mean + ci_length

#print(f"[{round(ci_lower, 2)}, {round(ci_upper, 2)}]")
#print(f"{round(ci_lower, 2)} {round(ci_upper, 2)}")
print(round(ci_length, 2))
