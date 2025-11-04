
import sys

import torch
import transformers

def check(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def load_model(pretrained_model, state_dict_path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=1)
    state_dict = torch.load(state_dict_path, weights_only=True)

    model.load_state_dict(state_dict)

    return model

pretrained_model = sys.argv[1]
model1 = load_model(pretrained_model, sys.argv[2])
model2 = load_model(pretrained_model, sys.argv[3])

print(f"Are models equal? {check(model1, model2)}")
