
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from transformers import (
    get_linear_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
from torch.optim import Adam, AdamW, SGD
import transformers
import accelerate

use_accelerator = True # change manually if needed

if use_accelerator:
    accelerator = accelerate.Accelerator()

lr_scheduler_args = ["10%"] # inverse_sqrt: warmup_steps
optimizer_args = [0.9, 0.999, 1e-08, 0.0] # adam: beta1, beta2, eps, weight_decay

def get_lr_scheduler_inverse_sqrt(optimizer, num_warmup_steps, **kwargs):
    # Deprecated! use get_inverse_sqrt_schedule
    def inverse_sqrt(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        initial_lr = optimizer.defaults["lr"]
        decay_factor = initial_lr * num_warmup_steps**0.5
        lr = decay_factor * current_step**-0.5
        return lr / initial_lr
    scheduler_instance = LambdaLR(optimizer, inverse_sqrt, **kwargs)
    return scheduler_instance

model = transformers.AutoModel.from_pretrained("xlm-roberta-base")
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
learning_rate = 1e-5
optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }

training_steps = 10000 # fake value

if lr_scheduler_args[0][-1] == '%':
    scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps)]
else:
    scheduler_args = [int(lr_scheduler_args[0])]

#print(f"optimizer_kwargs: {optimizer_kwargs}")
#print(f"scheduler_args: {scheduler_args}")

optimizer = Adam(model_parameters, lr=learning_rate, **optimizer_kwargs)
#scheduler = get_lr_scheduler_inverse_sqrt(optimizer, *scheduler_args)
scheduler = get_inverse_sqrt_schedule(optimizer, *scheduler_args)

if use_accelerator:
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

step = 0

while step < training_steps:
    current_lr = scheduler.get_last_lr()

    assert len(current_lr) == 1, current_lr

    current_lr = current_lr[0]

    print(f"{step}\t{current_lr}")

    scheduler.step()

    step += 1

################
# No accelerate:
################
## 0       0.0
## 1       1e-08
## 2       2e-08
## 3       3.0000000000000004e-08
## 4       4e-08
## 5       5.0000000000000004e-08
## 6       6.000000000000001e-08
## 7       7e-08
## 8       8e-08
## 9       9e-08
## 10      1.0000000000000001e-07
## 11      1.1e-07
## 12      1.2000000000000002e-07
## 13      1.3e-07
## 14      1.4e-07
## 15      1.5000000000000002e-07
## 16      1.6e-07
#################
# accelerate n=2:
#################
## 0       0.0
## 1       2e-08
## 2       4e-08
## 3       6.000000000000001e-08
## 4       8e-08
## 5       1.0000000000000001e-07
## 6       1.2000000000000002e-07
## 7       1.4e-07
## 8       1.6e-07
#################
# accelerate n=4:
#################
## 0       0.0
## 1       4e-08
## 2       8e-08
## 3       1.2000000000000002e-07
## 4       1.6e-07
#################
# accelerate n=8:
#################
## 0       0.0
## 1       8e-08
## 2       1.6e-07
