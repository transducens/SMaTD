
import os
import sys
import copy
import random
import logging
import argparse
import warnings
import contextlib

import mtdetect.utils.utils as utils
import mtdetect.dataset as dataset
import mtdetect.inference as inference

import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.optim import Adam, AdamW, SGD
import torch.nn as nn
import transformers
from transformers import (
    get_linear_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
import accelerate
import numpy as np
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("mtdetect")
accelerator = None
writer = None

_lr_scheduler_args = {
    "none": {},
    "linear": {
        "nargs": 1,
        "metavar": ("warmup_steps",),
        "default": ("10%",), # '%' is optional, and if not provided, absolute number of steps is taken
        "type": utils.argparse_nargs_type(str),
    },
    "CLR": {
        "nargs": 6,
        "metavar": ("max_lr", "step_size", "mode", "gamma", "max_lr_factor", "step_size_factor"),
        "default": (8e-5, 2000, "triangular2", 1.0, 4, 2),
        "type": utils.argparse_nargs_type(float, int, str, float, {"type": int, "choices": (3, 4)},
                                            {"type": int, "choices": tuple(range(2,8+1))}),
    },
    "inverse_sqrt": {
        "nargs": 1,
        "metavar": ("warmup_steps",),
        "default": ("10%",), # '%' is optional, and if not provided, absolute number of steps is taken
        "type": utils.argparse_nargs_type(str),
    }
}
_optimizer_args = {
    "none": {},
    "adam": {
        "nargs": 4,
        "metavar": ("beta1", "beta2", "eps", "weight_decay"),
        "default": (0.9, 0.999, 1e-08, 0.0),
        "type": utils.argparse_nargs_type(float, float, float, float),
    },
    "adamw": {
        "nargs": 4,
        "metavar": ("beta1", "beta2", "eps", "weight_decay"),
        "default": (0.9, 0.999, 1e-08, 0.01),
        "type": utils.argparse_nargs_type(float, float, float, float),
    },
    "sgd": {
        "nargs": 2,
        "metavar": ("momentum", "weight_decay"),
        "default": (0.0, 0.0),
        "type": utils.argparse_nargs_type(float, float),
    }
}

def get_lr_scheduler(scheduler, optimizer, *args, **kwargs):
    scheduler_instance = None
    mandatory_args = ""

    def check_args(num_args, str_args):
        if len(args) != num_args:
            raise Exception(f"LR scheduler: '{scheduler}' mandatory args: {str_args}")

    if scheduler == "none":
        pass
    elif scheduler == "linear":
        mandatory_args = "num_warmup_steps, num_training_steps"

        check_args(2, mandatory_args)

        scheduler_instance = get_linear_schedule_with_warmup(optimizer, *args, **kwargs)
    elif scheduler == "CLR": # CyclicLR
        mandatory_args = "base_lr, max_lr"

        check_args(2, mandatory_args)

        scheduler_instance = CyclicLR(optimizer, *args, **kwargs)
    elif scheduler == "inverse_sqrt":
        mandatory_args = "num_warmup_steps"

        check_args(1, mandatory_args)

        if optimizer is None:
            raise Exception(f"Optimizer not provided, so the selected LR scheduler can't be configured: {scheduler}")

        scheduler_instance = get_inverse_sqrt_schedule(optimizer, *args, **kwargs)
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    logger.debug("LR scheduler: '%s' mandatory args: %s: %s", scheduler, mandatory_args, str(args))
    logger.debug("LR scheduler: '%s' optional args: %s", scheduler, str(kwargs))

    return scheduler_instance

def save_model(model, model_output=None, name="mtd"):
    # Be aware that all threads need to reach this function
    accelerator.wait_for_everyone()

    if model_output is None:
        return

    unwrapped_model = accelerator.unwrap_model(model)
    #unwrapped_model.save_pretrained(
    #    model_output,
    #    is_main_process=accelerator.is_main_process,
    #    save_function=accelerator.save,
    #)

    if accelerator.is_local_main_process:
        accelerator.save(unwrapped_model.state_dict(), f"{model_output}/{name}.pt")

        logger.info("Model saved: %s", f"{model_output}/{name}.pt")

    accelerator.wait_for_everyone()

def load_model(model_input, pretrained_model, device, name=None, classifier_dropout=0.0):
    local_model = model_input is not None
    config = transformers.AutoConfig.from_pretrained(pretrained_model, num_labels=1, classifier_dropout=classifier_dropout)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
    loaded_model = f"{pretrained_model}:{model_input}" if local_model else pretrained_model

    if local_model:
        model = accelerator.unwrap_model(model)
        _model_input = f"{model_input}/{name}.pt" if name is not None else model_input
        state_dict = torch.load(_model_input, weights_only=True, map_location=device) # weights_only: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
                                                                                      # map_location: avoid creating a new process and using additional and useless memory

        model.load_state_dict(state_dict)

    logger.info("Model loaded: %s (local instead of pretrained? %s)", loaded_model, local_model)

    model = model.to(device)

    return model

def load_dataset(filename_dataset, set_desc, **kwargs):
    logger.debug("Allocated memory before starting tokenization (%s): %d", set_desc, utils.get_current_allocated_memory_size())

    file_dataset = open(filename_dataset, mode="rt", errors="backslashreplace")
    input_data = []
    output_data = []

    # Read data from input files
    batch = dataset.tokenize_batch_from_iterator(file_dataset, kwargs["tokenizer"], kwargs["batch_size"], ignore_source_side=kwargs["monolingual"])

    for batch_urls in batch:
        input_data.extend(batch_urls["urls"])
        output_data.extend(batch_urls["labels"])

        if len(input_data) != len(output_data):
            raise Exception(f"Different lengths for input and output data in {set_desc} set: {len(input_data)} vs {len(output_data)}")

    non_parallel_urls = len([l for l in output_data if l == 0])
    parallel_urls = len([l for l in output_data if l == 1])

    if non_parallel_urls + parallel_urls != len(input_data):
        raise Exception(f"Number of non-parallel + parallel URLs doesn't match the input data ({set_desc}): "
                        f"{non_parallel_urls} + {parallel_urls} != {len(input_data)}")

    logger.info("%d pairs of parallel URLs loaded (%s)", parallel_urls, set_desc)
    logger.info("%d pairs of non-parallel URLs loaded (%s)", non_parallel_urls, set_desc)
    logger.debug("Allocated memory after tokenization (%s): %d", set_desc, utils.get_current_allocated_memory_size())

    # Datasets
    dataset_instance = dataset.SmartBatchingURLsDataset(input_data, output_data, kwargs["tokenizer"], kwargs["max_length_tokens"], set_desc=set_desc)

    logger.debug("Allocated memory after encoding the data: %d", utils.get_current_allocated_memory_size())
    logger.debug("Total tokens (%s): %d", set_desc, dataset_instance.total_tokens)

    # Remove data in order to free memory
    del input_data
    del output_data

    logger.debug("Allocated memory after removing pairs of URLs (str): %d", utils.get_current_allocated_memory_size())

    dataloader_instance = dataset_instance.get_dataloader(kwargs["batch_size"], kwargs["device"], kwargs["dataset_workers"])

    file_dataset.close()

    return dataset_instance, dataloader_instance

def apply_patience(model, tokenizer, dataset_dev, loss_function, device, threshold, dev_best_metric, epoch, dev_best_metric_value,
                   current_patience, patience, dev_best_epoch, model_output):
    #if accelerator.is_local_main_process: # Do not even think about it ;) for some good reason (that I would like to know), it gets stuck
    dev_eval = inference.inference_eval(model, tokenizer, dataset_dev, loss_function=loss_function, device=device, threshold=threshold)
    dev_patience_metric = dev_eval[dev_best_metric]

    if accelerator.is_local_main_process:
        logger.debug("[step_or_epoch:%s] Dev eval all metrics: %s", epoch, dev_eval)
        logger.info("[step_or_epoch:%s] Dev eval metric (%s): %s", epoch, dev_best_metric, dev_patience_metric)

        writer.add_scalar(f"{dev_best_metric}/dev", dev_patience_metric, epoch)

    if dev_patience_metric <= dev_best_metric_value:
        current_patience += 1

        if patience > 0 and accelerator.is_local_main_process:
            logger.info("Exhausting patience... %d/%d", current_patience, patience)
    else:
        if accelerator.is_local_main_process:
            logger.info("Best dev patience metric update: %s -> %s", dev_best_metric_value, dev_patience_metric)

        dev_best_metric_value = dev_patience_metric
        dev_best_epoch = epoch
        current_patience = 0

        save_model(model, model_output=model_output, name="mtd_best_dev")

    return current_patience, dev_best_metric_value, dev_best_epoch

def main(args):
    apply_inference = args.inference

    if not apply_inference:
        filename_dataset_train = args.dataset_train_filename
        filename_dataset_dev = args.dataset_dev_filename
        filename_dataset_test = args.dataset_test_filename

    # Args
    batch_size = args.batch_size
    epochs = args.epochs # BE AWARE! "epochs" might be fake due to --train-until-patience
    pretrained_model = args.pretrained_model
    max_length_tokens = args.max_length_tokens
    model_input = utils.resolve_path(args.model_input)
    model_output = utils.resolve_path(args.model_output)
    seed = args.seed
    patience = args.patience
    train_until_patience = args.train_until_patience
    learning_rate = args.learning_rate
    scheduler_str = args.lr_scheduler
    lr_scheduler_args = args.lr_scheduler_args # Content might vary depending on the value of scheduler_str
    optimizer_str = args.optimizer
    optimizer_args = args.optimizer_args # Content might vary depending on the value of optimizer_str
    dataset_workers = args.dataset_workers
    monolingual = args.monolingual
    threshold = args.threshold
    eval_strategy = args.strategy
    eval_steps = args.strategy_steps
    classifier_dropout = args.classifier_dropout
    gradient_accumulation_steps = args.gradient_accumulation

    if gradient_accumulation_steps > 1:
        assert (batch_size % gradient_accumulation_steps) == 0, f"batch_size % gradient_accumulation_steps != 0 -> {batch_size % gradient_accumulation_steps} != 0"

        _batch_size = batch_size // gradient_accumulation_steps

        logger.info("Gradient accumulation enabled: batch_size: %d -> %d", batch_size, _batch_size)

        batch_size = _batch_size

    # Model
    device = accelerator.device
    tokenizer = utils.get_tokenizer(pretrained_model, logger=logger)
    model = load_model(model_input, pretrained_model, device, name=None, classifier_dropout=classifier_dropout)
    num_processes = accelerator.num_processes
    save_model_prefix = f"mtd_{eval_strategy}"
    actual_batch_size = num_processes * batch_size * gradient_accumulation_steps # https://discuss.huggingface.co/t/what-is-my-batch-size/41390

    if accelerator.is_local_main_process:
        logger.debug("Number of processes: %d", num_processes)
        logger.info("Actual batch size: %d", actual_batch_size)

    save_model(model, model_output=model_output, name=f"{save_model_prefix}_0")

    ##################

    training_context_manager = accelerator.accumulate if gradient_accumulation_steps > 1 else contextlib.nullcontext

    if scheduler_str in ("linear",) and train_until_patience:
        # Depending on the LR scheduler, the training might even stop at some point (e.g. linear LR scheduler will set the LR=0 if the run epochs is greater than the provided epochs)
        logger.warning("You set a LR scheduler ('%s' scheduler) which conflicts with --train-until-patince: you might want to check this out and change the configuration", scheduler_str)

    if apply_inference and not model_input:
        logger.warning("Flag --model-input is recommended when --inference is provided")

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.debug("Deterministic values enabled (not fully-guaranteed): seed %d", seed)
    else:
        logger.warning("Deterministic values disable (you set a negative seed)")

    max_length_tokens = utils.get_encoder_max_length(model, tokenizer, max_length_tokens=max_length_tokens, pretrained_model=pretrained_model, logger=logger)

    logger.info("Max token length: %d", max_length_tokens)

    if not apply_inference:
        logger.debug("Train data file/s: %s", filename_dataset_train)
        logger.debug("Dev data file: %s", filename_dataset_dev)
        logger.debug("Test data file: %s", filename_dataset_test)

    try:
        model_embeddings_size = model.base_model.embeddings.word_embeddings.weight.shape[0]

        if model_embeddings_size != len(tokenizer):
            # microsoft/deberta-v3-large -> 128100 vs 128001 (why 99 unknown tokens in the model?)
            logger.error("Embedding layer size does not match with the tokenizer size: %d vs %d", model_embeddings_size, len(tokenizer))
    except AttributeError:
        logger.warning("Could not get the embedding size...")

    loss_function = nn.BCEWithLogitsLoss(reduction="mean") # Regression: raw input, not normalized
                                                           #  (i.e. sigmoid is applied in the loss function)

    if apply_inference:
        assert writer is None

        if accelerator.is_local_main_process:
            metrics = inference.inference_from_stdin(model, tokenizer, batch_size, loss_function=loss_function, device=device, max_length_tokens=max_length_tokens,
                                                     threshold=threshold, monolingual=monolingual, dataset_workers=dataset_workers)

            if metrics is not None:
                logger.info("Inference metrics: %s", metrics)

        logger.info("Done!")

        # Stop execution
        return

    # Unfreeze model layers
    for param in model.parameters():
        param.requires_grad = True

    # Load data
    dataset_static_args = {
        "batch_size": batch_size,
        "device": device,
        "dataset_workers": dataset_workers,
        "tokenizer": tokenizer,
        "max_length_tokens": max_length_tokens,
        "monolingual": monolingual,
    }
    dataset_train, dataloader_train = load_dataset(filename_dataset_train, "train", **dataset_static_args)
    dataset_dev, _ = load_dataset(filename_dataset_dev, "dev", **dataset_static_args)

    #training_steps_per_epoch = len(dataloader_train) // num_processes
    training_steps_per_epoch = len(dataloader_train) # it counts batches, not samples (value = samples // batch_size)!
                                                     # it is not necessary to divide by 'num_processes' -> if warmup is 10 steps and 2 GPUs, warmup will finish at step 5
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to --train-until-patience

    # I have checked out that the use of accelerate+LR scheduler is correctly adapted according to the number of GPUs used for training

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    logger.debug("Optimizer args: %s", optimizer_args)

    if optimizer_str == "none":
        optimizer = None

        logger.debug("Be aware that even with the optimizer disabled minor changes might be observed while training since the model is "
                     "not in inference mode, so layers like Dropout have a random component which is enabled")
    elif optimizer_str == "adam":
        optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }
        optimizer = Adam(model_parameters, lr=learning_rate, **optimizer_kwargs)
    elif optimizer_str == "adamw":
        optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }
        optimizer = AdamW(model_parameters, lr=learning_rate, **optimizer_kwargs)
    elif optimizer_str == "sgd":
        optimizer_kwargs = {
            "momentum": optimizer_args[0],
            "weight_decay": optimizer_args[1],
        }
        optimizer = SGD(model_parameters, lr=learning_rate, **optimizer_kwargs)
    else:
        raise Exception(f"Unknown optimizer: {optimizer_str}")

    # Get LR scheduler args
    scheduler_args = []
    scheduler_kwargs = {}

    logger.debug("LR scheduler args: %s", lr_scheduler_args)

    if scheduler_str == "none":
        pass
    elif scheduler_str == "linear":
        if lr_scheduler_args[0][-1] == '%':
            scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps), training_steps]
        else:
            scheduler_args = [int(lr_scheduler_args[0]), training_steps]
    elif scheduler_str == "CLR":
        scheduler_max_lr, scheduler_step_size, scheduler_mode, scheduler_gamma, scheduler_max_lr_factor, scheduler_step_size_factor \
            = lr_scheduler_args

        if learning_rate > scheduler_max_lr:
            new_scheduler_max_lr = learning_rate * scheduler_max_lr_factor # Based on the CLR paper (possible values are [3.0, 4.0])

            logger.warning("LR scheduler: '%s': provided LR (%f) is greater than provided max. LR (%f): setting max. LR to %f",
                           scheduler_str, learning_rate, scheduler_max_lr, new_scheduler_max_lr)

            scheduler_max_lr = new_scheduler_max_lr
        if scheduler_step_size <= 0:
            scheduler_step_size = scheduler_step_size_factor * training_steps_per_epoch # Based on the CLR paper (possible values are [2, ..., 8])

            logger.warning("LR scheduler: '%s': provided step size is 0 or negative: setting value to %d", scheduler_str, scheduler_step_size)

        scheduler_args = [learning_rate, scheduler_max_lr]
        scheduler_kwargs = {
            "step_size_up": scheduler_step_size,
            "step_size_down": scheduler_step_size,
            "mode": scheduler_mode,
            "gamma": scheduler_gamma,
            "cycle_momentum": False, # https://github.com/pytorch/pytorch/issues/73910
        }
    elif scheduler_str == "inverse_sqrt":
        if lr_scheduler_args[0][-1] == '%':
            scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps)]
        else:
            scheduler_args = [int(lr_scheduler_args[0])]
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    scheduler = get_lr_scheduler(scheduler_str, optimizer, *scheduler_args, **scheduler_kwargs)

    model, optimizer, dataloader_train, scheduler = accelerator.prepare(model, optimizer, dataloader_train, scheduler)

    stop_training = False
    epoch = 0
    current_patience = 0
    dev_best_metric = "acc"
    dev_best_epoch = 0
    dev_best_metric_value = -np.inf
    global_step = 0
    saved_epochs_or_steps = [0]
    all_loss = []

    while not stop_training:
        if accelerator.is_local_main_process:
            logger.info("Epoch %d", epoch + 1)

        model.train()

        duplicated_data = {}
        step = 0
        patience_applied = False

        for batch in dataloader_train:
            with training_context_manager(model):
                #optimizer.zero_grad() # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
                model.zero_grad()

                inputs = batch["url_tokens"]
                result = inference.inference(model, batch, loss_function=loss_function, device=device, threshold=threshold)
                loss = result["loss"]

                for i in inputs:
                    # Sanity check for later
                    _i = tokenizer.decode(i.squeeze().tolist(), skip_special_tokens=True)

                    if _i not in duplicated_data:
                        duplicated_data[_i] = 0

                    duplicated_data[_i] += 1

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                if accelerator.sync_gradients:
                    loss = accelerator.gather(loss).mean().item()

                    all_loss.append(loss)

                    step += 1
                    global_step += 1
                else:
                    continue # avoid executing the following conditions more than once, because some variables will have the same value as in the previous iteration of the loop

            # When we reach this point, we have processed actual_batch_size samples (i.e., 1 batch)!

            if accelerator.is_main_process:
                writer.add_scalar("loss/train", loss, global_step)

                if global_step % 100 == 0:
                    _loss = sum(all_loss[-100:])

                    logger.debug("[global_step:%d] Training loss (sum last 100 steps): %s", global_step, _loss)

            if eval_strategy == "steps" and global_step % eval_steps == 0:
                # Patience
                save_model_suffix = epoch if eval_strategy == "epoch" else global_step
                current_patience, dev_best_metric_value, dev_best_epoch = apply_patience(
                    model, tokenizer, dataset_dev, loss_function, device, threshold, dev_best_metric, save_model_suffix, dev_best_metric_value,
                    current_patience, patience, dev_best_epoch, model_output)
                patience_applied = True

                # Save model
                save_model(model, model_output=model_output, name=f"{save_model_prefix}_{save_model_suffix}")
                saved_epochs_or_steps.append(save_model_suffix)

                # Stop training?
                if patience > 0 and current_patience >= patience:
                    # End of patience
                    stop_training = True

                    if accelerator.is_local_main_process:
                        logger.debug("Training stopped in the middle of an epoch")

                    break

            patience_applied = False

        pre_stop = False

        if stop_training:
            pre_stop = True

            assert eval_strategy == "steps", eval_strategy
            assert patience_applied, patience_applied
        elif not monolingual: # if monolingual, duplicate data may be found...
            total_duplicated = 0

            for i, v in duplicated_data.items():
                total_duplicated += v - 1

            #assert total_duplicated < batch_size, f"{total_duplicated} >= {batch_size}" # True when there are no duplicates in the data...
            #assert abs(len(duplicated_data) - batch_size * training_steps_per_epoch // num_processes) < batch_size, f"abs({len(duplicated_data)} - {batch_size} * {training_steps_per_epoch} // {num_processes}) >= {batch_size}" # True when there are no duplicates in the data...

        epoch += 1

        if not patience_applied:
            assert not stop_training, stop_training
            assert not pre_stop, pre_stop

        last_epoch = epoch >= epochs
        last_step_patience = (eval_strategy == "steps") and (not train_until_patience) and last_epoch and (not patience_applied) # Let's check the patience again in the last step if strategy is steps

        if eval_strategy == "epoch" or last_step_patience:
            # Patience
            save_model_suffix = epoch if eval_strategy == "epoch" else global_step
            current_patience, dev_best_metric_value, dev_best_epoch = apply_patience(
                model, tokenizer, dataset_dev, loss_function, device, threshold, dev_best_metric, save_model_suffix, dev_best_metric_value,
                current_patience, patience, dev_best_epoch, model_output)

            # Save model
            save_model(model, model_output=model_output, name=f"{save_model_prefix}_{save_model_suffix}")
            saved_epochs_or_steps.append(save_model_suffix)

        # Stop training?
        if patience > 0 and current_patience >= patience:
            # End of patience
            # We do not set as condition 'train_until_patience'; then, if patience > 0, training can stop because of patience or epochs

            stop_training = True
        elif not train_until_patience:
            stop_training = last_epoch

        if pre_stop:
            assert stop_training

    # Free memory
    del dataloader_train
    del dataset_train
    del dataset_dev
    del model

    torch.cuda.empty_cache()

    # Eval test
    models_not_available = model_output is None
    results_keys, results_values = [], []
    dataset_test, _ = load_dataset(filename_dataset_test, "test", **dataset_static_args)

    if models_not_available:
        saved_epochs_or_steps = [epoch if strategy == "epoch" else global_step]

    for idx, epoch_or_step in enumerate(saved_epochs_or_steps):
        if idx % accelerator.num_processes != accelerator.process_index:
            continue

        logger.debug("[step_or_epoch:%s] Evaluating test set", epoch_or_step)

        if not models_not_available:
            # Eval model trained on epoch {epoch_or_step}
            model_name = f"{save_model_prefix}_{epoch_or_step}"
            desc = f" (best dev model)" if epoch_or_step == dev_best_epoch else ''
            model = load_model(model_output, pretrained_model, device, name=model_name, classifier_dropout=classifier_dropout)
        # else: eval last model

        test_eval = inference.inference_eval(model, tokenizer, dataset_test, loss_function=loss_function, device=device, threshold=threshold)

        results_keys.append(epoch_or_step)
        results_values.append(test_eval)

    accelerator.wait_for_everyone()

    results_keys = accelerate.utils.gather_object(results_keys)
    results_values = accelerate.utils.gather_object(results_values)
    results = {k: v for k, v in zip(results_keys, results_values)}

    # Print results
    if accelerator.is_local_main_process:
        for idx, epoch_or_step in enumerate(saved_epochs_or_steps):
            test_eval = results[epoch_or_step]

            writer.add_scalar(f"{dev_best_metric}/test", test_eval[dev_best_metric], epoch_or_step)

            if models_not_available:
                logger.info("Test metrics: %s", test_eval)
            else:
                model_name = f"{save_model_prefix}_{epoch_or_step}"
                desc = f" (best dev model)" if epoch_or_step == dev_best_epoch else ''

                logger.info("Test metrics for model %s%s: %s", model_name, desc, test_eval)

    logger.info("Done!")

def get_options_from_argv(argv_flag, default_value, dict_with_options):
    choices = list(dict_with_options.keys())
    args_options = dict_with_options[default_value]

    if argv_flag in sys.argv:
        idx = sys.argv.index(argv_flag)

        if len(sys.argv) > idx + 1:
            value = sys.argv[idx + 1]

            if value in choices:
                args_options = dict_with_options[value]

    result = {
        "default": default_value,
        "choices": choices,
        "options": args_options,
    }

    return result

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="MTDetect")
    inference = "--inference" in sys.argv
    lr_scheduler_conf = get_options_from_argv("--lr-scheduler", "inverse_sqrt", _lr_scheduler_args)
    optimizer_conf = get_options_from_argv("--optimizer", "adamw", _optimizer_args)

    if not inference:
        parser.add_argument('dataset_train_filename', type=str, help="Filename with train data (TSV format). Format: original text (OT), machine (MT) or human (HT) translation, 0 if MT or 1 if HT")
        parser.add_argument('dataset_dev_filename', type=str, help="Filename with dev data (TSV format)")
        parser.add_argument('dataset_test_filename', type=str, help="Filename with test data (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size. Elements which will be processed before proceed to train")
    parser.add_argument('--epochs', type=int, default=3, help="Epochs")
    parser.add_argument('--dataset-workers', type=int, default=-1,
                        help="No. workers when loading the data in the dataset. When negative, all available CPUs will be used")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=512, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', help="Model input path which will be loaded")
    parser.add_argument('--model-output', help="Model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true",
                        help="Do not train, just apply inference reading from stdin (flag --model-input is recommended). "
                             "If this option is set, it will be necessary to do not provide the input dataset.")
    parser.add_argument('--patience', type=int, default=0,
                        help="Patience to stop training. If the specified value is greater than 0, epochs and patience will be taken into account")
    parser.add_argument('--train-until-patience', action="store_true",
                        help="Train until patience value is reached (--epochs will be ignored in order to stop, but will still be "
                             "used for other actions like LR scheduler)")
    parser.add_argument('--learning-rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--optimizer', choices=optimizer_conf["choices"], default=optimizer_conf["default"], help="Optimizer")
    parser.add_argument('--optimizer-args', **optimizer_conf["options"],
                        help="Args. for the optimizer (in order to see the specific configuration for a optimizer, use -h and set --optimizer)")
    parser.add_argument('--lr-scheduler', choices=lr_scheduler_conf["choices"], default=lr_scheduler_conf["default"], help="LR scheduler")
    parser.add_argument('--lr-scheduler-args', **lr_scheduler_conf["options"],
                        help="Args. for LR scheduler (in order to see the specific configuration for a LR scheduler, "
                             "use -h and set --lr-scheduler)")
    parser.add_argument('--monolingual', action="store_true",
                        help="Only the MT or HT will be processed instead of OT+MT|HT. This does not change the expected format in the data: OT+MT|HT+label")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Threshold to consider a given text to be HT")
    parser.add_argument('--strategy', type=str, choices=["steps", "epoch"], default="epoch", help="Strategy for evaluating and saving. This will also affect to --patience")
    parser.add_argument('--strategy-steps', type=int, default=1000, help="Steps to evaluate and save the model when the strategy is 'steps'")
    parser.add_argument('--classifier-dropout', type=float, default=0.1, help="Dropout applied to the classifier layer")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument('--seed', type=int, default=71213,
                        help="Seed in order to have deterministic results (not fully guaranteed). "
                             "Set a negative number in order to disable this feature")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

def cli():
    global logger
    global accelerator
    global writer

    assert accelerator is None
    assert writer is None

    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    # Accelerate
    assert args.gradient_accumulation >= 1, args.gradient_accumulation

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation)

    # Logging
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO, accelerator=accelerator)

    if accelerator.is_local_main_process:
        logger.debug("Arguments processed: {}".format(str(args))) # First logging message should be the processed arguments

        if not args.inference:
            # Let's avoid creating directories to do not log anything
            writer = SummaryWriter()

            logger.info("TensorBoard directory: %s", writer.log_dir)

    main(args)

if __name__ == "__main__":
    cli()
