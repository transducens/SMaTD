
import os
import sys
import copy
import random
import logging
import argparse

import mtdetect.utils.utils as utils
import mtdetect.dataset as dataset

import torch
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import Adam, AdamW, SGD
import torch.nn as nn
import transformers
from transformers import (
    get_linear_schedule_with_warmup,
)
import accelerate
import numpy as np

logger = logging.getLogger("mtdetect")
accelerator = None

DEBUG = bool(int(os.environ["MTD_DEBUG"])) if "MTD_DEBUG" in os.environ else False
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

        def inverse_sqrt(current_step):
            num_warmup_steps = args[0]

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # From https://fairseq.readthedocs.io/en/latest/_modules/fairseq/optim/lr_scheduler/inverse_square_root_schedule.html
            # In fairseq they set directly the LR to the optimizer, but we use it for a LR scheduler, so for us is a value which will multiply the LR
            initial_lr = optimizer.defaults["lr"]
            decay_factor = initial_lr * num_warmup_steps**0.5
            lr = decay_factor * current_step**-0.5

            return lr / initial_lr # This step makes that the multiplication of initial_lr doesn't affect, but the previous lines are just being similar
                                   #  to the version of fairseq

        scheduler_instance = LambdaLR(optimizer, inverse_sqrt, **kwargs)
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    logger.debug("LR scheduler: '%s' mandatory args: %s: %s", scheduler, mandatory_args, str(args))
    logger.debug("LR scheduler: '%s' optional args: %s", scheduler, str(kwargs))

    return scheduler_instance

def save_model(accelerator, model, model_output, name="mtd"):
    # Be aware that all threads need to reach this function
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    #unwrapped_model.save_pretrained(
    #    model_output,
    #    is_main_process=accelerator.is_main_process,
    #    save_function=accelerator.save,
    #)

    if accelerator.is_local_main_process:
        accelerator.save(unwrapped_model.state_dict(), f"{model_output}/{name}.pt")

        logger.info("%d: model saved: %s", accelerator.process_index, model_output)

    accelerator.wait_for_everyone()

def load_model(accelerator, model_input, pretrained_model, device, name="mtd"):
    local_model = model_input is not None
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=1)
    loaded_model = f"{pretrained_model}:{model_input}" if local_model else pretrained_model

    if local_model:
        model = accelerator.unwrap_model(model)
        state_dict = torch.load(f"{model_input}/{name}.pt", weights_only=True) # https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models

        model.load_state_dict(state_dict)

    logger.info("%d: model loaded: %s (local instead of pretrained? %s)", accelerator.process_index, loaded_model, local_model)

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
    dataset_instance = dataset.SmartBatchingURLsDataset(input_data, output_data, kwargs["tokenizer"],
                                                        kwargs["max_length_tokens"], set_desc=set_desc,
                                                        remove_instead_of_truncate=kwargs["remove_instead_of_truncate"])

    logger.debug("Allocated memory after encoding the data: %d", utils.get_current_allocated_memory_size())
    logger.debug("Total tokens (%s): %d", set_desc, dataset_instance.total_tokens)

    # Remove data in order to free memory
    del input_data
    del output_data

    logger.debug("Allocated memory after removing pairs of URLs (str): %d", utils.get_current_allocated_memory_size())

    dataloader_instance = dataset_instance.get_dataloader(kwargs["batch_size"], kwargs["device"],
                                                          kwargs["dataset_workers"], max_tokens=kwargs["max_tokens"])

    file_dataset.close()

    return dataset_instance, dataloader_instance

def main(args):
    apply_inference = args.inference

    if not apply_inference:
        filename_dataset_train = args.dataset_train_filename
        filename_dataset_dev = args.dataset_dev_filename
        filename_dataset_test = args.dataset_test_filename

    # Args
    batch_size = args.batch_size
    #block_size = args.block_size
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    epochs = args.epochs # BE AWARE! "epochs" might be fake due to --train-until-patience
    pretrained_model = args.pretrained_model
    max_length_tokens = args.max_length_tokens
    model_input = utils.resolve_path(args.model_input)
    model_output = utils.resolve_path(args.model_output)
    seed = args.seed
    inference_from_stdin = args.inference_from_stdin
    patience = args.patience
    train_until_patience = args.train_until_patience
    learning_rate = args.learning_rate
    scheduler_str = args.lr_scheduler
    lr_scheduler_args = args.lr_scheduler_args # Content might vary depending on the value of scheduler_str
    remove_instead_of_truncate = args.remove_instead_of_truncate
    optimizer_str = args.optimizer
    optimizer_args = args.optimizer_args # Content might vary depending on the value of optimizer_str
    dataset_workers = args.dataset_workers
    monolingual = args.monolingual

    # Model
    #use_cuda = utils.use_cuda(force_cpu=force_cpu) # Will be True if possible and False otherwise
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = accelerator.device
    #is_device_gpu = device.type.startswith("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
    model = load_model(accelerator, model_input, pretrained_model, device, name="mtd")
    num_processes = accelerator.num_processes

    logger.debug("Number of processes: %d", num_processes)

    if model_output is not None:
        save_model(accelerator, model, model_output, name="mtd")

    ##################

    if scheduler_str in ("linear",) and train_until_patience:
        # Depending on the LR scheduler, the training might even stop at some point (e.g. linear LR scheduler will set the LR=0 if the run epochs is greater than the provided epochs)
        logger.warning("You set a LR scheduler ('%s' scheduler) which conflicts with --train-until-patince: you might want to check this out and change the configuration", scheduler_str)

    if apply_inference and not model_input:
        logger.warning("Flag --model-input is recommended when --inference is provided: waiting %d seconds before proceed", waiting_time)

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.debug("Deterministic values enabled (not fully-guaranteed): seed %d", seed)
    else:
        logger.warning("Deterministic values disable (you set a negative seed)")

    if max_length_tokens > tokenizer.model_max_length:
        logger.warning("%s can handle a max. of %d tokens at once but you set %d: changing value to %d", pretrained_model, tokenizer.model_max_length, max_length_tokens, tokenizer.model_max_length)

        max_length_tokens = tokenizer.model_max_length

    if not apply_inference:
        logger.debug("Train data file/s: %s", filename_dataset_train)
        logger.debug("Dev data file: %s", filename_dataset_dev)
        logger.debug("Test data file: %s", filename_dataset_test)

    model_embeddings_size = model.base_model.embeddings.word_embeddings.weight.shape[0]

    assert model_embeddings_size == len(tokenizer), f"Embedding layer size does not match with the tokenizer size: {model_embeddings_size} vs {len(tokenizer)}"

    if max_tokens and max_tokens < max_length_tokens:
        logger.warning("The specified max_tokens has to be greater or equal that the max length tokens of the model: "
                       "changing value from %d to %d", max_tokens, max_length_tokens)

        max_tokens = max_length_tokens

    if apply_inference:
        # TODO

        logger.info("Done!")

        # Stop execution
        return

    # Unfreeze model layers
    for param in model.parameters():
        param.requires_grad = True

    # Load data
    dataset_static_args = {
        "remove_instead_of_truncate": remove_instead_of_truncate,
        "batch_size": batch_size,
        "device": device,
        "dataset_workers": dataset_workers,
        "max_tokens": max_tokens,
        "tokenizer": tokenizer,
        "max_length_tokens": max_length_tokens,
        "monolingual": monolingual,
    }
    dataset_train, dataloader_train = load_dataset(filename_dataset_train, "train", **dataset_static_args)
    dataset_dev, _ = load_dataset(filename_dataset_dev, "dev", **dataset_static_args)
    dataset_test, _ = load_dataset(filename_dataset_test, "test", **dataset_static_args)

    #training_steps_per_epoch = len(dataloader_train) // num_processes
    training_steps_per_epoch = len(dataloader_train) # it counts batches, not samples!
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to --train-until-patience
    loss_function = nn.BCEWithLogitsLoss(reduction="mean") # Regression: raw input, not normalized
                                                           #  (i.e. sigmoid is applied in the loss function)

    # TODO verify that accelerate is handling correctly training_steps (e.g., linear LR scheduler with % of steps; use 10 epochs and verify that the first epoch the LR factor increases from 0 to 1)

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

    while not stop_training:
        logger.info("Epoch %d", epoch + 1)

        model.train()

        duplicated_data = {}

        for batch in dataloader_train:
            if max_tokens and batch is None:
                # Batch is under construction using max_tokens...
                continue

            #optimizer.zero_grad() # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            model.zero_grad()

            #inputs, targets = batch
            inputs = batch["url_tokens"]
            attention_mask = batch["url_attention_mask"]
            targets = batch["labels"]
            outputs = model(inputs, attention_mask).logits # .last_hidden_state
            outputs = outputs.squeeze(1) # (batch_size, 1) -> (batch_size,)
            loss = loss_function(outputs, targets)

            for i in inputs:
                _i = tokenizer.decode(i.squeeze().tolist(), skip_special_tokens=True)

                if _i not in duplicated_data:
                    duplicated_data[_i] = 0

                duplicated_data[_i] += 1

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_duplicated = 0

        for i, v in duplicated_data.items():
            total_duplicated += v - 1

        assert total_duplicated < batch_size, f"{total_duplicated} >= {batch_size}"
        assert abs(len(duplicated_data) - batch_size * training_steps_per_epoch // num_processes) < batch_size, f"abs({len(duplicated_data)} - {batch_size} * {training_steps_per_epoch} // {num_processes}) >= {batch_size}"

        epoch += 1

        # Stop training?
        if patience > 0 and current_patience >= patience:
            # End of patience

            stop_training = True
        elif not train_until_patience:
            stop_training = epoch >= epochs

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

    #parser.add_argument('--batch-size', type=int, default=16,
    #                    help="Batch size. Elements which will be processed before proceed to train, but the whole batch will "
    #                         "be processed in blocks in order to avoid OOM errors")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size. Elements which will be processed before proceed to train")
    #parser.add_argument('--block-size', type=int, help="Block size. Elements which will be provided to the model at once")
    parser.add_argument('--max-tokens', type=int, default=-1,
                        help="Process batches in groups tokens size (fairseq style). "
                             "Batch size is still relevant since the value is used when batches are needed (e.g. sampler from dataset)")
    parser.add_argument('--epochs', type=int, default=3, help="Epochs")
    parser.add_argument('--do-not-fine-tune', action="store_true", help="Do not apply fine-tuning to the base model (default weights)")
    parser.add_argument('--dataset-workers', type=int, default=-1,
                        help="No. workers when loading the data in the dataset. When negative, all available CPUs will be used")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', help="Model input path which will be loaded")
    parser.add_argument('--model-output', help="Model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true",
                        help="Do not train, just apply inference (flag --model-input is recommended). "
                             "If this option is set, it will not be necessary to provide the input dataset")
    parser.add_argument('--inference-from-stdin', action="store_true", help="Read inference from stdin")
    parser.add_argument('--patience', type=int, default=0, help="Patience before stopping the training")
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
    parser.add_argument('--remove-instead-of-truncate', action="store_true",
                        help="Remove pairs which would need to be truncated (if not enabled, truncation will be applied). "
                             "This option will be only applied to the training set")
    parser.add_argument('--monolingual', action="store_true",
                        help="Only the MT or HT will be processed instead of OT+MT|HT. This does not change the expected format in the data: OT+MT|HT+label")

    parser.add_argument('--seed', type=int, default=71213,
                        help="Seed in order to have deterministic results (not fully guaranteed). "
                             "Set a negative number in order to disable this feature")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

def cli():
    global logger
    global accelerator

    assert accelerator is None

    accelerator = accelerate.Accelerator()

    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    # Logging
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args))) # First logging message should be the processed arguments

    main(args)

if __name__ == "__main__":
    cli()
