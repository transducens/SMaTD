
import os
import copy
import psutil
import logging
import gzip
import lzma
from contextlib import contextmanager
import argparse
import warnings

import random
import torch
import transformers
import numpy as np

logger = logging.getLogger("mtdetect")

def wc_l(fd, do_not_count_empty=True):
    no_lines = 0
    tell = fd.tell()

    for line in fd:
        if do_not_count_empty and line.strip() == '':
            continue

        no_lines += 1

    # Restore initial status of the fd
    fd.seek(tell)

    return no_lines

def get_layer_from_model(layer, name=None, deepcopy=True):
    could_get_layer = False

    # Get layer from model (we need to do it with a for loop since it is a generator which cannot be accessed with idx)
    for last_layer_name, last_layer_param in layer.named_parameters():
        if last_layer_name == name:
            could_get_layer = True

            break

    if name is not None:
        assert could_get_layer, f"Could not get the layer '{name}'"

    last_layer_param_data = last_layer_param.data

    if deepcopy:
        # Return a deepcopy instead of the value itself to avoid affect the model if modified

        return copy.deepcopy(last_layer_param_data)

    return last_layer_param_data

def encode(tokenizer, text, max_length=512, add_special_tokens=True, padding="do_not_pad", return_attention_mask=False,
           return_tensors="pt", truncation=True):
    encoder = tokenizer.batch_encode_plus if isinstance(text, list) else tokenizer.encode_plus

    return encoder(text, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding,
                   return_attention_mask=return_attention_mask, return_tensors=return_tensors, max_length=max_length)

def apply_model(model, tokenizer, tokens, encode=False):
    if encode:
        tokens = encode(tokenizer, tokens)

        output = model(**tokens)
    else:
        output = model(tokens)

    #input_ids = tokenized["input_ids"]
    #token_type_ids = tokenized["token_type_ids"]
    #attention_mask = tokenized["attention_mask"]

    #sentence_length = torch.count_nonzero(attention_mask).to("cpu").numpy()
    #tokens = input_ids[0,:sentence_length]

    return output

def get_current_allocated_memory_size():
    process = psutil.Process(os.getpid())
    size_in_bytes = process.memory_info().rss

    return size_in_bytes

def set_up_logging_logger(logger, filename=None, level=logging.INFO, lformat=None, display_when_file=False, accelerator=None):
    handlers = [
        logging.StreamHandler()
    ]

    if lformat is None:
        lformat = "[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s:%(lineno)d]"

        if accelerator is not None:
            lformat += f" [accelerator:{accelerator.process_index}]"

        lformat += " %(message)s"

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    formatter = logging.Formatter(lformat)

    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    logger.setLevel(level)

    logger.propagate = False # We don't want to see the messages multiple times

    return logger

def append_from_tuple(*tuples):
    """We expect tuples where:
        - First element: list
        - Second element: value to be inserted in the list of the first component of the tuple
    """
    for l, v in tuples:
        l.append(v)

@contextmanager
def open_xz_or_gzip_or_plain(file_path, mode='rt'):
    f = None
    try:
        if file_path[-3:] == ".gz":
            f = gzip.open(file_path, mode)
        elif file_path[-3:] == ".xz":
            f = lzma.open(file_path, mode)
        else:
            f = open(file_path, mode)
        yield f

    except Exception:
        raise Exception("Error occurred while loading a file!")

    finally:
        if f:
            f.close()

def resolve_path(p):
    result = os.path.realpath(os.path.expanduser(p)) if isinstance(p, str) else p

    return result.rstrip('/') if result else result

def exists(p, res_path=False, f=os.path.isfile):
    return f(resolve_path(p) if res_path else p) if isinstance(p, str) else False

def set_up_logging(filename=None, level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s",
                   display_when_file=False):
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    logging.basicConfig(handlers=handlers, level=level,
                        format=format)

def update_defined_variables_from_dict(d, provided_locals, smash=False):
    for v, _ in d.items():
        if v in provided_locals and not smash:
            raise Exception(f"Variable '{v}' is already defined and smash=False")

    provided_locals.update(d)

def argparse_nargs_type(*types):
    def f(arg):
        t = types[f._invoked]
        choices = None

        if isinstance(t, dict):
            choices = t["choices"]
            t = t["type"]

        if not isinstance(arg, t):
            type_arg = type(arg)

            try:
                arg = t(arg) # Cast from str
            except:
                raise argparse.ArgumentTypeError(f"Arg. #{f._invoked + 1} is not instance of {str(t)}, but {str(type_arg)}")
        elif choices is not None:
            if arg not in choices:
                raise argparse.ArgumentTypeError(f"Arg. #{f._invoked + 1} invalid value: value not in {str(choices)}")

        f._invoked += 1
        return arg

    f._invoked = 0

    return f

def get_tuple_if_is_not_tuple(obj, check_not_list=True):
    if not isinstance(obj, tuple):
        if check_not_list:
            if not isinstance(obj, list):
                return (obj,)
            else:
                return tuple(obj)
        else:
            return (obj,)

    return obj

def get_pytorch_version():
    try:
        torch_version = list(map(int, torch.__version__.split('+')[0].split('.')))
    except Exception as e:
        logger.error("%s", str(e))
        logger.error("Unexpected exception: returning -1.-1.-1 as torch version")

        return -1, -1, -1

    assert len(torch_version) == 3, f"Torch version is expected to be X.Y.Z, but got {'.'.join(torch_version)}"

    torch_version_major, torch_version_minor, torch_version_patch = torch_version

    return torch_version_major, torch_version_minor, torch_version_patch

def use_cuda(force_cpu=False):
    use_cuda = torch.cuda.is_available()

    return True if use_cuda and not force_cpu else False

def get_encoder_max_length(model, tokenizer, max_length_tokens=0, pretrained_model='', logger=None):
    if max_length_tokens <= 0:
        max_length_tokens = tokenizer.model_max_length
    if max_length_tokens > tokenizer.model_max_length:
        if logger:
            logger.warning("%s can handle a max. of %d tokens at once but you set %d: changing value to %d", pretrained_model, tokenizer.model_max_length, max_length_tokens, tokenizer.model_max_length)

        max_length_tokens = tokenizer.model_max_length

    max_length_tokens_max_value = 1000000000000000019884624838656 # https://discuss.huggingface.co/t/tokenizers-what-this-max-length-number/28484

    if max_length_tokens == max_length_tokens_max_value:
        alt1 = getattr(model.config, "max_length", -1)
        alt2 = getattr(model.config, "max_position_embeddings", -1)

        if alt1 > 0 or alt2 > 2:
            if alt1 > 0 and alt2 > 0:
                if alt1 == alt2:
                    max_length_tokens = alt1
                else:
                    max_length_tokens = min(alt1, alt2)

                    if logger:
                        logger.warning("Max tokens length (%d) has two alternatives, but they are different (%d vs %d): %d", max_length_tokens_max_value, alt1, alt2, max_length_tokens)
            elif alt1 > 0:
                max_length_tokens = alt1
            elif alt2 > 0:
                max_length_tokens = alt2
            else:
                raise Exception("Unexpected...")

    return max_length_tokens

def get_attention_mask(tokenizer, input_ids):
    return torch.ones_like(input_ids) * (input_ids != tokenizer.pad_token_id)

def get_tokenizer(pretrained_model, logger=None):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

        for warning in w:
            if "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option" in str(warning.message):
                if logger:
                    logger.warning("Loading slow tokenizer")

                tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)
            else:
                if logger:
                    logger.warning("Tokenizer warning: %s", str(warning.message))

    return tokenizer

def init_random_with_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_rng():
    result = {
        "rng_random": random.getstate(),
        "rng_np": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }

    try:
        result["rng_torch_cuda"] = torch.cuda.get_rng_state()
    except:
        result["rng_torch_cuda"] = None

    return result

def set_rng(rng_random, rng_np, rng_torch, rng_torch_cuda=None):
    random.setstate(rng_random)
    np.random.set_state(rng_np)
    torch.set_rng_state(rng_torch)

    if rng_torch_cuda is not None:
        torch.cuda.set_rng_state(rng_torch_cuda)

def chain_collate_fn(*list_of_collate_fn):
    def chained_collate_fn(batch):
        for fn in list_of_collate_fn:
            batch = fn(batch) # Pass the output of the previous collate_fn to the next one

        return batch

    return chained_collate_fn
