
import os
import sys
import copy
import pickle
import logging
import argparse

import mtdetect.transformer_mm_explainability.example_translation_nllb as example_translation_nllb
import mtdetect.inference as inference
import mtdetect.utils.utils as utils
import mtdetect.dataset as dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers

logger = logging.getLogger("mtdetect.explainability_nllb_vision_model")

def main(args):
    train_filename = args.dataset_train_filename
    dev_filename = args.dataset_dev_filename
    test_filename = args.dataset_test_filename
    batch_size = args.batch_size
    epochs = args.epochs
    matrix_max_width = args.matrix_max_width
    matrix_max_height = args.matrix_max_height
    source_lang = args.source_lang
    target_lang = args.target_lang
    direction = args.direction
    attention_matrix = args.attention_matrix
    explainability_normalization = args.explainability_normalization
    self_attention_remove_diagonal = not args.self_attention_do_not_remove_diagonal
    cnn_pooling = args.cnn_pooling
    vision_model_input = args.model_input
    save_model_path = args.model_output
    learning_rate = args.learning_rate
    multichannel = not args.not_multichannel
    pretrained_model = args.pretrained_model
    teacher_forcing = not args.not_teacher_forcing
    ignore_attention = args.ignore_attention
    lm_pretrained_model = args.lm_pretrained_model
    lm_model_input = args.lm_model_input
    lm_frozen_params = args.lm_frozen_params
    lm_learning_rate = args.lm_learning_rate
    model_inference = args.inference
    gradient_accumulation = args.gradient_accumulation
    _max_length_encoder = args.max_length_tokens
    classifier_dropout = args.lm_classifier_dropout
    dropout = args.dropout
    threshold = args.threshold
    optimizer_str = args.optimizer
    optimizer_args = args.optimizer_args
    scheduler_str = args.lr_scheduler
    lr_scheduler_args = args.lr_scheduler_args
    train_until_patience = args.train_until_patience
    patience = args.patience
    multiplicative_inverse_temperature_sampling = args.multiplicative_inverse_temperature_sampling
    seed = args.seed
    patience_metric = args.dev_patience_metric
    disable_vision_model = args.disable_vision_model
    model_inference_skip_train = args.skip_training_set_during_inference
    attention_matrix_preprocess = args.attention_matrix_preprocess
    vision_model = args.vision_model
    vit_patches_dim = args.vit_patches_dim
    total_patches = None

    logger.info("Vision model: %s", vision_model)

    if vision_model == "vit":
        assert attention_matrix_preprocess == "resize", attention_matrix_preprocess
        assert matrix_max_width == matrix_max_height
        assert matrix_max_width % vit_patches_dim == 0, matrix_max_width % vit_patches_dim
        assert (matrix_max_width * matrix_max_height) % (vit_patches_dim * vit_patches_dim) == 0, "This should not happen..."

        total_patches = (matrix_max_width * matrix_max_height) // (vit_patches_dim * vit_patches_dim)

        logger.debug("ViT model: %d total patches (%d x %d matrices -> %d x %d patches)", total_patches, matrix_max_width, matrix_max_height, vit_patches_dim, vit_patches_dim)

    assert batch_size > 0
    assert matrix_max_width > 0
    assert matrix_max_height > 0
    assert isinstance(direction, list)
    assert isinstance(direction[0], str)
    assert gradient_accumulation > 0, gradient_accumulation

    teacher_forcing = [True if teacher_forcing else False] * len(direction)
    ignore_attention = [True if ignore_attention else False] * len(direction)

    if train_until_patience:
        assert patience > 0, "Infinite training loop"

    if scheduler_str in ("linear",) and train_until_patience:
        # Depending on the LR scheduler, the training might even stop at some point (e.g. linear LR scheduler will set the LR=0 if the run epochs is greater than the provided epochs)
        logger.warning("You set a LR scheduler ('%s' scheduler) which conflicts with --train-until-patince: you might want to check this out and change the configuration", scheduler_str)

    if seed >= 0:
        logger.debug("Deterministic values enabled (not fully-guaranteed): seed %d", seed)
    else:
        seed = np.random.randint(2 ** 32 - 1)

        logger.warning("Deterministic values disable (you set a negative seed): random seed %d", seed)

    utils.init_random_with_seed(seed)

    skip_test = False # Change manually
    actual_batch_size = batch_size * gradient_accumulation

    if gradient_accumulation > 1:
        logger.info("Gradient accumulation enabled (i.e., >1): %d (note that if disabled, the same results would be obtained if dropout is disabled for both vision and language model, train shuffle is disabled, and float precision errors are ignored)", gradient_accumulation)

    logger.info("Batch size: %d (actual batch size: %d)", batch_size, actual_batch_size)

    model_inference_skip_train = model_inference_skip_train and model_inference

    if model_inference:
        lm_frozen_params = True

        assert save_model_path == '', save_model_path

    if save_model_path != '':
        assert not os.path.isfile(save_model_path), f"File found in the profided path to save the model: {save_model_path}"

    if lm_pretrained_model:
        logger.info("LM is going to be used: %s (local file: %s)", lm_pretrained_model, lm_model_input)

    if lm_pretrained_model and not lm_model_input and lm_frozen_params and not model_inference:
        logger.warning("LM provided but it is not a fine-tuned model and its parameters are frozen: the format is src<sep>trg, and the output is the first output token, which might not be the expected behaviour for the model")

    def load_model(model_input, pretrained_model, device, classifier_dropout=0.0):
        local_model = bool(model_input)
        config = transformers.AutoConfig.from_pretrained(pretrained_model, num_labels=1, classifier_dropout=classifier_dropout)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
        tokenizer = utils.get_tokenizer(pretrained_model)

        if local_model:
            state_dict = torch.load(model_input, weights_only=True, map_location=device) # weights_only: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
                                                                                        # map_location: avoid creating a new process and using additional and useless memory

            model.load_state_dict(state_dict)

        model = model.to(device)

        return model, tokenizer

    if skip_test:
        logger.warning("Test set evaluation is disabled")

    for _attention_matrix in attention_matrix:
        assert _attention_matrix in ("encoder", "decoder", "cross", "alti_plus_encoder", "alti_plus_decoder", "alti_plus_cross", "alti_plus_total"), attention_matrix

    if "alti_plus" in attention_matrix:
        assert explainability_normalization == "none", "Only option supported"

    for _cnn_pooling in cnn_pooling:
        assert _cnn_pooling in ("max", "avg"), cnn_pooling

    for _direction in direction:
        assert _direction in ("src2trg", "trg2src"), _direction

    assert len(cnn_pooling) in (1, len(attention_matrix)), cnn_pooling
    assert explainability_normalization in ("none", "absolute", "relative"), explainability_normalization
    assert matrix_max_width > 0, matrix_max_width
    assert matrix_max_height > 0, matrix_max_height

    attention_matrix = [f"explainability_{_attention_matrix}" for _attention_matrix in attention_matrix]
    translation_model_conf = {
        "pretrained_model": pretrained_model,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "direction": direction,
        "attention_matrix": attention_matrix,
        "explainability_normalization": explainability_normalization,
        "self_attention_remove_diagonal": self_attention_remove_diagonal,
        "teacher_forcing": teacher_forcing,
        "ignore_attention": ignore_attention,
    }
    translation_tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)

    logger.debug("NLLB conf: %s", translation_model_conf)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shuffle_training = True and not model_inference
    lang_model, tokenizer = load_model(lm_model_input, lm_pretrained_model, None, classifier_dropout=classifier_dropout) if lm_pretrained_model else (None, None)

    def extend_tensor_with_zeros_and_truncate(t, max_width, max_height, device):
        assert len(t.shape) == 2

        result = torch.zeros((max_width, max_height)).to(device)
        result[:t.shape[0], :t.shape[1]] = t[:max_width, :max_height]

        return result

    def read(filename, direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
             focus=["explainability_cross"], pickle_template=None, teacher_forcing=False, ignore_attention=False):
        matrix_width = -np.inf
        matrix_height = -np.inf
        loaded_samples = 0
        source_text = []
        target_text = []
        labels = []
        groups = [] # If more than 1 entry belongs to the same group, they will be randomly selected dynamically
        groups_balanced = [] # Data will be sampled according to NLLB and mBART papers temperature sampling
        uniq_groups = set()
        uniq_groups_balanced = set()
        explainability_data = {}
        fd = open(filename)
        fn_pickle_array = None
        limit_data = np.inf if "MTDETECT_LIMIT_DATA" not in os.environ else int(os.environ["MTDETECT_LIMIT_DATA"])
        limit_data = np.inf if limit_data <= 0 else limit_data
        envvar_prefix = "MTDETECT_PICKLE_FN"
        envvar = envvar_prefix

        if pickle_template:
            envvar = f"{envvar_prefix}_{pickle_template}"

            if envvar not in os.environ:
                envvar = envvar_prefix

        if envvar in os.environ:
            fn_pickle_array = os.environ[envvar]

            if envvar == envvar_prefix:
                if pickle_template:
                    fn_pickle_array = fn_pickle_array.replace("{template}", pickle_template)

                fn_pickle_array = fn_pickle_array.replace("{direction}", direction)
                fn_pickle_array = fn_pickle_array.replace("{teacher_forcing}", "yes" if teacher_forcing else "no")
                fn_pickle_array = fn_pickle_array.replace("{ignore_attention}", "yes" if ignore_attention else "no")

        fn_pickle_array_exists = os.path.isfile(fn_pickle_array)

        if not fn_pickle_array or not fn_pickle_array_exists:
            if fn_pickle_array:
                logger.warning("WARNING: provided envvar pickle path, but we could not find it: %s", fn_pickle_array)

            teacher_forcing_str = "yes" if teacher_forcing else "no"
            ignore_attention_str = "yes" if ignore_attention else "no"
            fn_pickle_array = f"{filename}.{direction}.{source_lang}.{target_lang}.teacher_forcing_{teacher_forcing_str}.ignore_attention_{ignore_attention_str}.pickle"

        fn_pickle_array_exists = os.path.isfile(fn_pickle_array)

        assert fn_pickle_array_exists, f"Pickle file not found: {fn_pickle_array}"

        logger.info("Loading explainability arrays: %s", fn_pickle_array)

        with open(fn_pickle_array, "rb") as pickle_fd:
            pickle_data = pickle.load(pickle_fd)
            explainability_data = {k: v[:None if limit_data == np.inf else limit_data] for k, v in pickle_data.items()}

        for idx, l in enumerate(fd):
            l = l.rstrip("\r\n").split('\t')
            source = l[0]
            target = l[1]
            label = l[2]
            group = l[3] if len(l) > 3 else str(idx)
            group_data = group.split(':')
            group = group_data[0] # remove "optional" part of the group
            group_balanced = "none"

            if len(group_data) > 1:
                group_data_balanced = ':'.join(group_data[1:]) # get the "other" group to balance different datasets from the "optional" part of the group

                if '#' in group_data_balanced:
                    group_balanced = group_data_balanced.split('#')[-1]

            assert label in ('0', '1'), label # 0 is NMT; 1 is HT

            label = float(label)

            source_text.append(source)
            target_text.append(target)
            labels.append(label)
            groups.append(group)
            groups_balanced.append(group_balanced)
            uniq_groups.add(group)
            uniq_groups_balanced.add(group_balanced)
            r_data = {k: v[idx] for k, v in explainability_data.items()}

            ##### code from example_translation_nllb.py #####
            if self_attention_remove_diagonal:
                np.fill_diagonal(r_data["explainability_encoder"], sys.float_info.epsilon)
                np.fill_diagonal(r_data["explainability_decoder"], sys.float_info.epsilon)

            if explainability_normalization == "none":
                pass
            elif explainability_normalization == "absolute":
                for k in r_data.keys():
                    r_data[k] = (r_data[k] - r_data[k].min()) / (r_data[k].max() - r_data[k].min())
            elif explainability_normalization == "relative":
                # "Relative" normalization (easier to analize per translated token)
                for k in r_data.keys():
                    r_data[k] = np.array([(r_data[k][i] - r_data[k][i].min()) / (r_data[k][i].max() - r_data[k][i].min()) for i in range(len(r_data[k]))])

            ##### code from example_translation_nllb.py #####

            width = -np.inf
            height = -np.inf

            for k in r_data.keys():
                if k in focus:
                    width = max(width, r_data[k].shape[0])
                    height = max(height, r_data[k].shape[1])

            assert width != -np.inf
            assert height != -np.inf

            matrix_width = matrix_max_width if matrix_max_width > 0 else max(matrix_width, width)
            matrix_height = matrix_max_height if matrix_max_height > 0 else max(matrix_height, height)
            loaded_samples += 1

            if loaded_samples % 100 == 0:
                logger.info("%d samples loaded: %s", loaded_samples, filename)

                sys.stdout.flush()

            if idx + 1 >= limit_data:
                break

        fd.close()

        # Explainability arrays were loaded
        expected_len = min(len(source_text), limit_data)

        for k, v in explainability_data.items():
            assert expected_len == len(v), f"{k}: {expected_len} != {len(v)}"

        logger.info("Samples: %d (limit: %s); Groups: %d; Balanced groups: %d", len(groups), limit_data, len(uniq_groups), len(uniq_groups_balanced))

        # groups
        # groups_balanced
        assert len(groups) == len(groups_balanced)

        groups2groups_balanced = {}

        for group, group_balanced in zip(groups, groups_balanced):
            if group not in groups2groups_balanced:
                groups2groups_balanced[group] = set()

            groups2groups_balanced[group].add(group_balanced)

            assert len(groups2groups_balanced[group]) == 1, f"Group {group} was found in more than one balanced group: {groups2groups_balanced[group]}. " \
                                                            "There can be multiple samples for the same group, but they must be in the same balanced group"

        r = {
            "matrix_width": matrix_width,
            "matrix_height": matrix_height,
            "loaded_samples": loaded_samples,
            "source_text": source_text,
            "target_text": target_text,
            "labels": labels,
            "groups": groups,
            "groups_balanced": groups_balanced,
        }

        r.update(explainability_data)

        return r

    def get_data(explainability_matrix, labels, loaded_samples, matrix_width, matrix_height, device, convert_labels_to_tensor=True, attention_matrix_preprocess="crop_or_pad"):
        assert attention_matrix_preprocess in ("crop_or_pad", "resize"), attention_matrix_preprocess

        inputs = []
        original_inputs = []

        for _input in explainability_matrix:
            original_inputs.append(_input)

            _input = torch.from_numpy(_input)

            assert len(_input.shape) == 2

            if attention_matrix_preprocess == "crop_or_pad":
                _input = extend_tensor_with_zeros_and_truncate(_input, matrix_width, matrix_height, None)
            elif attention_matrix_preprocess == "resize":
                _input = _input.unsqueeze(0).unsqueeze(0) # batch_size and channels
                _input = F.interpolate(_input, size=(matrix_width, matrix_height), mode="bicubic", align_corners=False)
                _input = _input.squeeze(0).squeeze(0)

                assert _input.shape == (matrix_width, matrix_height), _input.shape
            else:
                raise Exception(f"Unknown attention_matrix_preprocess: {attention_matrix_preprocess}")
            _input = _input.tolist()

            inputs.append(_input)

        inputs = torch.tensor(inputs)
        inputs = inputs.unsqueeze(1).to(device) # channel dim

        if convert_labels_to_tensor:
            labels = torch.tensor(labels).to(device)

        inputs_expected_shape = (loaded_samples, 1, matrix_width, matrix_height)
        labels_expected_shape = (loaded_samples,)

        assert inputs.shape == inputs_expected_shape, inputs.shape
        assert labels.shape == labels_expected_shape, labels.shape

        return inputs, labels, original_inputs

    channels_factor_len_set = set([len(direction), len(teacher_forcing), len(ignore_attention)])

    assert len(channels_factor_len_set) in (1, 2), channels_factor_len_set

    if len(channels_factor_len_set) == 2:
        assert 1 in channels_factor_len_set, channels_factor_len_set

    if disable_vision_model:
        assert lang_model, "disable_vision_model does not support not providing lang_model"
        assert not vision_model_input

        logger.warning("Vision model disabled")

    if multichannel:
        channels = 1
        channels_factor = 1
        cnn_pooling *= 1 if len(cnn_pooling) > 1 else len(attention_matrix)
        cnn_pooling *= max(channels_factor_len_set)

        if lang_model:
            cnn_pooling.append(cnn_pooling[0]) # fake value (it will be ignored) -> easier for further processing
    else:
        # Expected: for each value provided to direction, teacher_forcing, and ignore_attention, we will have an extra set of len(attention_matrix) channels
        # Example: {direction: src2trg+trg2src, teacher_forcing: True+False, ignore_attention: False} -> [(src2trg, True, False), (trg2src, False, False)] # ignore_attention is expanded
        # Example: {direction: src2trg+src2trg+trg2src+trg2src, teacher_forcing: True+False+True+False, ignore_attention: False+True+False+True} -> [(src2trg, True, False), (src2trg, False, True), (trg2src, True, False), (trg2src, False, True)]
        channels = len(attention_matrix)
        channels_factor = max(channels_factor_len_set)

    direction *= 1 if len(direction) > 1 else max(channels_factor_len_set)
    teacher_forcing *= 1 if len(teacher_forcing) > 1 else max(channels_factor_len_set)
    ignore_attention *= 1 if len(ignore_attention) > 1 else max(channels_factor_len_set)
    channels *= channels_factor

    logger.debug("Total channels: %d (factor: %d)", channels, channels_factor)

    matrix_width = -np.inf
    matrix_height = -np.inf
    data_input_all_keys = []
    train_data = {}
    dev_data = {}
    test_data = {}
    temperature_sampling = 1 / multiplicative_inverse_temperature_sampling

    assert len(direction) == len(teacher_forcing) == len(ignore_attention)

    for _direction, _teacher_forcing, _ignore_attention in zip(direction, teacher_forcing, ignore_attention):
        # TODO we are reading the files several times...

        _train_data = read(train_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                           focus=attention_matrix, pickle_template="train", teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention)
        _dev_data = read(dev_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                         focus=attention_matrix, pickle_template="dev", teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention)

        if skip_test:
            _test_data = {"matrix_width": -np.inf, "matrix_height": -np.inf}
        else:
            _test_data = read(test_filename, _direction, source_lang, target_lang, self_attention_remove_diagonal, explainability_normalization,
                              focus=attention_matrix, pickle_template="test", teacher_forcing=_teacher_forcing, ignore_attention=_ignore_attention)

        train_data.update(_train_data)
        dev_data.update(_dev_data)
        test_data.update(_test_data)

        matrix_width = max(train_data["matrix_width"], dev_data["matrix_width"], test_data["matrix_width"], matrix_width)
        matrix_height = max(train_data["matrix_height"], dev_data["matrix_height"], test_data["matrix_height"], matrix_height)
        first_time = True
        _teacher_forcing_str = "yes" if _teacher_forcing else "no"
        _ignore_attention_str = "yes" if _ignore_attention else "no"

        for _attention_matrix in attention_matrix:
            inputs = f"{_attention_matrix}_{_direction}_{_teacher_forcing_str}_{_ignore_attention_str}"
            train_data[f"inputs_{inputs}"], train_data["labels"], train_data[f"original_data_{inputs}"] = \
                get_data(train_data[_attention_matrix], train_data["labels"], train_data["loaded_samples"], matrix_width, matrix_height, None, convert_labels_to_tensor=first_time, attention_matrix_preprocess=attention_matrix_preprocess)
            dev_data[f"inputs_{inputs}"], dev_data["labels"], dev_data[f"original_data_{inputs}"] = \
                get_data(dev_data[_attention_matrix], dev_data["labels"], dev_data["loaded_samples"], matrix_width, matrix_height, None, convert_labels_to_tensor=first_time, attention_matrix_preprocess=attention_matrix_preprocess)

            if not skip_test:
                test_data[f"inputs_{inputs}"], test_data["labels"], test_data[f"original_data_{inputs}"] = \
                    get_data(test_data[_attention_matrix], test_data["labels"], test_data["loaded_samples"], matrix_width, matrix_height, None, convert_labels_to_tensor=first_time, attention_matrix_preprocess=attention_matrix_preprocess)

            first_time = False

            assert inputs not in data_input_all_keys

            data_input_all_keys.append(inputs)

    translation_max_length_encoder = utils.get_encoder_max_length(lang_model, translation_tokenizer, max_length_tokens=_max_length_encoder)
    translation_max_length_encoder = min(translation_max_length_encoder, _max_length_encoder)
    checks = 0

    for d, desc in ((train_data, "train"), (dev_data, "dev"), (test_data, "test")):
        for idx, (source_text, target_text) in enumerate(zip(d["source_text"], d["target_text"])):
            # Sanity check to verify that the number of tokens of the source and target text match with the expected dimensions

            source_inputs = translation_tokenizer.encode_plus(source_text, return_tensors=None, add_special_tokens=True, max_length=translation_max_length_encoder,
                                                                return_attention_mask=False, truncation=True, padding="longest")["input_ids"]
            target_inputs = translation_tokenizer.encode_plus(target_text, return_tensors=None, add_special_tokens=True, max_length=translation_max_length_encoder,
                                                                return_attention_mask=False, truncation=True, padding="longest")["input_ids"]

            for _attention_matrix in attention_matrix:
                keys_filter_check_shape = list(filter(lambda s: s.startswith(f"original_data_{_attention_matrix}_"), d.keys()))

                assert len(keys_filter_check_shape) > 0

                for k in keys_filter_check_shape:
                    assert isinstance(d[k][idx], np.ndarray), f"{k}: {idx}: {type(d[k][idx])}"
                    assert len(d[k][idx].shape) == 2, d[k][idx].shape

                    if "_src2trg_" in k:
                        assert "_trg2src_" not in k
                        _direction = "src2trg"
                    elif "_trg2src_" in k:
                        assert "_src2src_" not in k
                        _direction = "trg2src"
                    else:
                        raise Exception(f"Unexpected direction: {k}")

                    if _direction == "src2trg":
                        _source_inputs, _target_inputs = source_inputs, target_inputs
                    elif _direction == "trg2src":
                        _source_inputs, _target_inputs = target_inputs, source_inputs
                    else:
                        raise Exception("?")

                    if _attention_matrix in ("explainability_encoder", "explainability_alti_plus_encoder"):
                        expected_shape = (len(_source_inputs), len(_source_inputs))
                    elif _attention_matrix in ("explainability_decoder", "explainability_alti_plus_decoder"):
                        expected_shape = (len(_target_inputs), len(_target_inputs))
                    elif _attention_matrix in ("explainability_cross", "explainability_alti_plus_cross"):
                        expected_shape = (len(_target_inputs), len(_source_inputs))
                    elif _attention_matrix in ("explainability_alti_plus_total",):
                        expected_shape = (len(_target_inputs), len(_target_inputs) + len(_source_inputs))
                    else:
                        raise Exception(f"Unexpected matrix key: {_attention_matrix}")

                    if d[k][idx].shape[0] <= translation_max_length_encoder and d[k][idx].shape[1] <= translation_max_length_encoder:
                        if d[k][idx].shape != expected_shape:
                            _model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
                            max_length = _model.config.max_length
                            max_new_tokens = _model.generation_config.max_length

                            del _model

                            if _attention_matrix in ("explainability_alti_plus_total",):
                                _expected_shape = (min(len(_target_inputs), expected_shape[0]), min(len(_target_inputs), expected_shape[1]) + min(len(_source_inputs), expected_shape[0]))
                            else:
                                _expected_shape = (min(max_length, expected_shape[0]), min(max_new_tokens, expected_shape[1]))

                            if d[k][idx].shape != _expected_shape:
                                logger.warning("Shape mismatch: %s: %s: %s: %s: %s vs %s (or %s): %s | %s", desc, idx, k, direction, d[k][idx].shape, expected_shape, _expected_shape, _source_inputs, _target_inputs)
                    else:
                        logger.warning("Different shape: %s: %s: %s != %s", k, direction, d[k][idx].shape, expected_shape)

                    checks += 1

        for _attention_matrix in attention_matrix:
            keys_filter_check_shape = list(filter(lambda s: s.startswith(f"original_data_{_attention_matrix}_"), d.keys()))

            assert len(keys_filter_check_shape) > 0

            for k in keys_filter_check_shape:
                del d[k]

    logger.debug("Dimension checks: %d", checks)

    if lang_model:
        # Add LM data to inputs

        max_length_encoder = utils.get_encoder_max_length(lang_model, tokenizer, max_length_tokens=_max_length_encoder)
        max_length_encoder = min(max_length_encoder, _max_length_encoder)
        data_input_all_keys.append("lm_inputs")

        logger.debug("Max length: %d", max_length_encoder)

        for d, desc in ((train_data, "train"), (dev_data, "dev"), (test_data, "test")):
            if desc == "test" and skip_test:
                continue

            assert "inputs_lm_inputs" not in d, desc

            d["inputs_lm_inputs"] = []
            all_texts = []

            for source_text, target_text in zip(d["source_text"], d["target_text"]):
                all_texts.append(f"{source_text}{tokenizer.sep_token}{target_text}")

            inputs = tokenizer.batch_encode_plus(all_texts, return_tensors=None, add_special_tokens=True, max_length=max_length_encoder,
                                                return_attention_mask=False, truncation=True, padding="longest")
            d["inputs_lm_inputs"] = inputs["input_ids"]
            inputs = torch.tensor(d["inputs_lm_inputs"])

            assert inputs.shape == (len(d["source_text"]), min(max_length_encoder, inputs.shape[1])), inputs.shape

            d["inputs_lm_inputs"] = inputs

    if not multichannel:
        if vision_model == "cnn":
            assert len(cnn_pooling) == 1, cnn_pooling

            cnn_pooling *= len(data_input_all_keys)

    len_data = len(data_input_all_keys)

    assert len(data_input_all_keys) == len_data, f"{data_input_all_keys} len is not {len_data}"
    assert len(cnn_pooling) == len_data, f"{cnn_pooling} len is not {len_data}"

    data_input_all_keys, cnn_pooling = \
        zip(*map(lambda s: s.split('|'), sorted([f"{d}|{c}" for d, c in zip(data_input_all_keys, cnn_pooling)]))) # We sort to get the same results when the order is different
    data_input_all_keys = list(data_input_all_keys)
    cnn_pooling = list(cnn_pooling)

    assert len(data_input_all_keys) == len_data, f"{data_input_all_keys} len is not {len_data}"
    assert len(cnn_pooling) == len_data, f"{cnn_pooling} len is not {len_data}"

    logger.debug("Matrices width and height: %d %d", matrix_width, matrix_height)
    logger.debug("All channels (keys): %s", ' '.join(data_input_all_keys))

    assert len(set([k[7:] for k in train_data.keys() if k.startswith("inputs_")]).intersection(set(data_input_all_keys))) == len(data_input_all_keys), f"{[k for k in train_data.keys() if k.startswith('inputs_')]} vs keys"
    assert len(set([k[7:] for k in dev_data.keys() if k.startswith("inputs_")]).intersection(set(data_input_all_keys))) == len(data_input_all_keys), f"{[k for k in dev_data.keys() if k.startswith('inputs_')]} vs keys"

    if not skip_test:
        assert len(set([k[7:] for k in test_data.keys() if k.startswith("inputs_")]).intersection(set(data_input_all_keys))) == len(data_input_all_keys), f"{[k for k in test_data.keys() if k.startswith('inputs_')]} vs keys"

    class MyDataset(Dataset):
        def __init__(self, data, all_keys, create_groups=False, return_group=False, add_text=False):
            self.create_groups = create_groups
            self.return_group = return_group
            self.data = {}
            self.uniq_groups = []
            self.groups_balanced_aligned_with_uniq_groups = []
            self.all_keys = all_keys
            self.add_text = add_text
            self.groups2group_balanced = {}

            if create_groups:
                self.groups = data["groups"]
                self.groups_balanced = data["groups_balanced"]
            else:
                self.groups = [str(idx) for idx in range(len(data["labels"]))]
                self.groups_balanced = ["none" for _ in range(len(data["labels"]))]

                if "groups" in data:
                    assert "groups_balanced" in data, data.keys()
                    assert len(data["groups"]) == len(self.groups)

                    _set_groups = set(data["groups"])

                    if len(_set_groups) < len(data["groups"]):
                        logger.warning("create_groups=False, but 'groups' was provided, and there are groups with >1 element: %d groups with >1 element", len(data['groups']) - len(_set_groups))

                if "groups_balanced" in data:
                    assert "groups" in data, data.keys()
                    assert len(data["groups_balanced"]) == len(self.groups_balanced)

                    _set_groups = set(data["groups_balanced"])

                    if len(_set_groups) > 1:
                        logger.warning("create_groups=False, but 'groups_balanced' was provided, and there are >1 balanced groups: %d balanced groups", len(_set_groups))

            assert len(self.groups) == len(self.groups_balanced)

            for i in range(len(self.all_keys) - 1):
                k1 = self.all_keys[i]
                k2 = self.all_keys[i + 1]

                assert len(data[f"inputs_{k1}"]) == len(data[f"inputs_{k2}"])
                assert len(data[f"inputs_{k1}"]) == len(data["labels"])

            assert len(data["labels"]) == len(self.groups)
            assert len(data["labels"]) == len(self.groups_balanced)

            for idx in range(len(self.groups)):
                group = self.groups[idx]
                group_balanced = self.groups_balanced[idx]

                assert ':' not in group
                assert '#' not in group_balanced

                if group_balanced not in self.groups_balanced_aligned_with_uniq_groups:
                    assert group not in self.data, f"This error might be caused by using the same group in different balanced groups, which is not supported: {group} | {group_balanced}"

                if group not in self.data:
                    self.uniq_groups.append(group)
                    self.groups_balanced_aligned_with_uniq_groups.append(group_balanced)

                    self.data[group] = {
                        'x': {k: [] for k in self.all_keys},
                        'y': [],
                        "group_balanced": group_balanced,
                    }

                    if self.add_text:
                        self.data[group]["source_text"] = []
                        self.data[group]["target_text"] = []

                    self.groups2group_balanced[group] = group_balanced

                assert self.groups2group_balanced[group] == group_balanced # this should be also true when group in self.data (i.e., did not enter the previous if statement)
                assert len(self.groups_balanced_aligned_with_uniq_groups) == len(self.uniq_groups) # these two should be aligned in terms of nubmer of elements

                for k in self.all_keys:
                    self.data[group]['x'][k].append(data[f"inputs_{k}"][idx])

                self.data[group]['y'].append(data["labels"][idx])

                if self.add_text:
                    self.data[group]["source_text"].append(data["source_text"][idx])
                    self.data[group]["target_text"].append(data["target_text"][idx])

            assert len(self.groups2group_balanced) == len(self.uniq_groups)
            assert len(set(self.groups2group_balanced.values())) == len(set(self.groups_balanced_aligned_with_uniq_groups))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            group = self.uniq_groups[idx]
            group_balanced = self.data[group]["group_balanced"]
            x = {k: [v.clone().detach().type(torch.float32) if k != "lm_inputs" else v.clone().detach() for v in l] for k, l in self.data[group]['x'].items()}
            y = [_y.clone().detach().type(torch.float32) for _y in self.data[group]['y']]

            if not self.create_groups:
                assert len(y) == 1

                for k in self.all_keys:
                    assert len(x[k]) == 1

            result = {"x": x, "y": y}

            if self.return_group:
                result["group"] = group
                result["group_balanced"] = group_balanced

            if self.add_text:
                result["source_text"] = self.data[group]["source_text"]
                result["target_text"] = self.data[group]["target_text"]

            return result

    def wrapper_select_random_group_collate_fn(tokenizer=None, remove_padding=True, return_text=False):
        remove_padding = False if tokenizer is None else remove_padding
        padding_id = None if tokenizer is None else tokenizer.pad_token_id

        def collate_fn(batch, remove_padding=True, padding_id=None, return_text=False):
            data, target = [], []
            source_text, target_text = [], []

            for idx in range(len(batch)):
                x = batch[idx]["x"]
                y = batch[idx]["y"]
                _source_text = batch[idx]["source_text"] if return_text else None
                _target_text = batch[idx]["target_text"] if return_text else None

                assert len(y) > 0

                for _x in x.values():
                    assert len(_x) == len(y)

                group_idx = np.random.randint(len(y))
                output_x = {k: v[group_idx] for k, v in x.items()}
                output_y = y[group_idx]

                data.append(output_x)
                target.append(output_y)

                if return_text:
                    assert len(_source_text) == len(y)
                    assert len(_target_text) == len(y)

                    source_text.append(_source_text[group_idx])
                    target_text.append(_target_text[group_idx])

            target = torch.stack(target, dim=0)
            data = {k: torch.stack([v[k] for v in data], dim=0) for k in data[0].keys()}

            if remove_padding and padding_id is not None and "lm_inputs" in data.keys():
                lm_len = data["lm_inputs"].shape[1]

                assert data["lm_inputs"].shape == (len(batch), lm_len), data["lm_inputs"].shape

                mask = ~(torch.all(data["lm_inputs"] == padding_id, dim=0))

                assert mask.shape == (lm_len,), mask.shape

                uc, uc_counts = torch.unique_consecutive(mask, return_counts=True)
                uc = uc.tolist()
                uc_counts = uc_counts.tolist()

                assert uc in ([True, False], [True]), mask

                data["lm_inputs"] = data["lm_inputs"][:, mask]

                assert data["lm_inputs"].shape == (len(batch), uc_counts[0]), data["lm_inputs"].shape

            if return_text:
                assert "source_text" not in data.keys()
                assert "target_text" not in data.keys()

                data["source_text"] = source_text
                data["target_text"] = target_text

            return data, target

        return lambda batch: collate_fn(batch, remove_padding=remove_padding, padding_id=padding_id, return_text=return_text)

    class VisionModel(nn.Module):
        def __init__(self, c, w, h, num_classes, all_keys, only_conv=True, lang_model=None, disable_vision_model=False, dropout_p=0.5, vision_model_str="cnn", **model_kwargs):
            super().__init__()

            if disable_vision_model:
                self.channels = 1
            else:
                self.channels = c

            self.only_conv = only_conv
            self.all_keys = list(all_keys)
            self.dimensions = (w, h)
            self.lang_model_hidden_size = 0
            self.disable_vision_model = disable_vision_model
            self.vision_model_str = vision_model_str

            assert self.vision_model_str in ("cnn", "vit"), self.vision_model_str

            if lang_model is not None:
                assert "lm_inputs" in self.all_keys

                self.lang_model_hidden_size = lang_model.config.hidden_size

            if "lm_inputs" in self.all_keys:
                assert lang_model is not None

            if lang_model:
                self.all_keys.remove("lm_inputs")

            if self.vision_model_str == "cnn":
                # First convolutional layer
                self.kernel_size = model_kwargs["kernel_size"]
                self.layer_size = model_kwargs["conv_layer_size"]
                self.conv_layers = model_kwargs["conv_layers"]
                self.in_channels = [self.channels, *[self.layer_size * (2 ** i) for i in range(self.conv_layers - 1)]]
                self.out_channels = self.in_channels[1:] + [self.layer_size * (2 ** len(self.in_channels[1:]))]
                self.vision_model = nn.ModuleList([nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.out_channels[i],
                                                             kernel_size=self.kernel_size, stride=1, padding=1,
                                                             padding_mode="zeros") for i in range(self.conv_layers)])

                assert len(self.in_channels) == self.conv_layers
                assert len(self.out_channels) == self.conv_layers
                assert len(self.vision_model) == self.conv_layers
            elif self.vision_model_str == "vit":
                assert self.dimensions[0] == self.dimensions[1]

                # It will be initialized again later to ensure that the weight initialization is handled correctly
                config = transformers.ViTConfig(num_hidden_layers=model_kwargs["vit_num_hidden_layers"], num_channels=self.channels, image_size=self.dimensions[0],
                                                patch_size=model_kwargs["vit_patches_dim"], num_attention_heads=model_kwargs["vit_num_attention_heads"])
                self.vision_model = transformers.ViTModel(config)
            else:
                raise Exception(f"Unknown vision model: {self.vision_model_str}")

            # Second convolutional layer
            if model_kwargs["pooling"] == "max":
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            elif model_kwargs["pooling"] == "avg":
                pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            else:
                raise Exception(f"Unexpected pooling: {model_kwargs['pooling']}")

            self.pool = pool

            # Calculate the size of the feature map after the convolutional layers and pooling
            if self.disable_vision_model:
                self._to_linear = 0
            else:
                self._to_linear = self.linear(torch.rand(1, self.channels, *self.dimensions)).numel()

            self._to_linear += self.lang_model_hidden_size

            # Fully connected layers
            self.hidden = args.fc_hidden_size
            self.fc1 = nn.Linear(self._to_linear, self.hidden)
            self.fc2 = nn.Linear(self.hidden, num_classes)

            self.dropout = nn.Dropout(p=dropout_p)

            self._initialize_weights()

            # Store lang model after weights initialization to avoid problems......
            self.lang_model = lang_model

            if self.vision_model_str == "vit":
                # ViT model handles properly weight initialization
                self.vision_model = transformers.ViTModel(config)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def linear(self, x):
            if self.vision_model_str == "cnn":
                for conv in self.vision_model:
                    x = self.pool(F.relu(conv(x)))
            elif self.vision_model_str == "vit":
                x = self.vision_model(x).last_hidden_state
                x = x[:,0,:] # CLS token
            else:
                raise Exception(f"Unknown vision model: {self.vision_model_str}")

            return x

        def forward(self, x):
            lm_input_ids = None

            if isinstance(x, dict):
                if self.lang_model:
                    lm_input_ids = x["lm_inputs"]

                x = torch.cat([x[k] for k in self.all_keys], dim=1)

                assert x.shape[1:] == (self.channels, *self.dimensions), x.shape

            x = self.linear(x)

            if self.only_conv:
                return x

            bs = x.shape[0]
            x = x.view(bs, self._to_linear - self.lang_model_hidden_size)

            if self.lang_model:
                lm_attention_mask = utils.get_attention_mask(tokenizer, lm_input_ids)
                output = self.lang_model(input_ids=lm_input_ids, attention_mask=lm_attention_mask, output_hidden_states=True)
                last_hidden_state = output["hidden_states"][-1]
                classifier_token = last_hidden_state[:,0,:]

                assert classifier_token.shape == (bs, self.lang_model_hidden_size)

                x = torch.cat([x, classifier_token], dim=1)

            assert x.shape == (bs, self._to_linear)

            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    class MultiChannelVisionModel(nn.Module):
        def __init__(self, num_classes, simple_cnns, all_keys, lang_model=None, disable_vision_model=False, dropout_p=0.5):
            super().__init__()

            self.all_keys = list(all_keys)
            self.lang_model_hidden_size = 0
            self.disable_vision_model = disable_vision_model

            if lang_model is not None:
                assert "lm_inputs" in self.all_keys

                self.lang_model_hidden_size = lang_model.config.hidden_size

            if "lm_inputs" in self.all_keys:
                assert lang_model is not None

            if lang_model:
                self.all_keys.remove("lm_inputs")

            for k, simple_cnn in simple_cnns.items():
                assert k in self.all_keys
                assert isinstance(simple_cnn, VisionModel), type(simple_cnn)

            assert len(self.all_keys) == len(simple_cnns)

            if self.disable_vision_model:
                self.simple_cnns = None
                self._to_linear = {k: 0 for k in simple_cnns.keys()}
            else:
                self.simple_cnns = nn.ModuleDict({k: v for k, v in simple_cnns.items()})
                self._to_linear = {k: simple_cnns[k]._to_linear for k in simple_cnns.keys()}

            self._to_linear_sum = sum([self._to_linear[k] for k in self._to_linear.keys()])
            self._to_linear_sum += self.lang_model_hidden_size

            # Fully connected layers
            self.hidden = args.fc_hidden_size
            self.fc1 = nn.Linear(self._to_linear_sum, self.hidden)
            self.fc2 = nn.Linear(self.hidden, num_classes)

            self.dropout = nn.Dropout(p=dropout_p)

            self._initialize_weights()

            # Store lang model after weights initialization to avoid problems......
            self.lang_model = lang_model

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            assert isinstance(x, dict), type(x)

            lm_input_ids = x["lm_inputs"] if self.lang_model else None

            if not self.disable_vision_model:
                x = [self.simple_cnns[k](x[k]) for k in self.all_keys]
                x = [_x.view(-1, self._to_linear[k]) for _x, k in zip(x, self.all_keys)]
                x = torch.cat(x, dim=1)
                bs = x.shape[0]

            if self.lang_model:
                lm_attention_mask = utils.get_attention_mask(tokenizer, lm_input_ids)

                assert lm_attention_mask.shape == lm_input_ids.shape

                output = self.lang_model(input_ids=lm_input_ids, attention_mask=lm_attention_mask, output_hidden_states=True)
                last_hidden_state = output["hidden_states"][-1]
                classifier_token = last_hidden_state[:,0,:]

                if not self.disable_vision_model:
                    assert classifier_token.shape == (bs, self.lang_model_hidden_size)

                    x = torch.cat([x, classifier_token], dim=1)
                else:
                    x = classifier_token
                    bs = x.shape[0]

            assert x.shape == (bs, self._to_linear_sum)

            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    def apply_inference(model, data, target=None, loss_function=None, threshold=0.5, loss_apply_sigmoid=False):
        model_outputs = model(data)
        outputs = model_outputs
        outputs = outputs.squeeze(1)
        loss = None

        assert len(outputs.shape) == 1, outputs.shape

        if loss_function is not None and target is not None:
            loss = loss_function(torch.sigmoid(outputs) if loss_apply_sigmoid else outputs, target)

        outputs_classification = torch.sigmoid(outputs).cpu().detach().tolist()
        outputs_classification = list(map(lambda n: int(n >= threshold), outputs_classification))

        results = {
            "outputs": outputs,
            "outputs_classification_detach_list": outputs_classification,
            "loss": loss,
        }

        return results

    def eval(model, dataloader, all_keys, device, print_result=False, print_desc='-', threshold=0.5):
        training = model.training
        training_lm = False if not model.lang_model else model.lang_model.training

        model.eval()

        if model.lang_model:
            model.lang_model.eval()

        all_outputs = []
        all_labels = []
        print_idx = 0

        for data, target in dataloader:
            _data = {k: data[k].to(device) for k in all_keys}
            results = apply_inference(model, _data, target=None, loss_function=None, threshold=threshold, loss_apply_sigmoid=False)
            outputs_classification = results["outputs_classification_detach_list"]
            outputs = results["outputs"]
            outputs = torch.sigmoid(outputs).cpu().detach().tolist()
            labels = target.cpu()
            labels = torch.round(labels).type(torch.long)

            all_outputs.extend(outputs_classification)
            all_labels.extend(labels.tolist())

            if print_result:
                assert len(data["source_text"]) == len(outputs)
                assert len(data["target_text"]) == len(outputs)
                assert len(labels) == len(outputs)
                assert len(outputs) == len(outputs_classification)

                for source_text, target_text, output, label, output_classification_aux in zip(data["source_text"], data["target_text"], outputs, labels, outputs_classification):
                    output_classification = int(output >= threshold)

                    assert output_classification in (0, 1), output_classification
                    assert output_classification == output_classification_aux
                    assert label in (0, 1), label

                    if output_classification == 1 and label == 1:
                        conf_mat_value = "tp"
                    elif output_classification == 1 and label == 0:
                        conf_mat_value = "fp"
                    elif output_classification == 0 and label == 1:
                        conf_mat_value = "fn"
                    elif output_classification == 0 and label == 0:
                        conf_mat_value = "tn"
                    else:
                        raise Exception(f"Unexpected values: {output_classification} vs {label}")

                    print(f"inference: {print_desc}\t{print_idx}\t{output}\tlabel={label}\t{conf_mat_value}\t{source_text}\t{target_text}")

                    print_idx += 1

        all_outputs = torch.as_tensor(all_outputs)
        all_labels = torch.as_tensor(all_labels)
        results = inference.get_metrics(all_outputs, all_labels)

        if training:
            model.train()

            if not training_lm and model.lang_model:
                # Previous .train() might have enabled the language model training...
                model.lang_model.eval()

        if training_lm:
            model.lang_model.train()

        return results

    # Load data
    num_workers = 0
    collate_fn = wrapper_select_random_group_collate_fn(tokenizer=tokenizer, remove_padding=True, return_text=model_inference)
    train_dataset = MyDataset(train_data, data_input_all_keys, create_groups=True, return_group=True, add_text=model_inference)
    train_sampler = dataset.GroupBalancedSampler(train_dataset, batch_size, temperature_sampling=temperature_sampling, shuffle=shuffle_training, desc="train", logger=logger)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, collate_fn=collate_fn)
    dev_dataset = MyDataset(dev_data, data_input_all_keys, add_text=model_inference)
    dev_sampler = dataset.GroupBalancedSampler(dev_dataset, batch_size, temperature_sampling=1, shuffle=False, desc="dev", logger=logger)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=num_workers, sampler=dev_sampler, collate_fn=collate_fn)

    # Model
    num_classes = 1
    #epochs = 100
    #patience = 20
    if vision_model == "cnn":
        model_kwargs = {
            "kernel_size": args.kernel_size,
            "conv_layer_size": args.conv_layer_size,
            "conv_layers": args.conv_layers,
        }
    elif vision_model == "vit":
        model_kwargs = {
            "vit_patches_dim": vit_patches_dim,
            "vit_num_hidden_layers": args.vit_num_hidden_layers,
            "vit_num_attention_heads": args.vit_num_attention_heads,
        }
    else:
        raise Exception(f"Unknown vision model: {vision_model}")

    logger.debug("Vision model kwargs: %s", model_kwargs)

    if multichannel:
        _data_input_all_keys = list(data_input_all_keys)
        _cnn_pooling = list(cnn_pooling)

        if lang_model:
            i = _data_input_all_keys.index("lm_inputs")

            del _cnn_pooling[i]
            del _data_input_all_keys[i]

        simple_cnns = {k: VisionModel(channels, matrix_width, matrix_height, num_classes, _data_input_all_keys, only_conv=True, disable_vision_model=False, dropout_p=dropout, vision_model_str=vision_model, **{**model_kwargs, **{"pooling": pooling}}) for k, pooling in zip(_data_input_all_keys, _cnn_pooling)}
        model = MultiChannelVisionModel(num_classes, simple_cnns, data_input_all_keys, lang_model=lang_model, disable_vision_model=disable_vision_model, dropout_p=dropout)
    else:
        model = VisionModel(channels, matrix_width, matrix_height, num_classes, data_input_all_keys, only_conv=False, lang_model=lang_model, disable_vision_model=disable_vision_model, dropout_p=dropout, vision_model_str=vision_model, **{**model_kwargs, **{"pooling": cnn_pooling[0]}})

    if vision_model_input:
        logger.info("Loading vision model (and optionally LM): %s", vision_model_input)

        assert not disable_vision_model

        old_model_state_dict_keys = set(model.state_dict().keys())
        vision_model_state_dict = torch.load(vision_model_input, weights_only=True, map_location=device)

        if lang_model and model.state_dict()["fc1.weight"].shape[1] - vision_model_state_dict["fc1.weight"].shape[1] == lang_model.config.hidden_size:
            # Our model has a longer feed-forward layer because of lang_model
            logger.warning("Fixing shape of layers...")

            fc1_weight = copy.deepcopy(vision_model_state_dict["fc1.weight"])
            fc1_bias = copy.deepcopy(vision_model_state_dict["fc1.bias"])

            assert model.hidden == fc1_weight.shape[0] == fc1_bias.shape[0]

            vision_model_state_dict["fc1.weight"] = nn.Parameter(torch.rand(model.hidden, model.state_dict()["fc1.weight"].shape[1]), requires_grad=False)

            nn.init.xavier_normal_(vision_model_state_dict["fc1.weight"])

            vision_model_state_dict["fc1.bias"] = nn.Parameter(fc1_bias, requires_grad=True)
            vision_model_state_dict["fc1.weight"][:,:fc1_weight.shape[1]] = fc1_weight.clone().detach()

            vision_model_state_dict["fc1.weight"].requires_grad_(not model_inference)

        new_model_state_dict_keys = set(vision_model_state_dict.keys())

        model_state_dict_keys_intersection = set.intersection(old_model_state_dict_keys, new_model_state_dict_keys)
        model_state_dict_keys_new_old_diff = set.difference(new_model_state_dict_keys, old_model_state_dict_keys)
        model_state_dict_keys_old_new_diff = set.difference(old_model_state_dict_keys, new_model_state_dict_keys)

        logger.debug("Vision model keys (old: %d; new: %d; intersection: %d): new - old: %s: old - new: %s",
                     len(old_model_state_dict_keys), len(new_model_state_dict_keys), len(model_state_dict_keys_intersection), model_state_dict_keys_new_old_diff, model_state_dict_keys_old_new_diff)

        model.load_state_dict(vision_model_state_dict, strict=False)

    model = model.to(device)

    if model_inference:
        model.eval()
    else:
        model.train()

    for p in model.parameters():
        p.requires_grad_(not model_inference)

    if lang_model:
        if lm_frozen_params:
            lang_model.eval()
        else:
            lang_model.train()

        for p in lang_model.parameters():
            p.requires_grad_(not lm_frozen_params)

    training_steps_per_epoch = len(train_dataloader) # it counts batches, not samples (value = samples // batch_size)!
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to --train-until-patience

    logger.info("Steps per epoch: %d (total for %d epochs: %d)", training_steps_per_epoch, epochs, training_steps)

    if not model_inference:
        loss_function = nn.BCELoss(reduction="none") # BCELoss vs BCEWithLogitsLoss: check https://github.com/pytorch/pytorch/issues/49844
        loss_apply_sigmoid = True # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

        logger.debug("Optimizer args: %s", optimizer_args)

        lm_model_parameters = list(filter(lambda p: p.requires_grad, lang_model.parameters())) if lang_model else []
        model_parameters_data = list(filter(lambda d: d[1].requires_grad, [(k, p) for k, p in model.named_parameters() if not k.startswith("lang_model.")]))
        model_parameters = [d[1] for d in model_parameters_data]
        model_parameters_names = [d[0] for d in model_parameters_data]
        optimizer_args_params = [{"params": model_parameters, "lr": learning_rate}]

        assert len(model_parameters_data) == len(model_parameters) == len(model_parameters_names)

        if lm_model_parameters:
            optimizer_args_params.append({"params": lm_model_parameters, "lr": lm_learning_rate})

        logger.info("Parameters with requires_grad=True: %d (LM: %d)", len(model_parameters), len(lm_model_parameters))
        #print(f"Vision model parameters with requires_grad=True: {' '.join(model_parameters_names)}")

        optimizer, scheduler = \
            utils.get_lr_scheduler_and_optimizer_using_argparse_values(optimizer_str, scheduler_str, optimizer_args, lr_scheduler_args, optimizer_args_params, learning_rate, training_steps, training_steps_per_epoch, logger)
        early_stopping_best_result_dev = -np.inf # accuracy
        early_stopping_best_result_train = -np.inf # accuracy
        early_stopping_best_loss = np.inf
        current_patience = 0
        epoch_loss = []
        sum_epoch_loss = np.inf

    epoch = 0

    if model_inference:
        logger.info("Inference!")

        stop_training = True
    else:
        logger.info("Training!")

        stop_training = not train_until_patience and epoch >= epochs

    sys.stdout.flush()

    assert patience_metric in ("acc", "macro_f1"), f"{patience_metric} not supported"

    while not stop_training:
        logger.info("Epoch %d", epoch)

        train_results = eval(model, train_dataloader, data_input_all_keys, device, threshold=threshold)
        dev_results = eval(model, dev_dataloader, data_input_all_keys, device, threshold=threshold)
        better_loss_result = False

        if len(epoch_loss) > 0 and sum_epoch_loss < early_stopping_best_loss:
            logger.info("Better loss result: %s -> %s", early_stopping_best_loss, sum_epoch_loss)

            better_loss_result = True
            early_stopping_best_loss = sum_epoch_loss

        logger.info("Train eval: %s", train_results)
        logger.info("Dev eval: %s", dev_results)

        epoch_loss = []

        assert patience_metric in dev_results, patience_metric

        early_stopping_metric_train = train_results[patience_metric]
        early_stopping_metric_dev = dev_results[patience_metric]
        better_train_result = False
        patience_dev_equal = np.isclose(early_stopping_metric_dev, early_stopping_best_result_dev)
        patience_train_equal = np.isclose(early_stopping_metric_train, early_stopping_best_result_train)

        if early_stopping_metric_train > early_stopping_best_result_train:
            logger.info("Better train result (metric: %s): %s -> %s", patience_metric, early_stopping_best_result_train, early_stopping_metric_train)

            better_train_result = True
            early_stopping_best_result_train = early_stopping_metric_train

        if early_stopping_metric_dev > early_stopping_best_result_dev or ((patience_dev_equal and better_train_result) or (patience_dev_equal and patience_train_equal and better_loss_result)):
            logger.info("Patience better dev result (metric: %s): %s -> %s", patience_metric, early_stopping_best_result_dev, early_stopping_metric_dev)

            current_patience = 0
            early_stopping_best_result_dev = early_stopping_metric_dev

            if save_model_path:
                logger.info("Saving best model: %s", save_model_path)

                torch.save(model.state_dict(), save_model_path)
        elif patience > 0:
            current_patience += 1

            logger.info("Exhausting patience... %d / %d", current_patience, patience)

        if patience > 0 and current_patience >= patience:
            logger.info("Patience is over ...")

            stop_training = True

            break # we need to force the break to avoid the training of the current epoch

        model.zero_grad()
        final_loss = None
        loss_elements = 0

        for batch_idx, (data, target) in enumerate(train_dataloader, 1):
            data = {k: data[k].to(device) for k in data_input_all_keys}
            target = target.to(device)

            result = apply_inference(model, data, target=target, loss_function=loss_function, loss_apply_sigmoid=loss_apply_sigmoid, threshold=threshold)
            _loss = result["loss"]

            assert len(_loss.shape) == 1, _loss.shape

            loss_elements += _loss.numel()

            if final_loss is None:
                final_loss = torch.sum(_loss)
            else:
                final_loss += torch.sum(_loss)

            if batch_idx % gradient_accumulation == 0 or batch_idx == training_steps_per_epoch:
                assert final_loss is not None

                loss = final_loss / loss_elements
                final_loss = None
                loss_elements = 0

                epoch_loss.append(loss.cpu().detach().item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                model.zero_grad()

            if batch_idx % (100 * gradient_accumulation) == 0:
                sum_partial_loss = sum(epoch_loss[-100:])

                logger.info(f"Epoch loss (sum last 100 steps): step %d: %s", batch_idx, sum_partial_loss)

                sys.stdout.flush()

        assert batch_idx == training_steps_per_epoch, f"{batch_idx} vs {training_steps_per_epoch}"

        sum_epoch_loss = sum(epoch_loss)

        logger.info("Epoch loss: %s", sum_epoch_loss)

        assert str(sum_epoch_loss) != "nan", "Some values in the input data are NaN"

        sys.stdout.flush()

        epoch += 1

        stop_training = not train_until_patience and epoch >= epochs

    assert stop_training

    if save_model_path and not model_inference:
        logger.info("Loading best model: %s", save_model_path)

        model_state_dict = torch.load(save_model_path, weights_only=True, map_location=device)

        assert model.state_dict().keys() == model_state_dict.keys()

        model.load_state_dict(model_state_dict)

        model = model.to(device)

    if not model_inference_skip_train:
        train_results = eval(model, train_dataloader, data_input_all_keys, device, print_result=model_inference, print_desc="train", threshold=threshold)

        logger.info("Final train eval: %s", train_results)
    else:
        logger.info("Final train eval: skip (inference)")

    del train_dataset
    del train_dataloader

    torch.cuda.empty_cache()

    dev_results = eval(model, dev_dataloader, data_input_all_keys, device, print_result=model_inference, print_desc="dev", threshold=threshold)

    logger.info("Final dev eval: %s", dev_results)

    if not skip_test:
        del dev_dataset
        del dev_dataloader

        torch.cuda.empty_cache()

        test_dataset = MyDataset(test_data, data_input_all_keys, add_text=model_inference)
        test_sampler = dataset.GroupBalancedSampler(test_dataset, batch_size, temperature_sampling=1, shuffle=False, desc="test", logger=logger)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler, collate_fn=collate_fn)

        test_results = eval(model, test_dataloader, data_input_all_keys, device, print_result=model_inference, print_desc="test", threshold=threshold)

        logger.info("Final test eval: %s", test_results)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="MTDetect using explainability matrices and vision classifier (+LM)")

    lr_scheduler_conf = utils.get_options_from_argv("--lr-scheduler", "inverse_sqrt_chichirau_et_al", utils.argparse_pytorch_conf.lr_scheduler_args)
    optimizer_conf = utils.get_options_from_argv("--optimizer", "adamw_no_wd", utils.argparse_pytorch_conf.optimizer_args)

    # Mandatory
    parser.add_argument('dataset_train_filename', type=str, help="Filename with train data (TSV format). Format: original text (OT), machine (MT) or human (HT) translation, 0 if MT or 1 if HT")
    parser.add_argument('dataset_dev_filename', type=str, help="Filename with dev data (TSV format)")
    parser.add_argument('dataset_test_filename', type=str, help="Filename with test data (TSV format)")

    # Optional params
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size. Elements which will be processed before proceed to train")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs")
    parser.add_argument('--pretrained-model', default="facebook/nllb-200-distilled-600M", help="Pretrained translation model to calculate --attention-matrix")
    parser.add_argument('--lm-pretrained-model', help="Pretrained language model (encoder-like) to train together with our vision model classifier") # default empty: means NO lm
    parser.add_argument('--max-length-tokens', type=int, default=512, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', type=str, default='', help="Vision model input path which will be loaded")
    parser.add_argument('--model-output', type=str, default='', help="Vision model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true", help="Do not train, just apply inference to the train, dev and test files")
    parser.add_argument('--patience', type=int, default=6,
                        help="Patience to stop training. If the specified value is greater than 0, epochs and patience will be taken into account")
    parser.add_argument('--train-until-patience', action="store_true",
                        help="Train until patience value is reached (--epochs will be ignored in order to stop, but will still be "
                             "used for other actions like LR scheduler)")
    parser.add_argument('--lm-model-input', help="Encoder-like model input path where the model will be stored")
    parser.add_argument('--learning-rate', type=float, default=5e-3, help="Vision model learning rate")
    parser.add_argument('--lm-frozen-params', action='store_true', help="Freeze encoder-like model parameters (i.e., do not train)")
    parser.add_argument('--lm-learning-rate', type=float, default=1e-5, help="Encoder-like model learning rate")
    parser.add_argument('--matrix-max-width', type=int, default=64, help="Max. width for the attention matrices")
    parser.add_argument('--matrix-max-height', type=int, default=64, help="Max. height for the attention matrices")
    parser.add_argument('--source-lang', type=str, required=True, help="NLLB source language (e.g., eng_Latn)")
    parser.add_argument('--target-lang', type=str, required=True, help="NLLB target language")
    parser.add_argument('--direction', type=str, nargs='+', choices=["src2trg", "trg2src"], default=["src2trg"], help="Translation direction. Providing several values is supported")
    parser.add_argument('--attention-matrix', type=str, nargs='+', choices=["encoder", "decoder", "cross", "alti_plus_encoder", "alti_plus_decoder", "alti_plus_cross", "alti_plus_total"], default=["encoder", "decoder", "cross"], help="Explainability matrixes provided to the vision classifier. Providing several values is supported")
    parser.add_argument('--explainability-normalization', type=str, choices=["none", "absolute", "relative"], default="none", help="Normalization applied to --attention-matrix")
    parser.add_argument('--self-attention-do-not-remove-diagonal', action="store_true", help="Do not set 0s to the explainability self-attention matrices")
    parser.add_argument('--cnn-pooling', type=str, nargs='+', choices=["avg", "max"], default=["max"], help="CNN pooling. Providing several values is supported")
    parser.add_argument('--not-multichannel', action="store_true", help="Do not train vision model in a multichannel setting (i.e., process --attention-matrix together instead of independently)")
    parser.add_argument('--not-teacher-forcing', action="store_true", help="Do not apply teacher forcing when calculating --attention-matrix (i.e., 'free' translation)")
    parser.add_argument('--ignore-attention', action="store_true", help="Ignore attention matrices when calculating --attention-matrix")
    parser.add_argument('--optimizer', choices=optimizer_conf["choices"], default=optimizer_conf["default"], help="Optimizer")
    parser.add_argument('--optimizer-args', **optimizer_conf["options"],
                        help="Args. for the optimizer (in order to see the specific configuration for a optimizer, use -h and set --optimizer)")
    parser.add_argument('--lr-scheduler', choices=lr_scheduler_conf["choices"], default=lr_scheduler_conf["default"], help="LR scheduler")
    parser.add_argument('--lr-scheduler-args', **lr_scheduler_conf["options"],
                        help="Args. for LR scheduler (in order to see the specific configuration for a LR scheduler, "
                             "use -h and set --lr-scheduler)")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--multiplicative-inverse-temperature-sampling', type=float, default=0.3, help="See https://arxiv.org/pdf/1907.05019 (section 4.2). Default value has been set the one used in the NLLB paper")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout applied to the vision (+LM) model")
    parser.add_argument('--lm-classifier-dropout', type=float, default=0.1, help="Dropout applied to the classifier layer of the encoder-like model")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold to consider a given text to be HT")
    parser.add_argument('--dev-patience-metric', type=str, choices=["acc", "macro_f1"], default="acc", help="Metric to calculate patience using the dev set")
    parser.add_argument('--disable-vision-model', action="store_true", help="Do not train vision model classifier. Debug purposes")
    parser.add_argument('--skip-training-set-during-inference', action="store_true", help="Skip training evaluation during inference to speed up result")
    parser.add_argument('--fc-hidden-size', type=int, default=128, help="Vision model fully connected layer hidden size")
    parser.add_argument('--conv-layers', type=int, default=2, help="Number of convolutional layers")
    parser.add_argument('--conv-layer-size', type=int, default=32, help="Convolutional layer size")
    parser.add_argument('--kernel-size', type=int, default=3, help="Convolutional layer kernel size")
    parser.add_argument('--vision-model', type=str, default="cnn", choices=["cnn", "vit"])
    parser.add_argument('--attention-matrix-preprocess', type=str, default="crop_or_pad", choices=["crop_or_pad", "resize"])
    parser.add_argument('--vit-patches-dim', type=int, default=16)
    parser.add_argument('--vit-num-hidden-layers', type=int, default=1)
    parser.add_argument('--vit-num-attention-heads', type=int, default=12)

    parser.add_argument('--seed', type=int, default=71213,
                        help="Seed in order to have deterministic results (not fully guaranteed). "
                             "Set a negative number in order to disable this feature")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

def cli():
    global logger

    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    # Logging
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: %s", str(args)) # First logging message should be the processed arguments

    main(args)

if __name__ == "__main__":
    cli()
