
import os
import logging
import multiprocessing

import mtdetect.utils.utils as utils

import torch
from torch.utils.data import (
    Sampler,
    Dataset,
    DataLoader,
)
import numpy as np
import more_itertools
import transformers

logger = logging.getLogger("mtdetect")

# TODO adapt

def remove_padding(sequence_batch, pad_token_id):
    _sequence_batch = sequence_batch.tolist()

    for idx, sequence in enumerate(_sequence_batch):
        try:
            padding_token_idx = sequence.index(pad_token_id)

            if padding_token_idx > 0:
                _sequence_batch[idx] = _sequence_batch[idx][:padding_token_idx - 1]
        except ValueError:
            pass # Hasn't been padded

    return _sequence_batch

def pad_sequence(sequence_batch, pad_token_id, max_length=0):
    max_batch_len = max(len(sequence) for sequence in sequence_batch)
    max_len = min(max_batch_len, max_length) if max_length > 0 else max_batch_len
    padded_sequences, attention_masks = [], []
    attend, no_attend = 1, 0

    for sequence in sequence_batch:
        # Truncate if exceeds max_len
        new_sequence = list(sequence[:max_len])

        attention_mask = [attend] * len(new_sequence)
        pad_length = max_len - len(new_sequence)

        new_sequence.extend([pad_token_id] * pad_length)
        attention_mask.extend([no_attend] * pad_length)

        padded_sequences.append(new_sequence)
        attention_masks.append(attention_mask)

    padded_sequences = torch.tensor(padded_sequences)
    attention_masks = torch.tensor(attention_masks)

    return padded_sequences, attention_masks

class SmartBatchingURLsDataset(Dataset):
    # Code based on https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies?scriptVersionId=67176227&cellId=2

    def __init__(self, input_data, output_data, tokenizer, max_length, regression=False, sampler_better_randomness=True,
                 remove_instead_of_truncate=False, set_desc='', imbalanced_strategy='', tasks_data={}):
        super(SmartBatchingURLsDataset, self).__init__()

        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.regression = regression
        self.sampler_better_randomness = sampler_better_randomness
        self.dataloader = None
        self.set_desc = set_desc
        self.labels = {
            "urls_classification": np.array(output_data)
        }

        # Tokenize data (we need to tokenize one by one because the length of all the provided URLs will not be the same)
        self.tokens = utils.encode(tokenizer, input_data, max_length=max_length, return_tensors=None, truncation=False)["input_ids"]

        # Truncate or remove
        if remove_instead_of_truncate:
            initial_pairs = len(self.token)

            self.tokens = list(filter(lambda pair: len(pair) <= max_length, self.tokens))

            after_remove_pairs = len(self.token)

            logger.debug("%d pairs of URLs have been removed%s: from %d to %d pairs", initial_pairs - after_remove_pairs,
                                                                                      f" ({self.set_desc})" if self.set_desc else '',
                                                                                      initial_pairs, after_remove_pairs)
        else:
            needs_truncation = sum([1 if len(pair) > max_length else 0 for pair in self.tokens])

            logger.debug("%d pairs of URLs need truncation of %d pairs%s", needs_truncation, len(self.tokens),
                                                                           f" ({self.set_desc})" if self.set_desc else '')

            self.tokens = [pair[:max_length] for pair in self.tokens]

        self._total_tokens = sum([len(t) for t in self.tokens])
        disable_balance = False

        if len(self.labels["urls_classification"]) != len(self.tokens):
            raise Exception("Number of input entries from the main task is different of the labels len: "
                            f"{len(self.tokens)} vs {len(self.labels['urls_classification'])}")

        if "labels_language_identification" in tasks_data and len(tasks_data["labels_language_identification"]) != 0:
            logger.debug("Loading labels for task: language-identification and/or langid-and-urls_classification")

            if len(tasks_data["labels_language_identification"]) != len(self.labels["urls_classification"]):
                raise Exception("Number of labels from the main task is different for the lang. id. task: "
                                f"{len(self.labels['urls_classification'])} vs {len(tasks_data['labels_language_identification'])}")

            disable_balance = True
            self.labels["language-identification"] = np.zeros(len(self.tokens))
            self.labels["langid-and-urls_classification"] = np.zeros(len(self.tokens))

            for idx, (label, lang_id_label) in enumerate(zip(self.labels["urls_classification"], tasks_data["labels_language_identification"])):
                self.labels["language-identification"][idx] = lang_id_label

                if tasks_data["reward_if_only_langid_is_correct_too"]:
                    self.labels["langid-and-urls_classification"][idx] = (lang_id_label * label + lang_id_label) / 2 # (w * x * y + x) / (w + 1) ; w = 1
                else:
                    self.labels["langid-and-urls_classification"][idx] = lang_id_label * label

        # Imbalanced strategy?
        if imbalanced_strategy:
            balance = False

            if imbalanced_strategy in ("none", "weighted-loss"):
                # It is not our responsability
                pass
            elif imbalanced_strategy == "over-sampling":
                balance = True
                imbalanced_obj = imblearn.over_sampling.RandomOverSampler(sampling_strategy="minority")
            else:
                raise Exception(f"Unknown imbalanced_strategy: {imbalanced_strategy}")

            if balance and disable_balance:
                logger.error("Currently is not supported to apply the selected imbalanced strategy with some auxiliary tasks: %s",
                             imbalanced_strategy)
            elif balance:
                lengths_before = [np.sum(self.labels["urls_classification"] == 0), np.sum(self.labels["urls_classification"] == 1)]
                self.tokens, _ = pad_sequence(self.tokens, self.pad_token_id, max_length=self.max_length) # We need to apply padding :/
                self.tokens, self.labels["urls_classification"] = imbalanced_obj.fit_resample(self.tokens, self.labels["urls_classification"])
                self.tokens = remove_padding(self.tokens, self.pad_token_id) # Remove padding
                lengths_after = [np.sum(self.labels["urls_classification"] == 0), np.sum(self.labels["urls_classification"] == 1)]

                logger.info("Imbalanced strategy '%s': from %s to %s", imbalanced_strategy, str(lengths_before), str(lengths_after))

        # Postprocess labels
        for task in ("urls_classification", "language-identification", "langid-and-urls_classification"):
            if task in self.labels:
                self.labels[task] = torch.from_numpy(self.labels[task])
                self.labels[task] = self.labels[task].type(torch.float) if regression else self.labels[task].type(torch.long)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #return {"url_str": self.data[idx], "label": self.labels[idx]}
        #return {
        #    "url_tokens": self.tokens[idx],
        #    #"url_attention_mask": self.attention_mask[idx],
        #    "label": self.labels[idx],
        #}

        result = {
            "url_tokens": self.tokens[idx],
            "label": self.labels["urls_classification"][idx],
        }

        if "language-identification" in self.labels:
            result["label-language-identification"] = self.labels["language-identification"][idx]
        if "langid-and-urls_classification" in self.labels:
            result["label-langid-and-urls_classification"] = self.labels["langid-and-urls_classification"][idx]

        return result

    def get_dataloader(self, batch_size, device, num_workers, sampler=None, max_tokens=None, set_dataloader=True):
        is_device_gpu = device.type.startswith("cuda")

        if sampler:
            self.sampler = sampler
        elif self.sampler_better_randomness:
            # LengthGroupedSampler handles worse the padding problem (suboptimal) but better the randomness than SmartBatchingSampler
            lengths = [len(seq) for seq in self.tokens]
            self.sampler = transformers.trainer_pt_utils.LengthGroupedSampler(batch_size, lengths=lengths)
        else:
            self.sampler = SmartBatchingSampler(
                data_source=self.tokens,
                batch_size=batch_size,
            )

        if max_tokens:
            logger.info("Batch size will be data-dependant%s: batches of, approximately, %d tokens will be returned",
                        f" ({self.set_desc})" if self.set_desc else '', max_tokens)

            collate_fn = MaxTokensCollate(
                pad_token_id=self.pad_token_id,
                max_tokens=max_tokens,
                total_number_of_batches=len(self.tokens),
            )
        else:
            collate_fn = SmartBatchingCollate(
                pad_token_id=self.pad_token_id,
            )

        # "RuntimeError: DataLoader worker (pid 22966) is killed by signal: Killed."
        #  Workaround: num_workers = 0
        #  Solution: https://github.com/pytorch/pytorch/issues/8976

        if not is_device_gpu and num_workers < 0:
            num_workers = len(os.sched_getaffinity(0)) # Same value used by dataloader implementation

            logger.debug("Num. workers%s: %d", f" ({self.set_desc})" if self.set_desc else '', num_workers)
        else:
            num_workers = 0 if is_device_gpu else num_workers # 0 is the adviced value in https://pytorch.org/docs/stable/data.html
                                                              #  when GPU is being used

        dataloader_kwargs = {
            "pin_memory": True,
            "pin_memory_device": device.type,
            "num_workers": num_workers,
        }

        # Check if we can use recent pytorch features
        pytorch_major, pytorch_minor, pytorch_patch = utils.get_pytorch_version()

        if pytorch_major > 1 or (pytorch_major == 1 and pytorch_minor >= 12):
            # Ok
            pass
        else:
            logger.warning("Unexpected pytorch version: making some changes in DataLoader")

            del dataloader_kwargs["pin_memory_device"]

        dataloader = DataLoader(
            dataset=self,
            batch_size=None if max_tokens else batch_size, # https://pytorch.org/docs/stable/data.html#disable-automatic-batching
            sampler=self.sampler,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        if set_dataloader:
            if self.dataloader:
                logger.warning("Be aware that the dataloader has been updated%s", f" ({self.set_desc})" if self.set_desc else '')

            self.dataloader = dataloader

        return dataloader

    @property
    def total_tokens(self):
        return self._total_tokens

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)

        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths) # Get indexes of tokens sorted by length
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size)) # Batches of indexes sorted by tokens length
        self._backsort_inds = None

    def __iter__(self):
        _batches = self.batches

        if _batches:
            last_batch = _batches.pop(-1) # Remove last element before randomizing since its length might be less than the batch size

            np.random.shuffle(_batches) # Randomize batches
            _batches.append(last_batch) # Add the previously removed last element

        self._inds = list(more_itertools.flatten(_batches))

        yield from self._inds # Return index of the randomized batches flattened but sorted by tokens length

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)

        return self._backsort_inds

class SmartBatchingCollate:
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        targets_lang_id = None
        sequences = [b["url_tokens"] for b in batch]
        targets = [b["label"] for b in batch]
        targets_lang_id = None
        targets_lang_id_and_urls_classification = None

        if "label-language-identification" in batch[0]:
            targets_lang_id = [b["label-language-identification"] for b in batch]
        if "label-langid-and-urls_classification" in batch[0]:
            targets_lang_id_and_urls_classification = [b["label-langid-and-urls_classification"] for b in batch]

        input_ids, attention_mask = pad_sequence(sequences, self._pad_token_id)

        output = {
            "url_tokens": input_ids,
            "url_attention_mask": attention_mask,
            "labels": torch.tensor(targets),
        }

        if targets_lang_id is not None:
            output["labels_task_language_identification"] = torch.tensor(targets_lang_id)
        if targets_lang_id_and_urls_classification is not None:
            output["labels_task_language_identification_and_urls_classification"] = torch.tensor(targets_lang_id_and_urls_classification)

        return output

class MaxTokensCollate:
    # Issues related:
    #  https://github.com/microsoft/DeepSpeed/issues/1051
    #  Mentioning --max_tokens from fairseq: https://github.com/huggingface/transformers/issues/10512

    def __init__(self, pad_token_id, max_tokens, total_number_of_batches):
        self._pad_token_id = pad_token_id
        self._max_tokens = max_tokens
        self._total_number_of_batches = total_number_of_batches

        self.reset_max_tokens_variables(last_or_first_batch=True)

    def reset_max_tokens_variables(self, last_or_first_batch=False):
        # Max tokens variables
        self._current_tokens = 0
        self._current_batch = []
        self._current_max_length = 0

        if last_or_first_batch:
            self._current_number_batch = 0
            self._aux_batch = [] # Auxiliar storage (we want to avoid to exceed max_tokens)

    def __call__(self, batch):
        targets_lang_id = None
        sequence = batch["url_tokens"]

        if len(self._aux_batch) > 0:
            self._current_batch.extend(self._aux_batch)
            self._aux_batch = []
            self._current_max_length = max(self._current_max_length, max([len(b["url_tokens"]) for b in self._current_batch]))

        self._current_max_length = max(self._current_max_length, len(sequence)) # Necessary for padding
        self._current_tokens = self._current_max_length * (len(self._current_batch) + 1) # Simulate padding with the current longest sentence
        self._current_number_batch += 1
        equal_max_tokens_processed = self._current_tokens == self._max_tokens
        more_max_tokens_processed = self._current_tokens > self._max_tokens
        max_tokens_processed = equal_max_tokens_processed or more_max_tokens_processed
        last_batch = self._current_number_batch >= self._total_number_of_batches
        force_return = False

        if more_max_tokens_processed and not last_batch:
            self._aux_batch.append(batch)

            force_return = True
        else:
            self._current_batch.append(batch)

        if more_max_tokens_processed and last_batch:
            logger.warning("Specified max_tokens have been exceeded: edge case where we had some element in the auxiliary "
                           "storage because of the previous iteration but we hit the last batch and has to be processed: "
                           "this might cause an OOM if using GPU: %d extra tokens", self._current_tokens - self._max_tokens)

        if force_return or max_tokens_processed or last_batch:
            # Return dynamic batch when max_tokens criteria is met or last batch is being processed
            sequences = [b["url_tokens"] for b in self._current_batch]
            targets = [b["label"] for b in self._current_batch]
            targets_lang_id = None
            targets_lang_id_and_urls_classification = None

            if "label-language-identification" in self._current_batch[0]:
                targets_lang_id = [b["label-language-identification"] for b in self._current_batch]
            if "label-langid-and-urls_classification" in self._current_batch[0]:
                targets_lang_id_and_urls_classification = [b["label-langid-and-urls_classification"] for b in self._current_batch]

            input_ids, attention_mask = pad_sequence(sequences, self._pad_token_id)

            output = {
                "url_tokens": input_ids,
                "url_attention_mask": attention_mask,
                "labels": torch.tensor(targets),
            }

            if targets_lang_id is not None:
                output["labels_task_language_identification"] = torch.tensor(targets_lang_id)
            if targets_lang_id_and_urls_classification is not None:
                output["labels_task_language_identification_and_urls_classification"] = torch.tensor(targets_lang_id_and_urls_classification)

            # Reset variables
            self.reset_max_tokens_variables(last_or_first_batch=last_batch)

            # Return batch
            return output
        else:
            # Keep accumulating partial batches
            return None

def tokenize_batch_from_iterator(iterator, tokenizer, batch_size, f=None, ignore_source_side=False, return_urls=False):
    def reset():
        urls = {
            "urls": [],
            "labels": [],
        }
        initial_urls = []

        return urls, initial_urls

    f = lambda q: q if f is None else f
    urls, initial_urls = reset()

    for idx, url in enumerate(iterator, 1):
        url = url.strip().split('\t')

        assert len(url) == 3, f"src, trg, label"

        src_url, trg_url, parallel_urls_output = f(url[0]), f(url[1]), int(url[2])

        if isinstance(src_url, list):
            assert len(src_url) == 1, src_url

            src_url = src_url[0]
        if isinstance(trg_url, list):
            assert len(trg_url) == 1, trg_url

            trg_url = trg_url[0]

        assert parallel_urls_output in (0, 1), parallel_urls_output

        if ignore_source_side:
            urls["urls"].append(trg_url)
        else:
            if tokenizer.sep_token in src_url or tokenizer.sep_token in trg_url:
                logger.warning("Skip since data contain the separator token: ('%s', '%s')", src_url, trg_url)

                continue

            urls["urls"].append(f"{src_url}{tokenizer.sep_token}{trg_url}") # We don't need to add [CLS] and final [SEP]
                                                                            #  (or other special tokens) since they are automatically added
                                                                            #  by tokenizer.encode_plus / tokenizer.batch_encode_plus

        urls["labels"].append(parallel_urls_output)
        initial_urls.append((url[0], url[1]))

        if len(urls["urls"]) >= batch_size:
            if return_urls:
                yield urls, initial_urls
            else:
                yield urls

            urls, initial_urls = reset()

    if len(urls["urls"]) != 0:
        if return_urls:
            yield urls, initial_urls
        else:
            yield urls

        urls, initial_urls = reset()
