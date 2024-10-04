
import os
import logging

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

    def __init__(self, input_data, output_data, tokenizer, max_length, sampler_better_randomness=True, set_desc='', groups=None, groups_balanced=None):
        super().__init__()

        if groups is None:
            self.groups = [str(idx) for idx in range(len(input_data))]
        else:
            self.groups = groups

        if groups_balanced is None:
            self.groups_balanced = ["none" for _ in range(len(input_data))]
        else:
            self.groups_balanced = groups_balanced

        assert len(input_data) == len(output_data)
        assert len(input_data) == len(self.groups)
        assert len(input_data) == len(self.groups_balanced)

        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.sampler_better_randomness = sampler_better_randomness
        self.dataloader = None
        self.set_desc = set_desc
        self.labels = {
            "urls_classification": np.array(output_data)
        }
        self.groups2group_balanced = {}

        # Tokenize data (we need to tokenize one by one because the length of all the provided URLs will not be the same)
        # We let the tokenizer do the truncation because manual truncation may remove special tokens...
        self.tokens = utils.encode(tokenizer, input_data, max_length=max_length, return_tensors=None, truncation=True)["input_ids"]

        self._total_tokens = sum([len(t) for t in self.tokens])

        if len(self.labels["urls_classification"]) != len(self.tokens):
            raise Exception("Number of input entries from the main task is different of the labels len: "
                            f"{len(self.tokens)} vs {len(self.labels['urls_classification'])}")

        # Postprocess labels
        self.labels["urls_classification"] = torch.from_numpy(self.labels["urls_classification"])
        self.labels["urls_classification"] = self.labels["urls_classification"].type(torch.float) # Regression

        assert self.labels["urls_classification"].shape == (len(output_data),), self.labels["urls_classification"].shape

        self.uniq_groups = []
        self.groups_balanced_aligned_with_uniq_groups = []
        self.data = {}
        self.max_tokens_per_group = {}

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
                    "tokens": [],
                    "labels": [],
                }
                self.max_tokens_per_group[group] = -np.inf
                self.groups2group_balanced[group] = group_balanced

            assert self.groups2group_balanced[group] == group_balanced # this should be also true when group in self.data (i.e., did not enter the previous if statement)
            assert len(self.groups_balanced_aligned_with_uniq_groups) == len(self.uniq_groups) # these two should be aligned in terms of nubmer of elements

            self.data[group]["tokens"].append(self.tokens[idx])
            self.data[group]["labels"].append(self.labels["urls_classification"][idx])
            self.max_tokens_per_group[group] = max(self.max_tokens_per_group[group], len(self.tokens[idx]))

            assert self.max_tokens_per_group[group] >= 0, self.max_tokens_per_group[group]

        assert len(self.data) == len(self.uniq_groups)
        assert len(self.groups2group_balanced) == len(self.uniq_groups)
        assert len(set(self.groups2group_balanced.values())) == len(set(self.groups_balanced_aligned_with_uniq_groups))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert isinstance(idx, int), type(idx)

        group = self.uniq_groups[idx]
        url_tokens = [list(t) for t in self.data[group]["tokens"]]
        label = [l.clone().detach() for l in self.data[group]["labels"]]
        result = {
            "url_tokens": url_tokens,
            "label": label,
            "group": group,
        }

        return result

    def get_dataloader(self, batch_size, device, num_workers, sampler=None, max_tokens=None, set_dataloader=True):
        is_device_gpu = device.type.startswith("cuda")

        if not sampler:
            lengths = [self.max_tokens_per_group[group] for group in self.uniq_groups] # We use the max tokens length of the group to cover the worst case

        if sampler:
            self.sampler = sampler

            assert max_tokens is None
        elif self.sampler_better_randomness:
            # LengthGroupedSampler handles worse the padding problem (suboptimal) but better the randomness than SmartBatchingSampler
            self.sampler = transformers.trainer_pt_utils.LengthGroupedSampler(batch_size, lengths=lengths)
        else:
            self.sampler = SmartBatchingSampler(batch_size, lengths)

        if max_tokens:
            logger.info("Batch size will be data-dependant%s: batches of, approximately, %d tokens will be returned",
                        f" ({self.set_desc})" if self.set_desc else '', max_tokens)

            main_collate_fn = MaxTokensCollate(
                pad_token_id=self.pad_token_id,
                max_tokens=max_tokens,
                total_number_of_batches=len(self.tokens),
            )
        else:
            main_collate_fn = SmartBatchingCollate(
                pad_token_id=self.pad_token_id,
            )

        groups_collate_fn = SelectGroupCollate()
        collate_fn = utils.chain_collate_fn(groups_collate_fn, main_collate_fn)

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
            "pin_memory_device": device.type, # This does not actually copy data to the device, it only "reports" about the future intention: https://discuss.pytorch.org/t/attributeerror-dataloader-object-has-no-attribute-pin-memory-device/170129/3 (and /4)
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

    @property
    def groups_are_affecting_total_tokens_count(self):
        return len(self.uniq_groups) < len(self.groups)

class SmartBatchingSampler(Sampler):
    def __init__(self, batch_size, lengths):
        super().__init__() # class Sampler: 'data_source' argument is not used and will be removed in 2.2.0

        self.len = len(lengths)
        argsort_inds = np.argsort(lengths) # Get indexes of tokens sorted by length
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

class GroupBalancedSampler(Sampler):
    # https://arxiv.org/abs/1907.05019 section 4.2

    def __init__(self, dataset, batch_size, temperature_sampling=1, normalize_p=True, shuffle=False, force_last_batch_to_be_complete=False,
                 desc='-', logger=None):
        self.total_elements = len(dataset.data.keys())
        self.desc = desc
        self.logger_debug_func = logger.debug if logger else lambda s: print(f"DEBUG: {s}")

        if force_last_batch_to_be_complete:
            self.total_elements += max(self.total_elements, self.batch_size) % self.batch_size

        self.batch_size = batch_size
        self.temperature_sampling = temperature_sampling
        self.normalize_p = normalize_p
        self.shuffle = shuffle

        assert normalize_p, "np.random.choice needs that p adds up to 1.0"

        # Obtain relevant elements from dataset
        self.groups = dataset.groups
        self.uniq_groups = dataset.uniq_groups
        self.groups_balanced = dataset.groups_balanced
        self.groups_balanced_aligned_with_uniq_groups = dataset.groups_balanced_aligned_with_uniq_groups
        self.uniq_groups_balanced = set(self.groups_balanced_aligned_with_uniq_groups)

        assert len(set(self.uniq_groups)) == len(self.uniq_groups)
        assert len(self.uniq_groups) == self.total_elements
        assert len(self.uniq_groups) == len(self.groups_balanced_aligned_with_uniq_groups) # aligned in terms of number of elements
        assert len(set(self.groups_balanced)) == len(self.uniq_groups_balanced)
        assert len(self.groups) == len(self.groups_balanced)

        # Obtain p
        # NOTICE that p is calculated using self.uniq_groups and not self.groups, meaning that the considered size of the dataset is the number of unique groups
        #self.pre = {uniq_group_balanced: sum([1 if group_balanced == uniq_group_balanced else 0 for group_balanced in self.groups_balanced]) for uniq_group_balanced in self.uniq_groups_balanced}
        self.pre = {uniq_group_balanced: len(set([group for group, group_balanced in zip(self.groups, self.groups_balanced) if group_balanced == uniq_group_balanced])) for uniq_group_balanced in self.uniq_groups_balanced}
        self.p = {k: (p / self.total_elements) ** (1 / self.temperature_sampling) for k, p in self.pre.items()}
        self.normalization_ratio = sum(self.p.values())

        #assert sum(self.pre.values()) == len(self.groups) # This is True if p is calculated using self.groups instead of self.uniq_groups
        assert sum(self.pre.values()) == self.total_elements, f"{self.pre} (sum. of values: {sum(self.pre.values())}) vs {self.total_elements}"

        if self.normalize_p:
            self.p = {k: p / self.normalization_ratio for k, p in self.p.items()}

            assert np.isclose(sum([p for p in self.p.values()]), 1.0)

        # Necessary elements to sample
        #self.group_balanced2groups = {uniq_group_balanced: [group for group, group_balanced in zip(self.groups, self.groups_balanced) if group_balanced == uniq_group_balanced] for uniq_group_balanced in self.uniq_groups_balanced}
        self.group_balanced2groups = {uniq_group_balanced: [uniq_group for uniq_group, inner_uniq_group_balanced in zip(self.uniq_groups, self.groups_balanced_aligned_with_uniq_groups) if inner_uniq_group_balanced == uniq_group_balanced] for uniq_group_balanced in self.uniq_groups_balanced}
        self.group_balanced_n_elements = {k: len(v) for k, v in self.group_balanced2groups.items()}

        for uniq_group_balanced in self.uniq_groups_balanced:
            initial_n = self.group_balanced_n_elements[uniq_group_balanced]
            initial_values = list(self.group_balanced2groups[uniq_group_balanced])

            assert initial_n == len(initial_values)

            while self.group_balanced_n_elements[uniq_group_balanced] < self.total_elements:
                if self.shuffle:
                    np.random.shuffle(initial_values) # necessary for the last iteration, so we do not obtain just the first elements

                self.group_balanced2groups[uniq_group_balanced] += initial_values # replicate the list as many times as needed
                self.group_balanced_n_elements[uniq_group_balanced] += initial_n

            assert self.group_balanced_n_elements[uniq_group_balanced] >= self.total_elements
            assert self.group_balanced_n_elements[uniq_group_balanced] - initial_n < self.total_elements

            self.group_balanced2groups[uniq_group_balanced] = self.group_balanced2groups[uniq_group_balanced][:self.total_elements] # remove extra (unnecessary) elements
            self.group_balanced_n_elements[uniq_group_balanced] = len(self.group_balanced2groups[uniq_group_balanced]) # adjust count

        assert len(set(self.group_balanced_n_elements.values())) == 1
        assert list(self.group_balanced_n_elements.values())[0] == self.total_elements

        set_data = {k: set(v) for k, v in self.group_balanced2groups.items()}
        list_uniq_groups_balanced = list(self.uniq_groups_balanced)

        for idx1 in range(len(set_data)):
            for idx2 in range(idx1 + 1, len(set_data)):
                bgroup1 = list_uniq_groups_balanced[idx1]
                bgroup2 = list_uniq_groups_balanced[idx2]

                assert len(set.intersection(set_data[bgroup1], set_data[bgroup2])) == 0, "Intersection between different balanced groups is not supported"

        self.logger_debug_func(f"{self.desc}: indices are ready to be created. Value of p: {self.p} (number of unique groups: {self.pre})")

    def __iter__(self):
        # Create indices

        if self.shuffle:
            # shuffling each new epoch is necessary, if self.shuffle is set

            for uniq_group_balanced in self.uniq_groups_balanced:
                np.random.shuffle(self.group_balanced2groups[uniq_group_balanced])

        indices = []
        sampled_elements = {k: 0 for k in self.uniq_groups_balanced}
        uniq_groups_balanced, p = zip(*[(k, p) for k, p in self.p.items()])
        uniq_groups_balanced, p = list(uniq_groups_balanced), list(p)

        for _ in range(self.total_elements):
            group_balanced = np.random.choice(uniq_groups_balanced, size=None, replace=False, p=p)
            idx = sampled_elements[group_balanced]
            group = self.group_balanced2groups[group_balanced][idx]
            sampled_elements[group_balanced] += 1
            group_idx = self.uniq_groups.index(group)

            indices.append(group_idx)

        sampled_elements_total = sum(sampled_elements.values())
        sampled_elements_p = {k: v / sampled_elements_total * 100 for k, v in sampled_elements.items()}

        self.logger_debug_func(f"{self.desc}: new indices have been created. Sampled elements: {sampled_elements} (total: {sampled_elements_total}; perc: {sampled_elements_p})")

        yield from indices

    def __len__(self):
        return self.total_elements

class SmartBatchingCollate:
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        sequences = [b["url_tokens"] for b in batch]
        targets = [b["label"] for b in batch]

        input_ids, attention_mask = pad_sequence(sequences, self._pad_token_id)

        output = {
            "url_tokens": input_ids,
            "url_attention_mask": attention_mask,
            "labels": torch.tensor(targets),
        }

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

            input_ids, attention_mask = pad_sequence(sequences, self._pad_token_id)

            output = {
                "url_tokens": input_ids,
                "url_attention_mask": attention_mask,
                "labels": torch.tensor(targets),
            }

            # Reset variables
            self.reset_max_tokens_variables(last_or_first_batch=last_batch)

            # Return batch
            return output
        else:
            # Keep accumulating partial batches
            return None

class SelectGroupCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        #data, labels = [], []
        result = []

        for idx in range(len(batch)):
            x = batch[idx]["url_tokens"]
            y = batch[idx]["label"]
            group = batch[idx]["group"]

            assert len(y) > 0
            assert len(x) == len(y)
            assert ':' not in group

            group_idx = np.random.randint(len(y))
            output_x = x[group_idx]
            output_y = y[group_idx]

            result.append({
                "url_tokens": output_x,
                "label": output_y,
            })

        # The output format is the same that DataLoaded is using because this collate_fn is expected to be used before an actual collate_fn
        # Check utils.chain_collate_fn

        return result

def tokenize_batch_from_iterator(iterator, tokenizer, batch_size, f=None, ignore_source_side=False, return_urls=False):
    def reset():
        urls = {
            "urls": [],
            "labels": [],
            "groups": [],
            "groups_balanced": [],
        }
        initial_urls = []

        return urls, initial_urls

    f = (lambda q: q) if f is None else f
    urls, initial_urls = reset()

    for idx, url in enumerate(iterator, 1):
        url = url.strip().split('\t')

        assert len(url) >= 3, f"src trg label [group]"

        src_url, trg_url, label = f(url[0]), f(url[1]), int(url[2])
        group = url[3] if len(url) > 3 else str(idx)
        group_data = group.split(':')
        group = group_data[0] # remove "optional" part of the group
        group_balanced = "none"

        if len(group_data) > 1:
            group_data_balanced = ':'.join(group_data[1:]) # get the "other" group to balance different datasets from the "optional" part of the group

            if '#' in group_data_balanced:
                group_balanced = group_data_balanced.split('#')[-1]

        if isinstance(src_url, list):
            assert len(src_url) == 1, src_url

            src_url = src_url[0]
        if isinstance(trg_url, list):
            assert len(trg_url) == 1, trg_url

            trg_url = trg_url[0]

        assert label in (0, 1), label

        if ignore_source_side:
            urls["urls"].append(trg_url)
        else:
            if tokenizer.sep_token in src_url or tokenizer.sep_token in trg_url:
                logger.warning("Skip since data contain the separator token: ('%s', '%s')", src_url, trg_url)

                continue

            urls["urls"].append(f"{src_url}{tokenizer.sep_token}{trg_url}") # We don't need to add [CLS] and final [SEP]
                                                                            #  (or other special tokens) since they are automatically added
                                                                            #  by tokenizer.encode_plus / tokenizer.batch_encode_plus

        urls["labels"].append(label)
        urls["groups"].append(group)
        urls["groups_balanced"].append(group_balanced)
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
