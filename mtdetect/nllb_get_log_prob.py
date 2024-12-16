
import sys
import gzip
import pickle

import torch
import torch.nn.functional as F
import transformers
import numpy as np

#pretrained_model = "facebook/nllb-200-1.3B"
#pretrained_model = "facebook/nllb-200-3.3B"
_src_lang = sys.argv[1]
_trg_lang = sys.argv[2]
direction = sys.argv[3]
pickle_output_fn = sys.argv[4] if len(sys.argv) > 4 else None
layer = sys.argv[5] if len(sys.argv) > 5 else -1
pretrained_model = sys.argv[6] if len(sys.argv) > 6 and len(sys.argv[6]) > 0 else "facebook/nllb-200-distilled-600M"
bsz = int(sys.argv[7]) if len(sys.argv) > 7 else 32

assert direction in ("src2trg", "trg2src")

if layer != "all":
    layer = int(layer)

print(f"Pretrained model: {pretrained_model}")
print(f"Batch size: {bsz}")

src_lang, trg_lang = (_src_lang, _trg_lang) if direction == "src2trg" else (_trg_lang, _src_lang)
source_lang_token = src_lang
target_lang_token = trg_lang

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
model = model.to(device).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=src_lang, tgt_lang=trg_lang)
max_length = model.config.max_length
max_new_tokens = model.generation_config.max_length
eos_token_token = tokenizer.convert_ids_to_tokens(model.generation_config.eos_token_id)
decoder_start_token_token = tokenizer.convert_ids_to_tokens(model.generation_config.decoder_start_token_id)
#bsz = 2
#bsz = 16
#bsz = 32

def get_log_prob(src_inputs, trg_inputs, length_normalization=False):
    model_output = model(**src_inputs, decoder_input_ids=trg_inputs["input_ids"])
    logits = model_output["logits"]
    log_probs = F.log_softmax(logits, dim=-1).cpu().detach()
    all_ntokens = []
    batch_log_probs = []
    batch_log_probs_all = []

    for bsz_idx in range(trg_inputs['input_ids'].shape[0]):
        batch_log_probs.append(0)
        batch_log_probs_all.append([])
        ntokens = 0

        for idx in range(1, trg_inputs['input_ids'].shape[-1] - 1):
            token_id = trg_inputs['input_ids'][bsz_idx,idx + 1]

            if token_id == tokenizer.pad_token_id:
                break

            log_prob = log_probs[bsz_idx,idx,token_id].item()
            batch_log_probs[-1] += log_prob
            batch_log_probs_all[-1].append(log_prob)
            ntokens += 1

        all_ntokens.append(ntokens)

        if length_normalization:
            batch_log_probs[-1] /= all_ntokens[-1]

    return batch_log_probs, all_ntokens, list(np.exp(batch_log_probs)), batch_log_probs_all

def get_model_hidden_state(src_inputs, trg_inputs, layer=-1):
    model_output = model(**src_inputs, decoder_input_ids=trg_inputs["input_ids"], output_hidden_states=True)
    results = {}

    assert layer == "all" or layer < 0, layer

    #for module, tokens in zip(("encoder", "decoder"), (src_inputs, trg_inputs)):
    for module, tokens in zip(("decoder",), (trg_inputs,)):
        n_layers = getattr(model.config, f"{module}_layers") + 1 # +1 for the embedding layer (https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput)

        assert len(model_output[f"{module}_hidden_states"]) == n_layers, len(model_output[f"{module}_hidden_states"])

        for l in range(n_layers):
            hidden_state = model_output[f"{module}_hidden_states"][l]
            expected_shape = (*tokens["input_ids"].shape, model.config.d_model)

            assert len(hidden_state.shape) == 3, f"{l}: hidden_state.shape"
            assert hidden_state.shape == expected_shape, f"{l}: {hidden_state.shape} vs {expected_shape}"

        if layer == "all":
            results[f"{module}_last_hidden_state"] = {l - n_layers: model_output[f"{module}_hidden_states"][l].cpu().detach() for l in range(n_layers)} # relative position starting on the last layer (i.e., -1)
        else:
            results[f"{module}_last_hidden_state"] = {layer: model_output[f"{module}_hidden_states"][layer].cpu().detach()}

        for k in tokens.keys():
            tokens[k] = tokens[k].cpu().detach()

        results[f"{module}_tokens"] = tokens

    return results

def preprocess_tokens(tokens, tokenizer, max_length):
    assert len(tokens["input_ids"].shape) == 2
    current_bsz, ntokens = tokens["input_ids"].shape

    if ntokens > max_length:
        tokens["input_ids"] = tokens["input_ids"][:,:max_length]
        tokens["attention_mask"] = tokens["attention_mask"][:,:max_length]

        for idx in range(current_bsz):
            if tokens["input_ids"][idx,-1] != tokenizer.pad_token_id:
                tokens["input_ids"][idx,-1] = tokenizer.eos_token_id

    return tokens

def preprocess(text, tokenizer, device, max_length):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, padding=True).to(device)
    inputs = preprocess_tokens(inputs, tokenizer, max_length)

    return inputs

def run_and_print_results(all_src, all_trg, extra_data, src_inputs, trg_inputs, perplexity_skip_values=5, previous_results=None, last_execution=False):
    batch_log_probs, all_ntokens, batch_probs, batch_log_probs_all = get_log_prob(src_inputs, trg_inputs)

    assert len(all_src) == len(all_trg) == len(extra_data) == len(batch_log_probs) == len(all_ntokens) == len(batch_probs) == len(batch_log_probs_all)
    assert isinstance(batch_log_probs_all, list), batch_log_probs_all

    for src, trg, ntokens, log_prob, prob, _extra_data, log_prob_per_token in zip(all_src, all_trg, all_ntokens, batch_log_probs, batch_probs, extra_data, batch_log_probs_all):
        assert isinstance(log_prob_per_token, list), log_prob_per_token

        _log_prob_per_token = '|'.join(map(str, log_prob_per_token))
        perplexity = [np.exp(-1. * sum(log_prob_per_token[:i + 1]) / (i + 1)) for i in range(len(log_prob_per_token))]
        _perplexity = '|'.join(map(str, perplexity))
        max_abs_diff_consecutive_perplexity_skip = {skip: [abs(perplexity[i] - perplexity[i + 1]) for i in range(skip, len(perplexity) - 1)] for skip in range(perplexity_skip_values)}
        max_abs_diff_consecutive_perplexity_skip = {skip: max(max_abs_diff_consecutive_perplexity_skip[skip]) if len(max_abs_diff_consecutive_perplexity_skip[skip]) else -1.0 for skip in range(perplexity_skip_values)}
        _max_abs_diff_consecutive_perplexity_skip = '|'.join(map(str, [max_abs_diff_consecutive_perplexity_skip[skip] for skip in range(perplexity_skip_values)]))
        max_abs_diff_consecutive_log_prob = max([abs(log_prob_per_token[i] - log_prob_per_token[i + 1]) for i in range(len(log_prob_per_token) - 1)])

        print(f"{ntokens}\t{log_prob}\t{prob}\t{src}\t{trg}\t{_extra_data}\t{max_abs_diff_consecutive_log_prob}\t{_log_prob_per_token}\t{_max_abs_diff_consecutive_perplexity_skip}\t{_perplexity}")

    return None

def run_and_store_last_hidden_layer_with_pickle(all_src, all_trg, extra_data, src_inputs, trg_inputs, pickle_output_fn=None, previous_results=None, last_execution=False, remove_padding=False, layer=-1):
    assert pickle_output_fn is not None
    assert isinstance(pickle_output_fn, str), type(pickle_output_fn)
    assert len(pickle_output_fn) > 0

    results = get_model_hidden_state(src_inputs, trg_inputs, layer=layer) # {key: torch.tensor(bsz, src|trg_tokens, d_model)}
    #modules = ("encoder", "decoder")
    modules = ("decoder",)
    layers = [sorted(list(set(results[f"{module}_last_hidden_state"].keys()))) for module in modules]

    for i in range(len(layers) - 1):
        assert layers[i] == layers[i + 1]

    layers = layers[0]
    results_aux = {f"{module}_last_hidden_state": {l: [] for l in layers} for module in modules}

    for l in layers:
        assert isinstance(l, int)

    current_bsz = {results[f"{module}_last_hidden_state"][layers[0]].shape[0] for module in modules}

    assert len(results.keys()) == len(modules) * 2, results.keys() # tokens and last_hidden_state
    assert len(current_bsz) == 1, current_bsz

    current_bsz = list(current_bsz)[0]

    for module in modules:
        assert f"{module}_last_hidden_state" in results
        assert f"{module}_tokens" in results

        tokens = results[f"{module}_tokens"]["input_ids"]

        assert tokens.shape[0] == current_bsz

        if remove_padding:
            for l in layers:
                for bsz_idx in range(current_bsz):
                    pad_found = False
                    first_pad_idx = -1
                    t = results[f"{module}_last_hidden_state"][l][bsz_idx]

                    assert len(t.shape) == 2, t.shape
                    assert t.shape[0] == tokens[bsz_idx].shape[0]

                    for position_idx in range(t.shape[0]):
                        token = tokens[bsz_idx][position_idx].cpu().detach().item()

                        assert isinstance(token, int), token

                        if pad_found:
                            assert token == tokenizer.pad_token_id
                        else:
                            pad_found = (token == tokenizer.pad_token_id)

                            if pad_found and first_pad_idx == -1:
                                first_pad_idx = position_idx

                    if pad_found:
                        assert first_pad_idx != -1

                        t = t[:first_pad_idx] # remove padding

                    t = t.cpu().detach()
                    results_aux[f"{module}_last_hidden_state"][l].append(t)
        else:
            for l in layers:
                results_aux[f"{module}_last_hidden_state"][l].append(results[f"{module}_last_hidden_state"][l])

    if previous_results is not None:
        assert set(previous_results.keys()) == set(results_aux.keys())

        for k in previous_results.keys():
            assert isinstance(previous_results[k], dict), type(previous_results[k])

            for l in previous_results[k]:
                assert isinstance(previous_results[k][l], list), type(previous_results[k][l])

                previous_results[k][l].extend(results_aux[k][l])

        last_hidden_state = previous_results
    else:
        last_hidden_state = results_aux

    if remove_padding:
        for module in modules:
            for l in layers:
                v = last_hidden_state[f"{module}_last_hidden_state"][l]

                if not last_execution:
                    assert len(v) % current_bsz == 0, f"{len(v)} % {current_bsz} vs 0"

                for bsz_idx in reversed(range(1, current_bsz + 1)):
                    print(f"{l}\t{(len(v) - current_bsz) + bsz_idx}\t{current_bsz}\t{module}\t{len(v)}\t{v[-1 * bsz_idx].shape[0]}")

    sys.stdout.flush()

    if last_execution:
        if layer == "all":
            for l in layers:
                _pickle_output_fn = f"{pickle_output_fn}.layer_{l}.gz"
                _last_hidden_state = {f"{module}_last_hidden_state": last_hidden_state[f"{module}_last_hidden_state"][l] for module in modules}

                print(f"Pickle output: layer {l}: {_pickle_output_fn}")
                sys.stdout.flush()

                with gzip.open(_pickle_output_fn, "wb") as pickle_fd:
                    pickle.dump(_last_hidden_state, pickle_fd)
        else:
            keys = {}

            for module in modules:
                assert len(last_hidden_state[f"{module}_last_hidden_state"].keys()) == 1, last_hidden_state[f"{module}_last_hidden_state"].keys()

                keys[module] = list(last_hidden_state[f"{module}_last_hidden_state"].keys())[0]

            _pickle_output_fn = f"{pickle_output_fn}.gz"
            _last_hidden_state = {f"{module}_last_hidden_state": last_hidden_state[f"{module}_last_hidden_state"][keys[module]] for module in modules}

            print(f"Pickle output: {_pickle_output_fn}")
            sys.stdout.flush()

            with gzip.open(_pickle_output_fn, "wb") as pickle_fd:
                pickle.dump(_last_hidden_state, pickle_fd)

    return last_hidden_state

if __name__ == "__main__":
    #run = run_and_print_results
    run = run_and_store_last_hidden_layer_with_pickle
    #run_kwargs = {"previous_results": None, "last_execution": False}
    run_kwargs = {"previous_results": None, "pickle_output_fn": pickle_output_fn, "last_execution": False, "layer": layer}
    all_src, all_trg, extra_data = [], [], []

    for l in sys.stdin:
        if len(all_src) >= bsz:
            src_inputs = preprocess(all_src, tokenizer, device, max_length)
            trg_inputs = preprocess(all_trg, tokenizer, device, max_new_tokens)

            results = run(all_src, all_trg, extra_data, src_inputs, trg_inputs, **run_kwargs)

            run_kwargs["previous_results"] = results
            all_src, all_trg, extra_data = [], [], []

        _data = l.strip("\r\n").split('\t')
        _src = _data[0]
        _trg = _data[1]
        _extra_data = '|'.join(_data[2:])

        src, trg = (_src, _trg) if direction == "src2trg" else (_trg, _src)
        src = f"{source_lang_token} {src}{eos_token_token}"
        trg = f"{decoder_start_token_token}{target_lang_token} {trg}{eos_token_token}"

        all_src.append(src)
        all_trg.append(trg)
        extra_data.append(_extra_data)

    assert len(all_src) > 0

    src_inputs = preprocess(all_src, tokenizer, device, max_length)
    trg_inputs = preprocess(all_trg, tokenizer, device, max_new_tokens)
    run_kwargs["last_execution"] = True

    run(all_src, all_trg, extra_data, src_inputs, trg_inputs, **run_kwargs)

    run_kwargs["previous_results"] = results
    all_src, all_trg, extra_data = [], [], []
