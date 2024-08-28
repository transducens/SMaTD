
import sys
import pickle

import torch
import torch.nn.functional as F
import transformers
import numpy as np

# NLLB supported languages: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

# Use wrapper_example_translation_nllb.py to parallelize with parallel

# Functions

def get_lang_token(tokenizer, lang):
    token = tokenizer.convert_tokens_to_ids(lang)
    aux_lang = tokenizer.convert_ids_to_tokens(token)

    assert lang == aux_lang, f"{lang} != {aux_lang}"

    return token

def decode(tokenizer, translated_tokens):
    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    return output

def translate_from_generate(tokenizer, model, inputs, target_lang_id, max_length, beam_size=1):
    translated_tokens = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_new_tokens=max_length, num_return_sequences=1, num_beams=beam_size)
    translation = decode(tokenizer, translated_tokens)

    return translation

def translate_from_pipeline(translator_pipeline, source_text, beam_size=1):
    output = translator_pipeline(source_text, num_beams=beam_size)
    translation = [_output["translation_text"] for _output in output]

    return translation

def capture_attention_grads(layer, component, attentions_grad_store):
    def _f(grad):
        if layer not in attentions_grad_store:
            attentions_grad_store[layer] = {}

        assert component not in attentions_grad_store[layer]

        attentions_grad_store[layer][component] = grad

    return _f

def normalize(r):
    # Code: https://github.com/hila-chefer/Transformer-MM-Explainability/blob/58eaea85ac9c34aff052f368514b35d2e4c8dd3c/lxmert/lxmert/src/ExplanationGenerator.py#L45

    self_attention = np.array(r, copy=True)
    diag_idx = range(self_attention.shape[-1])
    self_attention -= np.eye(self_attention.shape[-1])

    assert self_attention[diag_idx, diag_idx].min() >= 0

    self_attention = self_attention / self_attention.sum(axis=-1, keepdims=True)
    self_attention += np.eye(self_attention.shape[-1])

    return self_attention

def print_attention(tokens_a, tokens_b, rel):
    assert len(rel.shape) == 2
    assert rel.shape[0] == len(tokens_a)
    assert rel.shape[1] == len(tokens_b)

    #for token in tokens_b:
    #    sys.stdout.write(f"\t{token}")
    #sys.stdout.write('\n')

    for i in range(len(tokens_a)):
        sys.stdout.write(tokens_a[i])

        for j in range(len(tokens_b)):
            sys.stdout.write(f"\t{tokens_b[j]}:{rel[i][j]:.2f}")

        sys.stdout.write('\n')

translator_pipeline = None
model = None
tokenizer = None

def explainability(source_text, target_text='', source_lang="eng_Latn", target_lang="spa_Latn", debug=False, apply_normalization=True,
                   self_attention_remove_diagonal=True, explainability_normalization="relative", device=None, beam_size=1,
                   pretrained_model=None):
    # Load NLLB
    assert isinstance(source_text, str), type(source_text)
    assert isinstance(target_text, str), type(target_text)

    if not pretrained_model:
        pretrained_model = "facebook/nllb-200-distilled-600M"

    global model, tokenizer, translator_pipeline

    batch_size = 1

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        if debug:
            print("DEBUG: model initialization")

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model).to(device)

    if tokenizer is None:
        if debug:
            print("DEBUG: tokenizer initialization")

        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=source_lang, tgt_lang=target_lang)

    max_length_encoder = tokenizer.model_max_length
    max_length_decoder = model.generation_config.max_length
    num_hidden_layers = model.config.num_hidden_layers # Model layers
    num_attention_heads = model.config.num_attention_heads # Heads each attention layer has

    if debug:
        if translator_pipeline is None:
            translator_pipeline = transformers.pipeline("translation", model=model, tokenizer=tokenizer, batch_size=batch_size, src_lang=source_lang,
                                                        tgt_lang=target_lang, max_new_tokens=max_length_decoder, truncation=True, device=device)

    model.eval()

    target_lang_id = get_lang_token(tokenizer, target_lang)
    source_lang_id = get_lang_token(tokenizer, source_lang) # sanity check

    # Translate

    inputs = tokenizer(source_text, return_tensors="pt", add_special_tokens=True, max_length=max_length_encoder, truncation=True).to(device)
    input_tokens = [tokenizer.convert_ids_to_tokens(_id) for _id in inputs.input_ids[0].cpu().detach().tolist()]

    # We can't use model.generate because we need to apply teacher forcing and need the attention...
    # Generation strategy: following https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/generation_config.json
    #  ... and default model.generate parameters: https://huggingface.co/docs/transformers/v4.44.0/en/main_classes/text_generation#transformers.GenerationConfig

    # https://huggingface.co/facebook/nllb-200-distilled-600M/commit/716b434935682cabef30af23dc1128d84b9003d2:
    #
    #{
    #  "_from_model_config": true,
    #  "bos_token_id": 0,
    #  "decoder_start_token_id": 2,
    #  "eos_token_id": 2,
    #  "max_length": 200,
    #  "pad_token_id": 1,
    #  "transformers_version": "4.27.0.dev0"
    #}

    decoder_start_token_id = tokenizer.eos_token_id

    # _from_model_config : https://github.com/huggingface/transformers/blob/52cb4034ada381fe1ffe8d428a1076e5411a8026/src/transformers/trainer_seq2seq.py#L315
    #  ... it handles the generation configuration override by the user, so we do not need to worry about it

    teacher_forcing = bool(target_text)

    if teacher_forcing and beam_size != 1:
        print(f"warning: beam_size is going to be modified ({beam_size} -> 1) because teacher forcing is enabled")

        beam_size = 1

    outputs = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist() if teacher_forcing else []
    generated_tokens = [decoder_start_token_id, target_lang_id] + outputs # NLLB starts with these two tokens
    initial_tokens = len(generated_tokens)

    assert batch_size == 1, batch_size

    vocab_size = None
    beams = [(list(generated_tokens), 0.0)] # copy generated_tokens using list()
    completed_beams = []

    # Beam search generation (greedy generation iff beam_size=1)
    for i in range(max_length_decoder):
        new_beams = []

        if i + 1 == max_length_decoder:
            break

        for beam_tokens, beam_score in beams:
            decoder_input_ids = torch.tensor([beam_tokens]).to(device)

            assert decoder_input_ids.shape == (batch_size, i + initial_tokens), decoder_input_ids.shape

            model_output = model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=False if beam_size > 1 else True) # odict_keys(['logits', 'past_key_values', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_attentions'])
            logits = model_output.logits
            vocab_size = logits.shape[-1] if vocab_size is None else vocab_size

            assert logits.shape == (batch_size, i + initial_tokens, vocab_size), f"{logits.shape} ... {vocab_size}"

            logits = logits[:, -1, :] # Get last token logits
            log_probs = F.log_softmax(logits, dim=-1).cpu()
            token_ids = torch.topk(-log_probs, beam_size, largest=False, sorted=True).indices.squeeze(0).tolist() # get minimum (instead of maximum due to log) values (i.e., tokens)

            if beam_size > 1:
                del decoder_input_ids
                del model_output
                del logits

                torch.cuda.empty_cache()

            if teacher_forcing:
                # Next token might be different of EoS -> force
                assert len(token_ids) == 1, token_ids

                next_token_id = token_ids[0]

                if next_token_id != tokenizer.eos_token_id:
                    if debug:
                        print(f"DEBUG: last generated token is not EoS but {tokenizer.convert_ids_to_tokens(next_token_id)} (id: {next_token_id}). This is expected due to teacher forcing")

                    token_ids = [tokenizer.eos_token_id]

            for token_id in token_ids:
                token_log_prob = log_probs[0, token_id].item()
                new_score = beam_score + token_log_prob
                new_sequence = beam_tokens + [token_id]

                if token_id == tokenizer.eos_token_id:
                    new_sequence = new_sequence[:-1]

                    completed_beams.append((new_sequence, new_score / len(new_sequence)))
                else:
                    new_beams.append((new_sequence, new_score))

        # Sort and select top beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # If all beams are complete, sort them and break
        if len(beams) == 0 and len(completed_beams) > 0:
            break

    if len(completed_beams) < beam_size:
        beams.sort(key=lambda x: x[1], reverse=True)

        beams = beams[:beam_size - len(completed_beams)]

        for beam_tokens, beam_score in beams:
            completed_beams.append((beam_tokens, beam_score / len(beam_tokens)))

    completed_beams.sort(key=lambda x: x[1], reverse=True)

    completed_beams = completed_beams[:beam_size]
    best_sequence, best_score = completed_beams[0]
    generated_tokens = best_sequence

    if beam_size > 1:
        # Re-cacalculate to properly obtain the attention
        torch.cuda.empty_cache()

        decoder_input_ids = torch.tensor(generated_tokens).unsqueeze(0).to(device)
        model_output = model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)

    # Attention, and gradients hook

    attentions_grad_store = {}

    assert len(model_output.encoder_attentions) == num_hidden_layers, f"{len(model_output.encoder_attentions)} != {num_hidden_layers}"
    assert len(model_output.decoder_attentions) == num_hidden_layers, f"{len(model_output.decoder_attentions)} != {num_hidden_layers}"
    assert len(model_output.cross_attentions) == num_hidden_layers, f"{len(model_output.cross_attentions)} != {num_hidden_layers}"

    attentions = {l: {
        "encoder": model_output.encoder_attentions[l],
        "decoder": model_output.decoder_attentions[l],
        "cross": model_output.cross_attentions[l],
        } for l in range(num_hidden_layers)}

    for layer in range(num_hidden_layers):
        for component in ("encoder", "decoder", "cross"):
            attentions[layer][component].requires_grad

            attentions[layer][component].retain_grad()
            attentions[layer][component].register_hook(capture_attention_grads(layer, component, attentions_grad_store))

    source_text_seq_len = len(inputs.input_ids[0])
    target_text_seq_len = len(generated_tokens)
    attention_expected_shape = {
        "encoder": (batch_size, num_attention_heads, source_text_seq_len, source_text_seq_len),
        "decoder": (batch_size, num_attention_heads, target_text_seq_len, target_text_seq_len),
        "cross": (batch_size, num_attention_heads, target_text_seq_len, source_text_seq_len),
        }

    for l in range(num_hidden_layers):
        for component in ("encoder", "decoder", "cross"):
            assert attentions[l][component].shape == attention_expected_shape[component], f"{l}: {component}: {attentions[l][component].shape} != {attention_expected_shape[component]}"

    translated_tokens = torch.tensor([generated_tokens]).to(device)

    # Calculate gradients

    logits = model_output.logits
    logits_expected_shape = (batch_size, len(generated_tokens), vocab_size)

    assert logits.shape == logits_expected_shape, f"{logits.shape} != {logits_expected_shape}"

    target = torch.zeros(logits_expected_shape)

    for i, token in enumerate(generated_tokens):
        target[:, i, token] = 1.0

    target = target.to(device)
    loss = torch.sum(logits * target)

    assert len(attentions_grad_store) == 0, f"{len(attentions_grad_store)} != 0"

    model.zero_grad()
    loss.backward(retain_graph=True)

    assert len(attentions_grad_store) == num_hidden_layers, f"{len(attentions_grad_store)} != {num_hidden_layers}"

    # Decode

    output = decode(tokenizer, translated_tokens)
    output_tokens = [tokenizer.convert_ids_to_tokens(_id) for _id in generated_tokens]

    if debug:
        for i, (translated_tokens, score) in enumerate(completed_beams, 1):
            translated_text = decode(tokenizer, [translated_tokens])[0]

            print(f"DEBUG: translation #{i} (normalized score: {score} -> probability: {torch.e ** score}): {translated_text}")

    if debug:
        from_generate = translate_from_generate(tokenizer, model, inputs, target_lang_id, max_length_decoder, beam_size=beam_size)

        assert len(from_generate) == len(output)

        for o1, o2 in zip(output, from_generate):
            if o1 != o2 and not teacher_forcing:
                print(f"DEBUG: warning: different result than using generate: {o1} vs {o2}")

        from_pipeline = translate_from_pipeline(translator_pipeline, source_text, beam_size=beam_size)

        assert len(from_pipeline) == len(output)

        for o1, o2 in zip(output, from_pipeline):
            if o1 != o2  and not teacher_forcing:
                print(f"DEBUG: warning: different result than using pipeline: {o1} vs {o2}")

    # Apply explainability
    # e -> encoder, d -> decoder (following paper nomenclature section 3.2, encoder-decoder architecture)

    r_ee = np.identity(source_text_seq_len) # encoder
    r_dd = np.identity(target_text_seq_len) # decoder
    #r_ed = np.zeros((source_text_seq_len, target_text_seq_len)) # We only have co-attention in the decoder (i.e., r_de)
    r_de = np.zeros((target_text_seq_len, source_text_seq_len)) # influence of the input text on the translated text
    a_line = {}

    for layer in range(num_hidden_layers):
        a_line[layer] = {}

        for component in ("encoder", "decoder", "cross"):
            assert component not in a_line[layer]

            gradient = attentions_grad_store[layer][component]
            attention = attentions[layer][component]

            assert gradient.shape == attention.shape, f"{gradient.shape} != {attention.shape}"

            a_line_aux = gradient * attention

            assert a_line_aux.shape == attention.shape, f"{a_line_aux.shape} != {attention.shape}"
            assert a_line_aux.shape == attention_expected_shape[component], f"{a_line_aux.shape} != {attention_expected_shape[component]}"

            a_line_aux = torch.max(a_line_aux, torch.zeros_like(a_line_aux)).cpu().detach()
            a_line_aux = torch.mean(a_line_aux, dim=1)

            a_line[layer][component] = a_line_aux

            assert a_line[layer][component].shape == (batch_size, *attention_expected_shape[component][-2:]), a_line[layer].shape

    assert len(a_line) == num_hidden_layers

    # Update encoder self attention layers

    for layer in range(num_hidden_layers):
        # Only equation 6

        component = "encoder"
        a_line_aux = a_line[layer][component][0].numpy() # current layer, and batch size is 0

        assert a_line_aux.shape == attention_expected_shape[component][-2:], a_line_aux.shape
        assert a_line_aux.shape == r_ee.shape

        r_ee += np.matmul(a_line_aux, r_ee)

    # Update decoder self- and cross-attention

    for layer in range(num_hidden_layers):
        # Code: https://github.com/hila-chefer/Transformer-MM-Explainability/blob/58eaea85ac9c34aff052f368514b35d2e4c8dd3c/DETR/modules/ExplanationGenerator.py#L142

        # Encoder attention: equations 6 and 7

        component = "decoder"
        a_line_aux = a_line[layer][component][0].numpy() # current layer, and batch size is 0

        assert a_line_aux.shape == attention_expected_shape[component][-2:], a_line_aux.shape
        assert a_line_aux.shape == r_dd.shape

        r_dd += np.matmul(a_line_aux, r_dd)
        r_de += np.matmul(a_line_aux, r_de)

        if layer == 0:
            assert (r_de == np.zeros_like(r_de)).all()
        else:
            assert (r_de != np.zeros_like(r_de)).any() # Due to cross-attention!

        # Cross-attention: equations 8, 9, 10 and 11
        component = "cross"
        a_line_aux = a_line[layer][component][0].numpy() # current layer, and batch size is 0

        assert a_line_aux.shape == attention_expected_shape[component][-2:], a_line_aux.shape

        r_dd_normalized = r_dd
        r_ee_normalized = r_ee

        if apply_normalization:
            # Equations 8 and 9
            r_dd_normalized = normalize(r_dd_normalized)
            r_ee_normalized = normalize(r_ee_normalized)

        pre_r_de_addition = np.matmul(a_line_aux, r_ee_normalized)
        r_de_addition = np.matmul(r_dd_normalized.transpose(), pre_r_de_addition)

        # Update cross-attention
        r_de += r_de_addition

    assert r_de.shape == (len(output_tokens), len(input_tokens))

    # Min-max normalization

    # Fill self-attention diagonal with near-zeros to make easier the relevance analysis of the self-attention
    if self_attention_remove_diagonal:
        np.fill_diagonal(r_ee, sys.float_info.epsilon)
        np.fill_diagonal(r_dd, sys.float_info.epsilon)

    if explainability_normalization == "none":
        pass
    elif explainability_normalization == "absolute":
        r_ee = (r_ee - r_ee.min()) / (r_ee.max() - r_ee.min())
        r_dd = (r_dd - r_dd.min()) / (r_dd.max() - r_dd.min())
        r_de = (r_de - r_de.min()) / (r_de.max() - r_de.min()) # (target_text_seq_len, source_text_seq_len)
    elif explainability_normalization == "relative":
        # "Relative" normalization (easier to analize per translated token)
        r_ee = np.array([(r_ee[i] - r_ee[i].min()) / (r_ee[i].max() - r_ee[i].min()) for i in range(len(r_ee))])
        r_dd = np.array([(r_dd[i] - r_dd[i].min()) / (r_dd[i].max() - r_dd[i].min()) for i in range(len(r_dd))])
        r_de = np.array([(r_de[i] - r_de[i].min()) / (r_de[i].max() - r_de[i].min()) for i in range(len(r_de))])
    else:
        raise Exception(f"Unexpected value for explainability_normalization: {explainability_normalization}")

    return input_tokens, output_tokens, r_ee, r_dd, r_de

if __name__ == "__main__":
    debug = True # Change manually
    source_text = sys.argv[1]
    target_text = sys.argv[2] if len(sys.argv) > 2 else '' # Teacher forcing
    source_lang = sys.argv[3] if (len(sys.argv) > 3 and len(sys.argv[3]) > 0) else "eng_Latn" # e.g., eng_Latn
    target_lang = sys.argv[4] if (len(sys.argv) > 4 and len(sys.argv[4]) > 0) else "spa_Latn" # e.g., spa_Latn
    beam_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    device = sys.argv[6] if (len(sys.argv) > 6 and len(sys.argv[6]) > 0) else None
    pickle_prefix_filename = sys.argv[7] if (len(sys.argv) > 7 and len(sys.argv[7]) > 0) else ''
    pretrained_model = sys.argv[8] if (len(sys.argv) > 8 and len(sys.argv[8]) > 0) else None

    if pickle_prefix_filename:
        debug = False
        self_attention_remove_diagonal = False
        explainability_normalization = "none"
    else:
        self_attention_remove_diagonal = True
        explainability_normalization = "relative"

    print(f"Provided args: {sys.argv}")

    if source_text == '-' or target_text == '-':
        source_text, target_text = [], []
        for l in sys.stdin:
            s, t = l.rstrip("\r\n").split('\t')

            source_text.append(s)
            target_text.append(t)
    else:
        source_text = [source_text]
        target_text = [target_text]

    print(f"Translating from {source_lang} to {target_lang}")

    pickle_data = {
                "explainability_encoder": [],
                "explainability_decoder": [],
                "explainability_cross": [],
            }
    fn_pickle_array = f"{pickle_prefix_filename}.{source_lang}.{target_lang}.pickle"

    for idx, (_source_text, _target_text) in enumerate(zip(source_text, target_text), 1):
        if not pickle_prefix_filename:
            print()
            print(f"Source text: {_source_text}")
            print(f"Target text: {_target_text}")

        input_tokens, output_tokens, r_ee, r_dd, r_de = explainability(_source_text, target_text=_target_text, source_lang=source_lang,
                                                                       target_lang=target_lang, debug=debug, beam_size=beam_size, device=device,
                                                                       self_attention_remove_diagonal=self_attention_remove_diagonal,
                                                                       explainability_normalization=explainability_normalization,
                                                                       pretrained_model=pretrained_model)

        # Print results

        if not pickle_prefix_filename:
            print()
            print("Encoder self-attention:")
            print_attention(input_tokens, input_tokens, r_ee)
            print()
            print("Decoder self-attention:")
            print_attention(output_tokens, output_tokens, r_dd)
            print()
            print("Decoder cross-attention:")
            print_attention(output_tokens, input_tokens, r_de)
        else:
            if idx % 10 == 0:
                print(f"pickle: {idx}/{len(source_text)} sentences finished")

                sys.stdout.flush()

            pickle_data["explainability_encoder"].append(r_ee)
            pickle_data["explainability_decoder"].append(r_dd)
            pickle_data["explainability_cross"].append(r_de)

    if pickle_prefix_filename:
        with open(fn_pickle_array, "wb") as pickle_fd:
            pickle.dump(pickle_data, pickle_fd)

        print(f"pickle: all data stored: {fn_pickle_array}")
