
import sys
import pickle
import functools
import collections

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange

source_lang = sys.argv[1] # e.g., en (check https://huggingface.co/facebook/m2m100_1.2B)
target_lang = sys.argv[2]
img_output = sys.argv[3] if len(sys.argv) > 3 else ''
direction = sys.argv[4] if len(sys.argv) > 4 else "src2trg"
pretrained_model = sys.argv[5] if len(sys.argv) > 5 else "facebook/nllb-200-distilled-600M"
teacher_forcing = sys.argv[6] if len(sys.argv) > 6 else '1'
pickle_prefix_filename = sys.argv[7] if len(sys.argv) > 7 else ''

assert direction in ("src2trg", "trg2src")

teacher_forcing = bool(int(teacher_forcing))

assert teacher_forcing, "At the moment, only teacher_forcing=True is supported"

print(f"Provided args: {sys.argv[1:]}")

teacher_forcing_str = "yes" if teacher_forcing else "no"

#pretrained_model = "facebook/m2m100_418M"
#pretrained_model = "facebook/nllb-200-distilled-600M"

if pretrained_model == '':
    pretrained_model = "facebook/nllb-200-distilled-600M"

assert "m2m100" in pretrained_model or "nllb-200" in pretrained_model, "Script supports M2M or NLLB MT systems"

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
model = model.to(device).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, src_lang=source_lang, tgt_lang=target_lang)
max_length = model.config.max_length
max_new_tokens = model.generation_config.max_length

assert max_new_tokens != 1000000000000000019884624838656, "Check: https://discuss.huggingface.co/t/tokenizers-what-this-max-length-number/28484"

if "m2m" in pretrained_model:
    source_lang_token = tokenizer.convert_ids_to_tokens(tokenizer.get_lang_id(source_lang))
    target_lang_token = tokenizer.convert_ids_to_tokens(tokenizer.get_lang_id(target_lang))
else:
    source_lang_token = source_lang
    target_lang_token = target_lang

if direction == "trg2src":
    source_lang_token, target_lang_token = target_lang_token, source_lang_token

#bos_token_token = tokenizer.convert_ids_to_tokens(model.generation_config.bos_token_id) # BoS token is ignored
eos_token_token = tokenizer.convert_ids_to_tokens(model.generation_config.eos_token_id)
decoder_start_token_token = tokenizer.convert_ids_to_tokens(model.generation_config.decoder_start_token_id)
bsz = 1

assert bsz == 1, "Current supported batch size is 1"

def trace_forward(src_inputs, trg_inputs):
    with torch.no_grad():
        layer_inputs = collections.defaultdict(list)
        layer_outputs = collections.defaultdict(list)

        def save_activation(name, mod, inp, out):
            layer_inputs[name].append(inp)
            layer_outputs[name].append(out)

        handles = {}

        for name, layer in model.named_modules():
            handles[name] = layer.register_forward_hook(functools.partial(save_activation, name))

        # TODO use parameter for model
        model_output = model(**src_inputs, decoder_input_ids=trg_inputs["input_ids"], output_attentions=True, output_hidden_states=True) # odict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions'])

        for k, v in handles.items():
            handles[k].remove()
        
        return None, None, model_output, layer_inputs, layer_outputs

def parse_module_name(module_name):
    """ Returns (enc_dec, layer, module)"""
    parsed_module_name = module_name.split('.')
    if not isinstance(parsed_module_name, list):
        parsed_module_name = [parsed_module_name]
        
    if len(parsed_module_name) < 1 or len(parsed_module_name) > 3:
        raise AttributeError(f"'{module_name}' unknown")
        
    if len(parsed_module_name) > 1:
        try:
            parsed_module_name[1] = int(parsed_module_name[1])
        except ValueError:
            parsed_module_name.insert(1, None)
        if len(parsed_module_name) < 3:
            parsed_module_name.append(None)
    else:
        parsed_module_name.extend([None, None])

    return parsed_module_name

def get_module(module_name):
    e_d, l, m = parse_module_name(module_name)
    # TODO use parameter for model
    module = getattr(model.model, e_d)
    if l is not None:
        module = module.layers[l]
        if m is not None:
            module = getattr(module, m)
    else:
        if m is not None:
            raise AttributeError(f"Cannot get'{module_name}'")

    return module

def __get_attn_weights_module(layer_outputs, model_output, module_name):
    enc_dec_, l, attn_module_ = parse_module_name(module_name)
    attn_module = get_module(module_name)
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    k = layer_outputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}.k_proj"][0]
    q = layer_outputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0] * head_dim ** -0.5

    q, k = map(
        lambda x: rearrange(
            x,
            #'t b (n_h h_d) -> (b n_h) t h_d',
            'b t (n_h h_d) -> (b n_h) t h_d',
            n_h=num_heads,
            h_d=head_dim
        ),
        (q, k)
    )

    attn_weights = torch.bmm(q, k.transpose(1, 2)) # (b n_h) t_q t_k

    if enc_dec_ == 'decoder' and attn_module_ == 'self_attn':
        tri_mask = torch.triu(torch.ones_like(attn_weights), 1).bool()
        attn_weights[tri_mask] = -1e9

    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_weights = rearrange(
        attn_weights,
        '(b n_h) t_q t_k -> b n_h t_q t_k',
        n_h=num_heads
    )

    # assert attn weights
    attn_weights_model = model_output[f"{enc_dec_}_attentions"][l]
    cross_attention = enc_dec_ == "decoder" and attn_module_ == "encoder_attn"

    if not cross_attention:
        assert attn_weights.shape == attn_weights_model.shape

        if torch.dist(attn_weights_model, attn_weights).item() >= 1e-3 * attn_weights_model.numel():
            a = torch.dist(attn_weights_model, attn_weights).item()
            b = attn_weights_model.numel()

            raise Exception(f"{a} >= 1e-3 * {b} = {1e-3 * b}\n{attn_weights_model}\n{attn_weights}")

    return attn_weights

def __get_contributions_module(layer_inputs, layer_outputs, contrib_type, model_output, module_name, pre_layer_norm=True):
    # Get info about module: encoder, decoder, self_attn, cross-attn
    enc_dec_, l, attn_module_ = parse_module_name(module_name)
    attn_w = __get_attn_weights_module(layer_outputs, model_output, module_name) # (batch_size, num_heads, src_len, src_len)

    def l_transform(x, w_ln):
        '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
        ln_param_transf = torch.diag(w_ln)
        ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
            1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

        out = torch.einsum(
            '... e , e f , f g -> ... g',
            x,
            ln_mean_transf,
            ln_param_transf
        )
        return out

    attn_module = get_module(module_name)
    w_o = attn_module.out_proj.weight
    b_o = attn_module.out_proj.bias
    
    ln = get_module(f'{module_name}_layer_norm')
    w_ln = ln.weight.data
    b_ln = ln.bias
    eps_ln = ln.eps

    #in_q = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0][0].transpose(0, 1)
    #in_v = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}.v_proj"][0][0].transpose(0, 1)
    #in_res = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)
    in_q = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0][0]
    in_v = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}.v_proj"][0][0]
    in_res = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0]


    if "self_attn" in attn_module_:
        if pre_layer_norm:
            residual_ = torch.einsum('sk,bsd->bskd', torch.eye(in_res.size(1)).to(in_res.device), in_res)
        else:
            residual_ = torch.einsum('sk,bsd->bskd', torch.eye(in_q.size(1)).to(in_res.device), in_q)
    else:
        if pre_layer_norm:
            residual_ = in_res
        else:
            residual_ = in_q

    v = attn_module.v_proj(in_v)
    v = rearrange(
        v,
        'b t_v (n_h h_d) -> b n_h t_v h_d',
        n_h=attn_module.num_heads,
        h_d=attn_module.head_dim
    )

    w_o = rearrange(
        w_o,
        #'out_d (n_h h_d) -> n_h h_d out_d', # TODO 'out_d (n_h h_d)' or '(n_h h_d) out_d'? If different from current value, check attn_v_wo einsum
        'out_d (n_h h_d) -> n_h h_d out_d',
        n_h=attn_module.num_heads,
    )

    attn_v_wo = torch.einsum(
        'b h q k , b h k e , h e f -> b q k f',
        attn_w,
        v,
        w_o
    )

    # Add residual
    if "self_attn" in attn_module_:
        out_qv_pre_ln = attn_v_wo + residual_
    # Concatenate residual in cross-attention (as another value vector)
    else:
        out_qv_pre_ln = torch.cat((attn_v_wo,residual_.unsqueeze(-2)),dim=2)
    
    # Assert MHA output + residual is equal to 1st layer normalization input
    out_q_pre_ln = out_qv_pre_ln.sum(-2) + b_o

    #### NEW
    if pre_layer_norm:
        if 'encoder' in enc_dec_:
            # Encoder (self-attention) -> final_layer_norm
            #out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
            out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.final_layer_norm"][0][0]
        else:
            if "self_attn" in attn_module_:
                # Self-attention decoder -> encoder_attn_layer_norm
                #out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"][0][0].transpose(0, 1)
                out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"][0][0]
            else:
                # Cross-attention decoder -> final_layer_norm
                #out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
                out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.final_layer_norm"][0][0]
    else:
        # In post-ln we compare with the input of the first layernorm
        #out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)
        out_q_pre_ln_th = layer_inputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0]

    if torch.dist(out_q_pre_ln_th, out_q_pre_ln).item() >= 1e-3 * out_q_pre_ln.numel():
        a = torch.dist(out_q_pre_ln_th, out_q_pre_ln).item()
        b = out_q_pre_ln.numel()

        raise Exception(f"{out_q_pre_ln_th.shape} {out_q_pre_ln.shape} | {a} >= 1e-3 * {b} = {1e-3 * b}\n{out_q_pre_ln_th}\n{out_q_pre_ln}")
    
    if pre_layer_norm:
        transformed_vectors = out_qv_pre_ln
        resultant = out_q_pre_ln
    else:
        ln_std_coef = 1/(out_q_pre_ln_th + eps_ln).std(-1).view(1,-1, 1).unsqueeze(-1) # (batch,src_len,1,1)
        transformed_vectors = l_transform(out_qv_pre_ln, w_ln)*ln_std_coef # (batch,src_len,tgt_len,embed_dim)
        dense_bias_term = l_transform(b_o, w_ln)*ln_std_coef # (batch,src_len,1,embed_dim)
        attn_output = transformed_vectors.sum(dim=2) # (batch,seq_len,embed_dim)
        resultant = attn_output + dense_bias_term.squeeze(2) + b_ln # (batch,seq_len,embed_dim)

        # Assert resultant (decomposed attention block output) is equal to the real attention block output
        #out_q_th_2 = layer_outputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0].transpose(0, 1)
        out_q_th_2 = layer_outputs[f"model.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0]
        assert torch.dist(out_q_th_2, resultant).item() < 1e-3 * resultant.numel()


    if contrib_type == 'l1':
        contributions = -F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2), p=1)
        resultants_norm = torch.norm(torch.squeeze(resultant),p=1,dim=-1)

    elif contrib_type == 'l2':
        contributions = -F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2), p=2)
        resultants_norm = torch.norm(torch.squeeze(resultant),p=2,dim=-1)
        #resultants_norm=None
    elif contrib_type == 'koba':
        contributions = torch.norm(transformed_vectors, p=2, dim=-1)
        return contributions, None
    else:
        raise Exception(f"contribution_type '{contrib_type}' unknown")

    return contributions, resultants_norm

def normalize_contrib(x, mode=None, temperature=0.5, resultant_norm=None):
    """ Normalization applied to each row of the layer-wise contributions."""
    if mode == 'min_max':
        # Min-max normalization
        x_min = x.min(-1, keepdim=True)[0]
        x_max = x.max(-1, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
    elif mode == 'max_min':
        x = -x
        # Min-max normalization
        x_min = x.min(-1, keepdim=True)[0]
        x_max = x.max(-1, keepdim=True)[0]
        x_norm = (x_max - x) / (x_max - x_min)
        #x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
    elif mode == 'softmax':
        # Softmax
        x_norm = F.softmax(x / temperature, dim=-1)
    elif mode == 'sum_one':
        # Sum one
        x_norm = x / x.sum(dim=-1, keepdim=True)
    elif mode == 'min_sum':
        # Minimum value selection
        if resultant_norm == None:
            x_min = x.min(-1, keepdim=True)[0]
            x_norm = x + torch.abs(x_min)
            x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        else:
            x_norm = x + torch.abs(resultant_norm.unsqueeze(1))
            x_norm = torch.clip(x_norm,min=0)
            x_norm = x_norm / x_norm.sum(dim=-1,keepdim=True)
    elif mode is None:
        x_norm = x
    else:
        raise AttributeError(f"Unknown normalization mode '{mode}'")
    return x_norm

ATTN_MODULES = ['encoder.self_attn',
                'decoder.self_attn',
                'decoder.encoder_attn']

def get_contributions(src_inputs, trg_inputs, contrib_type='l1', norm_mode='min_sum', trace_forward_out=None):
    r"""
    Get contributions for each ATTN_MODULE: 'encoder.self_attn', 'decoder.self_attn', 'decoder.encoder_attn.
    Args:
        src_tensor ('tensor' ()):
            Source sentence tensor.
        tgt_tensor ('tensor' ()):
            Target sentence tensor (teacher forcing).
        contrib_type ('str', defaults to 'l1' (Ferrando et al ., 2022)):
            Type of layer-wise contribution measure: 'l1', 'l2', 'koba' (Kobayashi et al ., 2021) or 'attn_w'.
        norm_mode ('str', defaults to 'min_sum' (Ferrando et al ., 2022)):
            Type of normalization applied to layer-wise contributions: 'min_sum', 'min_max', 'sum_one', 'softmax'.
    Returns:
        Dictionary with elements in ATTN_MODULE as keys, and tensor with contributions (batch_size, num_layers, src_len, tgt_len) as values.
    """
    contributions_all = collections.defaultdict(list)

    if trace_forward_out is None:
        _, _, model_output, layer_inputs, layer_outputs = trace_forward(src_inputs, trg_inputs)
        trace_forward_out = (None, None, model_output, layer_inputs, layer_outputs)
    else:
        _, _, model_output, layer_inputs, layer_outputs = trace_forward_out
    
    if contrib_type == 'attn_w':
        f = functools.partial(__get_attn_weights_module, layer_outputs, model_output)
    else:
        f = functools.partial(
            __get_contributions_module,
            layer_inputs,
            layer_outputs,
            contrib_type,
            model_output
            )

    for attn in ATTN_MODULES:
        enc_dec_, _, attn_module_ = parse_module_name(attn)
        enc_dec = get_module(enc_dec_)

        for l in range(len(enc_dec.layers)):
            if contrib_type == 'attn_w':
                contributions = f(attn.replace('.', f'.{l}.'))
                resultant_norms = None
                contributions = contributions.sum(1)
                if norm_mode != 'sum_one':
                    print('Please change the normalization mode to sum one')
            else:
                contributions, resultant_norms = f(attn.replace('.', f'.{l}.'))
            contributions = normalize_contrib(contributions, norm_mode, resultant_norm=resultant_norms).unsqueeze(1)
            # Mask upper triangle of decoder self-attention matrix (and normalize)
            # if attn == 'decoder.self_attn':
            #     contributions = torch.tril(torch.squeeze(contributions,dim=1))
            #     contributions = contributions / contributions.sum(dim=-1, keepdim=True)
            #     contributions = contributions.unsqueeze(1)
            contributions_all[attn].append(contributions)

    contributions_all = {k: torch.cat(v, dim=1) for k, v in contributions_all.items()}
    return trace_forward_out, contributions_all

def get_contribution_rollout(src_tensor, tgt_tensor, contrib_type='l1', norm_mode='min_sum', **contrib_kwargs):
    # c = self.get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode, **contrib_kwargs)
    # if contrib_type == 'attn_w':
    #     c = {k: v.sum(2) for k, v in c.items()}
    
    # Rollout encoder (ALTI)
    def compute_joint_attention(att_mat):
        """ Compute attention rollout given contributions or attn weights + residual."""

        joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)

        layers = joint_attentions.shape[0]

        joint_attentions = att_mat[0].unsqueeze(0)

        for i in range(1,layers):

            C_roll_new = torch.matmul(att_mat[i],joint_attentions[i-1])

            joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)
            
        return joint_attentions

    c_roll = collections.defaultdict(list)
    enc_sa = 'encoder.self_attn'

    # Compute contributions rollout encoder self-attn
    trace_forward_out, contributions_all = get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode=norm_mode)
    enc_self_attn_contributions = torch.squeeze(contributions_all[enc_sa])
    layers, _, _ = enc_self_attn_contributions.size()
    enc_self_attn_contributions_mix = compute_joint_attention(enc_self_attn_contributions)
    c_roll[enc_sa] = enc_self_attn_contributions_mix.detach().clone()
    # repeat num_layers times

    # Get last layer relevances w.r.t input
    relevances_enc_self_attn = enc_self_attn_contributions_mix[-1]
    relevances_enc_self_attn = relevances_enc_self_attn.unsqueeze(0).repeat(layers, 1, 1)
        
    def rollout(C, C_enc_out):
        """ Contributions rollout whole Transformer-NMT model.
            Args:
                C: [num_layers, cross_attn;self_dec_attn] contributions decoder layers
                C_enc_out: encoder rollout last layer
        """
        src_len = C.size(2) - C.size(1)
        tgt_len = C.size(1)

        C_sa_roll = C[:, :, -tgt_len:]     # Self-att decoder, only has 1 layer
        C_ed_roll = torch.einsum(          # encoder rollout*cross-attn
            "lie , ef -> lif",
            C[:, :, :src_len],             # Cross-att
            C_enc_out                      # Encoder rollout
        )

        C_roll = torch.cat([C_ed_roll, C_sa_roll], dim=-1) # [(cross_attn*encoder rollout);self_dec_attn]
        C_roll_new_accum = C_roll[0].unsqueeze(0)

        for i in range(1, len(C)):
            C_sa_roll_new = torch.einsum(
                "ij , jk -> ik",
                C_roll[i, :, -tgt_len:],   # Self-att dec
                C_roll_new_accum[i-1, :, -tgt_len:], # Self-att (prev. roll)
            )
            C_ed_roll_new = torch.einsum(
                "ij , jk -> ik",
                C_roll[i, :, -tgt_len:],  # Self-att dec
                C_roll_new_accum[i-1, :, :src_len], # Cross-att (prev. roll)
            ) + C_roll[i, :, :src_len]    # Cross-att

            C_roll_new = torch.cat([C_ed_roll_new, C_sa_roll_new], dim=-1)
            C_roll_new = C_roll_new / C_roll_new.sum(dim=-1,keepdim=True)
            
            C_roll_new_accum = torch.cat([C_roll_new_accum, C_roll_new.unsqueeze(0)], dim=0)
            

        return C_roll_new_accum

    dec_sa = 'decoder.self_attn'
    dec_ed = 'decoder.encoder_attn'
    
    # Compute joint cross + self attention
    self_dec_contributions = torch.squeeze(contributions_all[dec_sa])
    cross_contributions = torch.squeeze(contributions_all[dec_ed])
    self_dec_contributions = (self_dec_contributions.transpose(1,2)*cross_contributions[:,:,-1].unsqueeze(1)).transpose(1,2)
    joint_self_cross_contributions = torch.cat((cross_contributions[:,:,:-1],self_dec_contributions),dim=-1)

    contributions_full_rollout = rollout(joint_self_cross_contributions, relevances_enc_self_attn[-1])

    c_roll['total'] = contributions_full_rollout
    c_roll[dec_ed] = cross_contributions

    return c_roll, trace_forward_out

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
import numpy as np

def contrib_tok2words_partial(contributions, tokens, axis, reduction):
    from string import punctuation

    reduction_fs = {
        'avg': np.mean,
        'sum': np.sum
    }

    words = []
    w_contributions = []
    for counter, (tok, contrib) in enumerate(zip(tokens, contributions.T)):
        if tok.startswith('▁') or tok.startswith('__') or tok.startswith('<') or counter==0:# or tok in punctuation:
            if tok.startswith('▁'):
                tok = tok[1:]
            words.append(tok)
            w_contributions.append([contrib])
        else:
            words[-1] += tok
            w_contributions[-1].append(contrib)

    reduction_f = reduction_fs[reduction]
    word_contrib = np.stack([reduction_f(np.stack(contrib, axis=axis), axis=axis) for contrib in w_contributions], axis=axis)

    return word_contrib, words

def contrib_tok2words(contributions, tokens_in, tokens_out):
    word_contrib, words_in = contrib_tok2words_partial(contributions, tokens_in, axis=0, reduction='sum')
    word_contrib, words_out = contrib_tok2words_partial(word_contrib, tokens_out, axis=1, reduction='avg')
    return word_contrib.T, words_in, words_out

def visualize_alti(img_output, total_alti, source_sentence, target_sentence, predicted_sentence, word_level, alignment, figsize=(17, 24), all_layers=False):

    def plot_heatmap_alti(_img_output, contributions_rollout_layer_np, source_sentence_, target_sentence_, predicted_sentence_):

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=figsize, dpi=200)
        gs = GridSpec(3, 6)
        gs.update(wspace=1, hspace=0.05)
        ax_main = plt.subplot(gs[0:3, :5])
        ax_yDist = plt.subplot(gs[1, 5])
        
        df = pd.DataFrame(contributions_rollout_layer_np, columns = source_sentence_ + target_sentence_, index = predicted_sentence_)

        sns.set(font_scale=1.2)
        sns.heatmap(df,cmap="Blues",square=True,ax=ax_main,cbar=False)
        ax_main.axvline(x = len(source_sentence_)-0.02, lw=1.5, linestyle = '--', color = 'grey')
        ax_main.set_xlabel('Source sentence | Target prefix', fontsize=17)
        ax_main.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)
        ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=60)
        #ax_main.set_title('Layer ' + str(layer+1))

        src_contribution = contributions_rollout_layer_np[:, :len(source_sentence_)].sum(-1)
        df_src_contribution = pd.DataFrame(src_contribution, columns = ['src_contribution'], index = predicted_sentence_)

        ax_yDist.barh(range(0, len(predicted_sentence_)), df_src_contribution.src_contribution, align='center')
        plt.yticks(ticks = range(0, len(predicted_sentence_)) ,labels = predicted_sentence_,fontsize='14')
        plt.gca().invert_yaxis()
        ax_yDist.grid(True, linestyle=(0, (5, 10)))
        ax_yDist.set_xlim(0,1)
        ax_yDist.spines['top'].set_visible(False)
        ax_yDist.spines['right'].set_visible(False)
        ax_yDist.spines['bottom'].set_visible(False)
        ax_yDist.spines['left'].set_visible(False)
        ax_yDist.xaxis.set_ticks_position("bottom")
        ax_yDist.set_title('Source contribution')

        fig.savefig(_img_output)
    
    if all_layers:

        for layer in range(0, total_alti.shape[0]):

            contributions_rollout_layer = total_alti[layer]
            contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()
                
            if word_level:
                if alignment:
                    tokens_out = target_sentence[1:] + ['</s>']
                else:
                    tokens_out = predicted_sentence
                contributions_rollout_layer_np, words_in, words_out = contrib_tok2words(
                    contributions_rollout_layer_np,
                    tokens_in=(source_sentence + target_sentence),
                    tokens_out=tokens_out
                )
            source_sentence_ = words_in[:words_in.index('</s>')+1] if word_level else source_sentence
            target_sentence_ = words_in[words_in.index('</s>')+1:] if word_level else target_sentence
            predicted_sentence_ = words_out if word_level else predicted_sentence


            plot_heatmap_alti(f"{img_output}.layer_{layer}.png", contributions_rollout_layer_np, source_sentence_,
                                target_sentence_, predicted_sentence_)

    else:
        layer = -1
        contributions_rollout_layer = total_alti[layer]
        contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()
            
        if word_level:
            if alignment:
                tokens_out = target_sentence[1:] + ['</s>']
            else:
                tokens_out = predicted_sentence
            contributions_rollout_layer_np, words_in, words_out = contrib_tok2words(
                contributions_rollout_layer_np,
                tokens_in=(source_sentence + target_sentence),
                tokens_out=tokens_out
            )
        source_sentence_ = words_in[:words_in.index('</s>')+1] if word_level else source_sentence
        target_sentence_ = words_in[words_in.index('</s>')+1:] if word_level else target_sentence
        predicted_sentence_ = words_out if word_level else predicted_sentence

        plot_heatmap_alti(img_output, contributions_rollout_layer_np, source_sentence_,
                            target_sentence_, predicted_sentence_)

    return contributions_rollout_layer_np, source_sentence_, predicted_sentence_

#word_level = True
word_level = False
alignment = False

pickle_data = {
    "explainability_alti_plus_encoder": [],
    "explainability_alti_plus_decoder": [],
    "explainability_alti_plus_cross": [],
    "explainability_alti_plus_total": [],
}

# Teacher forcing decoding
for idx, l in enumerate(sys.stdin, 1):
    _src, _trg = l.rstrip("\r\n").split('\t')

    if direction == "trg2src":
        _src, _trg = _trg, _src

    src = source_lang_token + _src + eos_token_token # Note BoS is not used
    src_inputs = tokenizer([src], return_tensors="pt", add_special_tokens=False, truncation=True, padding=True).to(device)
    src_inputs["input_ids"] = src_inputs["input_ids"][:,:max_length]
    src_inputs["attention_mask"] = src_inputs["attention_mask"][:,:max_length]
    #trg = decoder_start_token_token + target_lang_token + _trg + eos_token_token
    # TODO if not teacher forcing, _trg must contain the decoded tokens
    trg = decoder_start_token_token + target_lang_token + _trg
    trg_inputs = tokenizer([trg], return_tensors="pt", add_special_tokens=False, truncation=True, padding=True).to(device)
    trg_inputs["input_ids"] = trg_inputs["input_ids"][:,:max_new_tokens]
    trg_inputs["attention_mask"] = trg_inputs["attention_mask"][:,:max_new_tokens]
    #_, _, model_output, layer_inputs, layer_outputs = trace_forward(src_inputs, trg_inputs)
    #_, _, model_output, layer_inputs, layer_outputs = get_contributions(src_inputs, trg_inputs)
    relevances, trace_forward_out = get_contribution_rollout(src_inputs, trg_inputs)
    _, _, model_output, _, _ = trace_forward_out
    logits = model_output["logits"]
    src_tokens_shape = src_inputs['input_ids'].shape
    trg_tokens_shape = trg_inputs['input_ids'].shape

    print(f"Info: {idx}: logits {logits.shape}, src tokens {src_tokens_shape}, trg tokens {trg_tokens_shape}")

    log_probs = F.log_softmax(logits, dim=-1).cpu()
    token_ids = torch.topk(-log_probs, 1, largest=False, sorted=True).indices.squeeze(0).tolist() # get minimum (instead of maximum due to log) values (i.e., tokens)

    assert len(token_ids) == logits.shape[1]
    assert len(token_ids[0]) == 1
    assert isinstance(token_ids[0][0], int)

    #for k, v in relevances.items():
    #    print(f"{k}: {v.shape}\n{v}\n")

#    model_output = model(**src_inputs, decoder_input_ids=trg_inputs["input_ids"], output_attentions=True, output_hidden_states=True) # odict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions'])
#
#    #print(f"Debug info:\nmodel_output.keys(): {model_output.keys()}\nlayer_inputs: {layer_inputs}\nlayer_outputs: {layer_outputs}\n")
#
#    assert (model_output["encoder_hidden_states"][-1] == model_output["encoder_last_hidden_state"]).all().detach().cpu().item()
#
#    src_seq_len = src_inputs["input_ids"].shape[-1]
#    trg_seq_len = trg_inputs["input_ids"].shape[-1]
#
#    for module, seq_len, num_heads in (("encoder", (src_seq_len, src_seq_len), model.config.encoder_attention_heads),
#                                       ("decoder", (trg_seq_len, trg_seq_len), model.config.decoder_attention_heads),
#                                       ("cross",   (trg_seq_len, src_seq_len), model.config.decoder_attention_heads),):
#        for layer in range(len(model_output[f"{module}_attentions"])):
#            s = model_output[f"{module}_attentions"][layer].shape # (A_i,j^h)
#
#            assert s == (bsz, num_heads, *seq_len), f"{module}_attentions: {s}"
#
#    print(f"src tokens: {src_inputs}")
#    print(f"trg tokens: {trg_inputs}")
#
    src_inputs_token_ids_decoded = [tokenizer.convert_ids_to_tokens(token.item()) for token in src_inputs["input_ids"][0].detach().cpu()]
    trg_inputs_token_ids_decoded = [tokenizer.convert_ids_to_tokens(token.item()) for token in trg_inputs["input_ids"][0].detach().cpu()]
    predicted_sentence = [tokenizer.convert_ids_to_tokens(token[0]) for token in token_ids]
#
#    print(f"src: {src_inputs_token_ids_decoded}")
#    print(f"trg: {trg_inputs_token_ids_decoded}")

    source_sentence = src_inputs_token_ids_decoded
    target_sentence = trg_inputs_token_ids_decoded
    #predicted_sentence = target_sentence[1:] + [eos_token_token] # teacher forcing

    if img_output:
        final_img_output = f"{img_output}.idx_{idx}.png"
        alti_result, source_sentence_, predicted_sentence_ = visualize_alti(final_img_output, relevances["total"], source_sentence,
                                                                            [target_sentence[0]] + ['▁' + target_sentence[1]] + target_sentence[2:], predicted_sentence,
                                                                            word_level, alignment, all_layers = False)

        print(f"Plot output: {final_img_output}")

    last_layer_alti_plus = relevances["total"][-1].cpu().detach().numpy()

    assert last_layer_alti_plus.shape == (trg_tokens_shape[-1], src_tokens_shape[-1] + trg_tokens_shape[-1]), idx

    last_layer_alti_plus_cross_contributions = last_layer_alti_plus[:,:src_tokens_shape[-1]]
    last_layer_alti_plus_decoder_contributions = last_layer_alti_plus[:,src_tokens_shape[-1]:]

    assert (np.concatenate((last_layer_alti_plus_cross_contributions, last_layer_alti_plus_decoder_contributions), 1) == last_layer_alti_plus).all().item(), idx

    last_layer_alti_encoder_contributions = relevances["encoder.self_attn"][-1].cpu().detach().numpy() # ALTI

    assert last_layer_alti_plus_cross_contributions.shape == (trg_tokens_shape[-1], src_tokens_shape[-1]), idx
    assert last_layer_alti_plus_decoder_contributions.shape == (trg_tokens_shape[-1], trg_tokens_shape[-1]), idx
    assert last_layer_alti_encoder_contributions.shape == (src_tokens_shape[-1], src_tokens_shape[-1]), idx

    if pickle_prefix_filename:
        pickle_data["explainability_alti_plus_encoder"].append(last_layer_alti_encoder_contributions)
        pickle_data["explainability_alti_plus_decoder"].append(last_layer_alti_plus_decoder_contributions)
        pickle_data["explainability_alti_plus_cross"].append(last_layer_alti_plus_cross_contributions)
        pickle_data["explainability_alti_plus_total"].append(last_layer_alti_plus)

if pickle_prefix_filename:
    fn_pickle_array = f"{pickle_prefix_filename}.alti_plus.{direction}.{source_lang}.{target_lang}.teacher_forcing_{teacher_forcing_str}.pickle"

    with open(fn_pickle_array, "wb") as pickle_fd:
        pickle.dump(pickle_data, pickle_fd)

    print(f"pickle: all data stored: {fn_pickle_array}")
