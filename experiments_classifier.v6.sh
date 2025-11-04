#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    # Kill all child processes of the current script
    pkill --signal SIGTERM -P $$ # SIGINT does not work...

    exit -1
}

# Trap SIGINT (Ctrl+C) and SIGTERM signals to call the cleanup function
trap cleanup SIGINT SIGTERM

# https://arxiv.org/pdf/2305.19757
## Experiment 2
## Results in table 3
## Hyper-parameters: table 9 and https://github.com/Malina03/macocu-ht-vs-mt/ (https://github.com/Malina03/macocu-ht-vs-mt/tree/main/experiments/7)

seeds="71213 42 43"
default_args="--verbose --strategy epoch --train-until-patience --patience 6 --epochs 10 --batch-size 32 --classifier-dropout 0.1 --lr-scheduler inverse_sqrt_chichirau_et_al --lr-scheduler-args 400 --optimizer adamw_no_wd --learning-rate 1e-5"
default_args_explainability="--verbose --learning-rate 1e-04 --train-until-patience --epochs 10 --dropout 0.1 --lr-scheduler inverse_sqrt_chichirau_et_al --lr-scheduler-args 400 --optimizer adamw_no_wd --num-layers 3 --num-attention-heads 4 --lm-classifier-dropout 0.1"
#workers_init="10"
#workers_init="5"
workers_init="1"
njobs="0"
patience_metric="$1"

if [[ -z "$patience_metric" ]]; then
    patience_metric="acc"
fi

for seed in $(echo "$seeds"); do

output="./results_v2.seed_${seed}/reproduction"

mkdir -p $(dirname "$output")

echo "$(date) current output: $output"

#for l in $(echo "de-es es-de de-en zh-en ru-en en-de en-zh en-ru all"); do
for l in $(echo "de-es es-de de-en ru-en en-de en-ru all"); do
for m in $(if [[ "$l" == "all" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP_opus-mt-${l} all_balanced"; elif [[ "$l" == "de-en" ]] || [[ "$l" == "zh-en" ]] || [[ "$l" == "ru-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 all_balanced"; else echo "Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP_opus-mt-${l} all_balanced"; fi); do
    mfn=$(echo "$m" | tr '/' '_')
    data_fn_suffix=".${mfn}.out.all.shuf"
    f_prefix_cwd="./wmt_data/${l}-mtpaper.all.sentences.shuf"
    if [[ ! -f "./${f_prefix_cwd}.train${data_fn_suffix}" ]] || [[ ! -f "./${f_prefix_cwd}.dev${data_fn_suffix}" ]] || [[ ! -f "./${f_prefix_cwd}.test${data_fn_suffix}" ]]; then
        echo "ERROR: not all exist: ./${f_prefix_cwd}.{train,dev,test}${data_fn_suffix}"
        continue
    fi

    for mono_or_bilingual_data in $(if [[ "$l" == "all" ]] || [[ "$m" == "all_balanced" ]]; then echo "bilingual:0"; else echo "monolingual:0 bilingual:0"; fi); do
        mono_or_bilingual=$(echo "$mono_or_bilingual_data" | cut -d: -f1)
        swap_lang=$(echo "$mono_or_bilingual_data" | cut -d: -f2)
        extra_args=""

        if [[ "$mono_or_bilingual" == "monolingual" ]]; then
            extra_args="--monolingual $extra_args"
        fi
        if [[ "$swap_lang" == "1" ]]; then
            extra_args="--swap-lang $extra_args"
        fi

        for lm in $(echo "microsoft/mdeberta-v3-base"); do
            lmfn=$(echo "${lm}_${l}-mtpaper" | tr '/' '_')
            prefix="${output}_exp2_${mono_or_bilingual}_${lmfn}_${mfn}"
            prefix=$(echo "$prefix" | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')

            if [[ "$swap_lang" == "1" ]]; then
                prefix="${prefix}.swap_lang"
            fi

            if [[ "$patience_metric" != "acc" ]]; then
                prefix="${prefix}_pm_${patience_metric}"
            fi

            if [[ -f "${prefix}.out" ]]; then
                echo "Already exists: ${prefix}.out"
                continue
            fi

            echo "$(date) check: ${prefix}.out"

            mkdir -p "${prefix}.model"

            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python smatd-lm-baseline ./${f_prefix_cwd}.{train,dev,test}${data_fn_suffix} \
                --pretrained-model "$lm" --model-output "${prefix}.model" --dev-patience-metric "$patience_metric" --seed "$seed" $default_args $extra_args \
            &> "${prefix}.out"

            rm "${prefix}.model/mtd_epoch"*.pt
        done
    done
done
done
done
wait

seeds2="71213 42 43"

for seed in $(echo "$seeds2"); do
output="./results_v6.nllb_hidden_state.seed_${seed}/exp"

mkdir -p $(dirname "$output")

echo "$(date) current output: $output"

#for lp in $(echo "all:UNK:UNK de-en:deu_Latn:eng_Latn ru-en:rus_Cyrl:eng_Latn zh-en:zho_Hans:eng_Latn es-de:spa_Latn:deu_Latn en-zh:eng_Latn:zho_Hans en-ru:eng_Latn:rus_Cyrl de-es:deu_Latn:spa_Latn en-de:eng_Latn:deu_Latn"); do
for lp in $(echo "all:UNK:UNK de-en:deu_Latn:eng_Latn ru-en:rus_Cyrl:eng_Latn en-de:eng_Latn:deu_Latn en-ru:eng_Latn:rus_Cyrl de-es:deu_Latn:spa_Latn es-de:spa_Latn:deu_Latn"); do
thelang=$(echo "$lp" | cut -d: -f1)
nllb_l1=$(echo "$lp" | cut -d: -f2)
nllb_l2=$(echo "$lp" | cut -d: -f3)
mdeberta_pretrained_path="./results_v2.seed_{seed}/reproduction_exp2_bilingual_microsoft_mdeberta-v3-base_${thelang}-mtpaper_{template}.model/mtd_best_dev.pt"
mdeberta_pretrained_path_log="./results_v2.seed_{seed}/reproduction_exp2_bilingual_microsoft_mdeberta-v3-base_${thelang}-mtpaper_{template}.out"

for nlayerscat in $(echo "0"); do
for explainability_model in $(echo "facebook/nllb-200-3.3B"); do
    explainability_model_fn=$(echo "$explainability_model" | tr '/' '_')
    f_prefix_cwd="./wmt_data/${thelang}-mtpaper.all.sentences.shuf"

    for lm in $(echo "::: microsoft/mdeberta-v3-base:$mdeberta_pretrained_path:1:0"); do # smatd and smatd+lm
        lm_pretrained_model=$(echo "$lm" | cut -d: -f1)
        lm_model_input_str=$(echo "$lm" | cut -d: -f2)
        lm_frozen_params=$(echo "$lm" | cut -d: -f3)
        our_model_input_str=$(echo "$lm" | cut -d: -f4)

        if [[ ! -z "$lm_model_input_str" ]]; then
            lm_model_input_str="yes"
        fi

        if [[ "$lm" != ":::" ]]; then
            workers="1"
            if [[ "$workers_init" -gt "0" ]] && [[ "$lm_frozen_params" == "1" ]] && [[ "$our_model_input_str" != "1" ]]; then
                #workers="2"
                workers="1"
            fi
            if [[ "$njobs" -gt "0" ]]; then
                echo "Waiting... (lm provided)"
                wait
            fi
            njobs="0"
        else
            workers="$workers_init"
        fi
        if [[ "$nlayerscat" -gt "0" ]]; then
            workers=$((workers-nlayerscat-1))
        fi
        if [[ "$workers" -le "0" ]]; then
            workers="1"
            if [[ "$workers_init" -gt "0" ]] && [[ "$lm_frozen_params" == "1" ]] && [[ "$our_model_input_str" != "1" ]]; then
                workers="1"
            fi
        fi
    for direction in $(echo "src2trg"); do
    for t in $(if [[ "$thelang" == "all" ]]; then echo "all_balanced google deepl Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP/opus-mt-${thelang}"; elif [[ "$thelang" == "de-en" ]] || [[ "$thelang" == "zh-en" ]] || [[ "$thelang" == "ru-en" ]]; then echo "all_balanced google deepl Unbabel/TowerInstruct-7B-v0.2"; else echo "all_balanced Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP/opus-mt-${thelang}"; fi); do
    t=$(echo "$t" | tr '/' '_')
    last_layer=$(if [[ "$explainability_model" == "facebook/nllb-200-distilled-600M" ]]; then echo "-13"; elif [[ "$explainability_model" == "facebook/nllb-200-1.3B" ]] || [[ "$explainability_model" == "facebook/nllb-200-3.3B" ]]; then echo "-25"; elif [[ "$explainability_model" == "Helsinki-NLP/opus-mt-en-zh" ]] || [[ "$explainability_model" == "Helsinki-NLP/opus-mt-zh-en" ]]; then echo "-7"; else echo "-100"; fi)
    for lmsd in $(echo "0.7"); do
        lmsd_str=$(echo "$lmsd" | tr '.' '_')
    #for layer1 in $(if [[ "$lm" == ":::" ]]; then seq -1 -1 "$last_layer"; else echo "-15"; fi); do
    for layer1 in $(echo "-15"); do
        layer=($layer1)

        for i in $(seq $nlayerscat); do
            newlayer=$((layer1-i))
            layer+=($newlayer)
        done

        if [[ "$last_layer" -gt "${layer[-1]}" ]]; then
            continue
        fi

        if [[ ! -z "$lm_model_input_str" ]]; then
            mdeberta_pretrained_path_final1=$(echo "$mdeberta_pretrained_path" | sed 's/{template}/'"${t}"'/' | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')
            mdeberta_pretrained_path_log1=$(echo "$mdeberta_pretrained_path_log" | sed 's/{template}/'"${t}"'/' | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')
            best_mdeberta="-1.0"
            mdeberta_pretrained_path_final=""
            for seed2 in $(echo "$seeds"); do
                mdeberta_pretrained_path_final2=$(echo "$mdeberta_pretrained_path_final1" | sed 's/{seed}/'"${seed2}"'/')
                mdeberta_pretrained_path_log2=$(echo "$mdeberta_pretrained_path_log1" | sed 's/{seed}/'"${seed2}"'/')
                mdeberta_metric=$(cat "$mdeberta_pretrained_path_log2" | fgrep -a "Best dev patience metric update" | tail -1 | awk '{print $NF}')

                best_mdeberta2=$(python3 -c 'print('$mdeberta_metric') if '$mdeberta_metric' > '$best_mdeberta' else print('$best_mdeberta')')

                if [[ -z "$best_mdeberta2" ]]; then
                    echo "ERROR: could not obtain LM score"
                    mdeberta_pretrained_path_final=""
                    break
                fi
                if [[ "$best_mdeberta" != "$best_mdeberta2" ]]; then
                    mdeberta_pretrained_path_final="$mdeberta_pretrained_path_final2"
                    best_mdeberta="$best_mdeberta2"
                fi
            done

            if [[ -z "$mdeberta_pretrained_path_final" ]]; then
                echo "ERROR: pretrained path is empty"
                continue
            fi
            if [[ ! -f "$mdeberta_pretrained_path_final" ]]; then
                echo "ERROR: mdeberta file not found: $mdeberta_pretrained_path_final"
                continue
            fi

            lm_model_input="$mdeberta_pretrained_path_final"
        else
            lm_model_input=""
        fi

        data_fn_suffix=".${t}.out.all.shuf"
        error="0"
        pickle_train=()
        pickle_dev=()
        pickle_test=()

        for l in "${layer[@]}"; do
            if [[ "$thelang" == "all" ]] || [[ "$t" == "all_balanced" ]] || [[ "$t" == "all_balanced_no_pickle" ]]; then
                continue # pickle files are not loaded
            fi

            if [[ "$explainability_model" == "facebook/nllb-200-3.3B" ]]; then
                data_fn_pickle_suffix=".${t}.out.all.shuf.pm_${explainability_model_fn}.layer_${l}.direction_${direction}.out.pickle.gz"
            else
                data_fn_pickle_suffix=".${t}.out.all.shuf.pm_${explainability_model_fn}.layer_all.direction_${direction}.out.pickle.layer_${l}.gz"
            fi

            for s in $(echo "train dev test"); do
                data_file="./${f_prefix_cwd}.${s}${data_fn_suffix}"
                pickle_file="./${f_prefix_cwd}.${s}${data_fn_pickle_suffix}"

                if [[ ! -f "$data_file" ]]; then
                    echo "ERROR: file not found: $data_file"
                    error="1"
                fi
                if [[ ! -f "$pickle_file" ]]; then
                    echo "ERROR: pickle file not found: $pickle_file"
                    error="1"
                fi

                if [[ "$s" == "train" ]]; then
                    pickle_train+=("$pickle_file")
                elif [[ "$s" == "dev" ]]; then
                    pickle_dev+=("$pickle_file")
                elif [[ "$s" == "test" ]]; then
                    pickle_test+=("$pickle_file")
                else
                    echo "ERROR: unexpected set: $s"
                    error="1"
                fi
            done
        done
        if [[ "$error" == "1" ]]; then
            continue
        fi
        pickle_train=$(for f in "${pickle_train[@]}"; do echo -n ":$f"; done | sed 's/^://')
        pickle_dev=$(for f in "${pickle_dev[@]}"; do echo -n ":$f"; done | sed 's/^://')
        pickle_test=$(for f in "${pickle_test[@]}"; do echo -n ":$f"; done | sed 's/^://')
        layer_str=$(for l in "${layer[@]}"; do echo -n "_${l}"; done)
        layer_str="layer$layer_str"
        mfn=$(echo "${thelang}-mtpaper_${explainability_model_fn}${data_fn_suffix}.direction_${direction}.${layer_str}.lm_${lm_pretrained_model}.lm_input_${lm_model_input_str}.lm_frozen_params_${lm_frozen_params}" | tr '/' '_')
        patience_str="6"
        for patience in $(echo "$patience_str"); do

        prefix="${output}_${mfn}"
        if [[ "$patience" != "6" ]]; then
            prefix="${prefix}.patience_${patience}"
        fi
    
        prefix=$(echo "$prefix" | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')

        if [[ "$patience_metric" != "acc" ]]; then
            prefix="${prefix}_pm_${patience_metric}"
        fi

        direction_str=$(echo "$direction" | tr '+' ' ')
        extra_args=""

        if [[ "$our_model_input_str" == "1" ]]; then
            m_pretrained_path_final1=$(echo "${prefix}.model_output" | sed 's/lm_frozen_params_'$lm_frozen_params'/lm_frozen_params_1/' | sed -r 's/.seed_[0-9]+\//.seed_{seed}\//')
            m_pretrained_path_log1=$(echo "${prefix}.out" | sed 's/lm_frozen_params_'$lm_frozen_params'/lm_frozen_params_1/' | sed -r 's/.seed_[0-9]+\//.seed_{seed}\//')
            best_m="-1.0"
            m_pretrained_path_final=""
            for seed2 in $(echo "$seeds"); do
                m_pretrained_path_final2=$(echo "$m_pretrained_path_final1" | sed 's/{seed}/'"${seed2}"'/')
                m_pretrained_path_log2=$(echo "$m_pretrained_path_log1" | sed 's/{seed}/'"${seed2}"'/')
                m_metric=$(cat "$m_pretrained_path_log2" | fgrep -a "better dev result" | tail -1 | awk '{print $NF}')

                best_m2=$(python3 -c 'print('$m_metric') if '$m_metric' > '$best_m' else print('$best_m')')

                if [[ -z "$best_m2" ]]; then
                    echo "ERROR: could not obtain classifier score"
                    m_pretrained_path_final=""
                    break
                fi
                if [[ "$best_m" != "$best_m2" ]]; then
                    m_pretrained_path_final="$m_pretrained_path_final2"
                    best_m="$best_m2"
                fi
            done

            if [[ -z "$m_pretrained_path_final" ]]; then
                echo "ERROR: pretrained path is empty"
                continue
            fi

            #extra_args="--model-input $m_pretrained_path_final --stochastic-depth 0.7 --frozen-params $extra_args"
            extra_args="--model-input $m_pretrained_path_final $extra_args"

            #prefix="${prefix}.hs_classifier_input_yes"
            prefix="${prefix}.hs_classifier_input_yes.no_sd_nor_frozen"
        fi

        if [[ ! -z "$lm_pretrained_model" ]] && [[ "$lm_frozen_params" != "1" ]]; then
            extra_args="--lm-model-output ${prefix}.lm.model_output $extra_args"
        fi

        if [[ "$lm_frozen_params" == "1" ]]; then
            extra_args="--lm-frozen-params $extra_args"
        fi
        if [[ ! -z "$lm_model_input" ]]; then
            extra_args="--lm-model-input $lm_model_input $extra_args"
        fi
        if [[ ! -z "$lm_pretrained_model" ]]; then
            extra_args="--lm-pretrained-model $lm_pretrained_model --lm-ensemble-approach token $extra_args"

            if [[ "$our_model_input_str" != "1" ]]; then
                extra_args=" --lm-stochastic-depth $lmsd $extra_args"
            fi
        fi

        if [[ "$thelang" == "all" ]] && [[ "$t" == "all_balanced" ]]; then
            extra_args="--batch-size 2 --gradient-accumulation 16 $extra_args"
        elif [[ "$thelang" == "all" ]] || [[ "$t" == "all_balanced" ]] || [[ "$t" == "all_balanced_no_pickle" ]]; then
            extra_args="--batch-size 4 --gradient-accumulation 8 $extra_args"
        else
            if [[ "$thelang" == "es-de" ]] && [[ "$explainability_model" == "facebook/nllb-200-1.3B" ]]; then
                extra_args="--batch-size 16 --gradient-accumulation 2 $extra_args"
            elif [[ "$explainability_model" == "facebook/nllb-200-3.3B" ]]; then
                extra_args="--batch-size 16 --gradient-accumulation 2 $extra_args"
            else
                extra_args="--batch-size 32 --gradient-accumulation 1 $extra_args"
            fi

            extra_args="--pickle-train-filename $pickle_train --pickle-dev-filename $pickle_dev --pickle-test-filename $pickle_test $extra_args"
        fi

        if [[ "$lmsd" != "0.7" ]]; then
            prefix="${prefix}.sd_${lmsd_str}"
        fi

        ####

        if [[ -f "${prefix}.out" ]]; then
            echo "Already exists: ${prefix}.out"
            continue
        fi

        echo "$(date) check: ${prefix}.out"

        smatd \
            ./${f_prefix_cwd}.{train,dev,test}${data_fn_suffix} \
            $default_args_explainability --dev-patience-metric "$patience_metric" --source-lang "$nllb_l1" --target-lang "$nllb_l2" \
            --direction $direction_str --model-output "${prefix}.model_output" --pretrained-model "$explainability_model" \
            --seed "$seed" --pretrained-model-target-layer "$layer1" --patience "$patience" $extra_args \
        &> "${prefix}.out" &
        njobs=$((njobs+1))
        if [[ "$njobs" -ge "$workers" ]]; then
            echo "Waiting..."
            wait
            njobs="0"
        fi
    done
    done
    done
    done
    done
    done
    wait
done
done
done
done
wait

date
