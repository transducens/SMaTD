#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    # Kill all child processes of the current script
    pkill --signal SIGTERM -P $$ # SIGINT does not work...

    exit -1
}

# Trap SIGINT (Ctrl+C) and SIGTERM signals to call the cleanup function
trap cleanup SIGINT SIGTERM

workers="4"
#workers="8"
njobs="0"

# MT cross eval
input_dir="./results_v6.nllb_hidden_state.max_from_seeds_71213_42_43"

# Baseline

#output="./results_v2.seed_${seed}/reproduction"
output="${input_dir}/cross_eval.v2.baseline.load"

mkdir -p $(dirname "$output")

echo "$(date) current output: $output"

#for l1 in $(echo "de-es es-de de-en zh-en ru-en en-de en-zh en-ru all"); do
#for l2 in $(echo "de-es es-de de-en zh-en ru-en en-de en-zh en-ru fi-en"); do
for l1 in $(echo "de-es es-de de-en ru-en en-de en-ru all"); do
for l2 in $(echo "de-es es-de de-en ru-en en-de en-ru fi-en"); do
for m1 in $(if [[ "$l1" == "all" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP_opus-mt-${l1} all_balanced"; elif [[ "$l1" == "de-en" ]] || [[ "$l1" == "zh-en" ]] || [[ "$l1" == "ru-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 all_balanced"; else echo "Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP_opus-mt-${l1} all_balanced"; fi); do
for m2 in $(if [[ "$l2" == "fi-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP_opus-mt-${l2} facebook/m2m100_1.2B"; elif [[ "$l2" == "de-en" ]] || [[ "$l2" == "zh-en" ]] || [[ "$l2" == "ru-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 facebook/m2m100_1.2B"; else echo "Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP_opus-mt-${l2} facebook/m2m100_1.2B"; fi); do
for s in $(echo "test"); do
    # model: l1 m1
    # eval data: l2 m2

    mfn=$(echo "$m1" | tr '/' '_')
    mfn2=$(echo "$m2" | tr '/' '_')
    log_file="${input_dir}/reproduction_exp2_bilingual_mdeberta-v3-base_${l1}-mtpaper_${mfn}.out"
    log_file=$(echo "$log_file" | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')

    if [[ ! -f "$log_file" ]]; then
        echo "ERROR: log file not found: $log_file"
        continue
    fi

    lm_model_input=$(cat "$log_file" | head -1 | sed -r 's/^.*, model_output=('\''([^'\'']*)'\''|None).*$/\2/' | sed -e 's/microsoft_//g')

    if [[ ! -d "$lm_model_input" ]] && [[ ! -f "$lm_model_input" ]] && [[ "$lm_model_input" =~ results_v2/reproduction ]]; then
        lm_model_input=$(echo "$lm_model_input" | sed 's/results_v2\/reproduction/results_v2.seed_71213\/reproduction/')
    fi

    if [[ -d "$lm_model_input" ]]; then
        lm_model_input="${lm_model_input}/mtd_best_dev.pt"
    fi

    if [[ ! -f "$lm_model_input" ]]; then
        echo "ERROR: LM input not found: $lm_model_input"
        continue
    fi

    inference_file="./wmt_data/${l2}-mtpaper.all.sentences.shuf.${s}.${mfn2}.out.all.shuf"

    if [[ ! -f "$inference_file" ]]; then
        echo "ERROR: not all exist: $inference_file"
        continue
    fi

    extra_args=""

    lm="microsoft/mdeberta-v3-base"
    lmfn=$(echo "${lm}_${l1}-mtpaper" | tr '/' '_')
    prefix="${output}_bilingual_${lmfn}_${mfn}.inference_${l2}_${mfn2}_${s}"
    prefix=$(echo "$prefix" | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')

    if [[ -f "${prefix}.out" ]]; then
        echo "Already exists: ${prefix}.out"
        continue
    fi

    echo "$(date) check baseline: ${prefix}.out"

    cat "$inference_file" \
      | PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python smatd-lm-baseline --inference \
        --pretrained-model "$lm" --model-input "$lm_model_input" \
        --verbose --batch-size 32 \
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

if [[ "$njobs" -ge "$workers" ]]; then
    echo "Waiting..."
    wait
    njobs="0"
fi


# Ours

output="${input_dir}/cross_eval.v2.load"

mkdir -p $(dirname "$output")thelang nllb_l1 nllb_l2 t

echo "$(date) current output: $output"

#for lp in $(echo "all:UNK:UNK de-en:deu_Latn:eng_Latn ru-en:rus_Cyrl:eng_Latn zh-en:zho_Hans:eng_Latn es-de:spa_Latn:deu_Latn en-zh:eng_Latn:zho_Hans en-ru:eng_Latn:rus_Cyrl de-es:deu_Latn:spa_Latn en-de:eng_Latn:deu_Latn"); do
#for lp2 in $(echo "de-en:deu_Latn:eng_Latn ru-en:rus_Cyrl:eng_Latn zh-en:zho_Hans:eng_Latn es-de:spa_Latn:deu_Latn en-zh:eng_Latn:zho_Hans en-ru:eng_Latn:rus_Cyrl de-es:deu_Latn:spa_Latn en-de:eng_Latn:deu_Latn fi-en:fin_Latn:eng_Latn"); do
for lp in $(echo "all:UNK:UNK de-en:deu_Latn:eng_Latn ru-en:rus_Cyrl:eng_Latn es-de:spa_Latn:deu_Latn en-ru:eng_Latn:rus_Cyrl de-es:deu_Latn:spa_Latn en-de:eng_Latn:deu_Latn"); do
for lp2 in $(echo "de-en:deu_Latn:eng_Latn ru-en:rus_Cyrl:eng_Latn es-de:spa_Latn:deu_Latn en-ru:eng_Latn:rus_Cyrl de-es:deu_Latn:spa_Latn en-de:eng_Latn:deu_Latn fi-en:fin_Latn:eng_Latn"); do
thelang=$(echo "$lp" | cut -d: -f1)
nllb_l1=$(echo "$lp" | cut -d: -f2)
nllb_l2=$(echo "$lp" | cut -d: -f3)
thelang2=$(echo "$lp2" | cut -d: -f1)
nllb_l12=$(echo "$lp2" | cut -d: -f2)
nllb_l22=$(echo "$lp2" | cut -d: -f3)
explainability_model="facebook/nllb-200-3.3B"
explainability_model_fn=$(echo "$explainability_model" | tr '/' '_')

for lm in $(echo ":: microsoft/mdeberta-v3-base:1:1"); do

lm_pretrained_model=$(echo "$lm" | cut -d: -f1)
lm_model_input_str=$(echo "$lm" | cut -d: -f2)
lm_frozen_params=$(echo "$lm" | cut -d: -f3)

if [[ ! -z "$lm_model_input_str" ]]; then
    lm_model_input_str="yes"
fi

direction="src2trg"

for t in $(if [[ "$thelang" == "all" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP/opus-mt-${thelang} all_balanced"; elif [[ "$thelang" == "de-en" ]] || [[ "$thelang" == "zh-en" ]] || [[ "$thelang" == "ru-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 all_balanced"; else echo "Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP/opus-mt-${thelang} all_balanced"; fi); do
for t2 in $(if [[ "$thelang2" == "fi-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP/opus-mt-${thelang2} facebook/m2m100_1.2B"; elif [[ "$thelang2" == "de-en" ]] || [[ "$thelang2" == "zh-en" ]] || [[ "$thelang2" == "ru-en" ]]; then echo "google deepl Unbabel/TowerInstruct-7B-v0.2 facebook/m2m100_1.2B"; else echo "Unbabel/TowerInstruct-7B-v0.2 google/madlad400-3b-mt Helsinki-NLP/opus-mt-${thelang2} facebook/m2m100_1.2B"; fi); do

# model: thelang nllb_l1 nllb_l2 t
# eval data: thelang2 nllb_l12 nllb_l22 t2
t=$(echo "$t" | tr '/' '_')
t2=$(echo "$t2" | tr '/' '_')

f_prefix_cwd="./wmt_data/${thelang2}-mtpaper.all.sentences.shuf"
layer_str="layer_-15"
data_fn_pickle_suffix=".${t2}.out.all.shuf.pm_${explainability_model_fn}.${layer_str}.direction_${direction}.out.pickle.gz"
pickle_train="./${f_prefix_cwd}.train${data_fn_pickle_suffix}"
pickle_dev="./${f_prefix_cwd}.dev${data_fn_pickle_suffix}"
pickle_test="./${f_prefix_cwd}.test${data_fn_pickle_suffix}"

mfn=$(echo "${thelang}-mtpaper_${explainability_model_fn}.${t}.out.all.shuf.direction_${direction}.${layer_str}.lm_${lm_pretrained_model}.lm_input_${lm_model_input_str}.lm_frozen_params_${lm_frozen_params}" | tr '/' '_')
log_file="${input_dir}/exp_${mfn}.out"
log_file=$(echo "$log_file" | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')

if [[ ! -f "$log_file" ]]; then
    echo "ERROR: log file not found: $log_file"
    continue
fi

extra_args=""

m_pretrained_path_final=$(cat "$log_file" | head -1 | sed -r 's/^.*, model_output='\''([^'\'']*)'\''.*$/\1/')
lm_model_input=$(cat "$log_file" | head -1 | sed -r 's/^.*, lm_model_input=('\''([^'\'']*)'\''|None).*$/\2/')

if [[ ! -z "$lm_pretrained_model" ]]; then
    if [[ ! -f "$lm_model_input" ]]; then
        echo "ERROR: LM input not found: $lm_model_input"
        continue
    fi
    extra_args="--lm-model-input $lm_model_input $extra_args"
    extra_args="--lm-pretrained-model $lm_pretrained_model --lm-ensemble-approach token $extra_args"
fi

prefix="${output}_${mfn}.inference_${thelang2}_${t2}"
prefix=$(echo "$prefix" | sed -e 's/microsoft_//g' -e 's/facebook_//g' -e 's/Unbabel_//g')

if [[ ! -f "$m_pretrained_path_final" ]]; then
    echo "ERROR: model input not found: $m_pretrained_path_final"
    continue
fi

if [[ -f "${prefix}.out" ]]; then
    echo "Already exists: ${prefix}.out"
    continue
fi

if [[ "$thelang2" == "fi-en" ]]; then
    extra_args="--skip-dev-set-eval $extra_args"
fi

echo "check: ${prefix}.out"

smatd \
    ./${f_prefix_cwd}.{train,dev,test}.${t2}.out.all.shuf \
    --pickle-train-filename "$pickle_train" \
    --pickle-dev-filename "$pickle_dev" \
    --pickle-test-filename "$pickle_test" \
    --verbose --num-layers 3 --num-attention-heads 4 --inference \
    --source-lang "$nllb_l12" --target-lang "$nllb_l22" --skip-train-set-eval \
    --direction $direction --pretrained-model "$explainability_model" \
    --batch-size 16 --model-input $m_pretrained_path_final $extra_args \
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
wait

date
