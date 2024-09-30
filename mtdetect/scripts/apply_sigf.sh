#!/bin/bash

workers="1"
njobs="0"
focus="dev" # or test
p_value_statistically_significant="0.05"

add_mono_and_bilingual="$1"
nllb="$2" # e.g., distilled-600M 1.3B 3.3B

if [[ "$focus" != "dev" ]] && [[ "$focus" != "test" ]]; then
    echo "ERROR: focus must be either dev or test"
    exit 1
fi
if [[ -z "$nllb" ]]; then
    echo "ERROR: nllb was not provided"
    exit 1
fi

if [[ "$add_mono_and_bilingual" != "0" ]] && [[ "$add_mono_and_bilingual" != "1" ]]; then
    echo "ERROR: first argument must be either 0 or 1"
    exit 1
fi

mkdir -p "./inference.nllb-200-${nllb}"

# inference lm

for l in $(echo "de-en-mtpaper zh-en-mtpaper ru-en-mtpaper"); do
for m in $(echo "deepl google Unbabel_TowerInstruct-7B-v0.2"); do
for mono_or_bilingual in $(echo "monolingual bilingual"); do
input_model="./results/reproduction_exp2_${mono_or_bilingual}_microsoft_mdeberta-v3-base_${l}_${m}.model/mtd_best_dev.pt"

if [[ ! -f "$input_model" ]]; then
    >&2 echo "ERROR: LM model not found: $input_model"
    continue
fi

extra_args=""
cut_fields="-f1,2,3"

if [[ "$mono_or_bilingual" == "monolingual" ]]; then
    extra_args="--monolingual $extra_args"
    cut_fields="-f2,3"
fi

lm="microsoft/mdeberta-v3-base"
lm_str=$(echo "$lm" | tr '/' '_')
lm_input="yes"
lm_frozen="0"
inference_output="./inference.nllb-200-${nllb}/inference.${l}.${mono_or_bilingual}.${m}.lm_${lm_str}.lm_input_${lm_input}.lm_frozen_${lm_frozen}"
input_file="./wmt_data/${l}.all.sentences.shuf.${focus}.${m}.out.all.shuf"
echo "Check (LM): $inference_output"
if [[ ! -f "$input_file" ]]; then
    >&2 echo "ERROR: input file not found: $input_file"
    continue
fi
if [[ -f "$inference_output" ]]; then
    echo "Already exists: $inference_output"
    r=$(cat "$inference_output" | fgrep -a "Inference metrics:" | sed 's/^.*Inference metrics:/Inference metrics:/')
    if [[ "$r" != "" ]]; then
        echo "Eval: $inference_output: $r"
        continue
    else
        echo "WARNING: already exists, but could not find evaluation metric: repeating: $inference_output"
    fi
fi

(cat "$input_file" | cut $cut_fields | mtdetect --inference --inference-print-results --pretrained-model "$lm" --model-input "$input_model" $extra_args &> "$inference_output"; \
exit_status="$?"; \
r=$(cat "$inference_output" | fgrep -a "Inference metrics:" | sed 's/^.*Inference metrics:/Inference metrics:/'); \
echo "Eval (status: $exit_status): $inference_output: $r") &

njobs=$((njobs+1))
if [[ "$njobs" -ge "$workers" ]]; then
    echo "Waiting..."
    wait
    njobs="0"
fi

done
done
done

if [[ "$njobs" -gt "0" ]]; then
    echo "Waiting..."
    wait
    njobs="0"
fi

# inference cnn and cnn+lm

for all_l in $(echo "de-en-mtpaper:deu_Latn:eng_Latn zh-en-mtpaper:zho_Hans:eng_Latn ru-en-mtpaper:rus_Cyrl:eng_Latn"); do
l=$(echo "$all_l" | cut -d: -f1)
nllb_l1=$(echo "$all_l" | cut -d: -f2)
nllb_l2=$(echo "$all_l" | cut -d: -f3)
for all_d in $(echo "src2trg:yes:no trg2src:yes:no src2trg+trg2src:yes+yes:no+no"); do
d=$(echo "$all_d" | cut -d: -f1)
tf=$(echo "$all_d" | cut -d: -f2)
igat=$(echo "$all_d" | cut -d: -f3)
for m in $(echo "deepl google Unbabel_TowerInstruct-7B-v0.2"); do
for lmdata in $(echo ":: microsoft/mdeberta-v3-base::0 microsoft/mdeberta-v3-base:yes:0 microsoft/mdeberta-v3-base:yes:1"); do
lm=$(echo "$lmdata" | cut -d: -f1)
lm_str=$(echo "$lm" | tr '/' '_')
lm_input=$(echo "$lmdata" | cut -d: -f2)
lm_frozen=$(echo "$lmdata" | cut -d: -f3)
input_model="./results/cnn_explainability_${l}_facebook_nllb-200-${nllb}.${m}.out.all.shuf.direction_${d}.tf_${tf}.igat_${igat}.lm_${lm_str}.lm_input_${lm_input}.lm_frozen_params_${lm_frozen}.model_output"
if [[ ! -f "$input_model" ]]; then
    >&2 echo "ERROR: model not found: $input_model"
    continue
fi
inference_output="./inference.nllb-200-${nllb}/inference.${l}.${d}.${m}.lm_${lm_str}.lm_input_${lm_input}.lm_frozen_${lm_frozen}"
echo "Check: $inference_output"
if [[ -f "$inference_output" ]]; then
    echo "Already exists: $inference_output"
    r=$(cat "$inference_output" | egrep -a "^Final $focus eval: ")
    if [[ "$r" != "" ]]; then
        echo "Eval: $inference_output: $r"
        continue
    else
        echo "WARNING: already exists, but could not find evaluation metric: repeating: $inference_output"
    fi
fi
(
#CUDA_VISIBLE_DEVICES="" \
MTDETECT_MODEL_INFERENCE_SKIP_TRAIN="1" \
MTDETECT_PICKLE_FN="./wmt_data/${l}.all.sentences.shuf.{template}.${m}.out.all.shuf.pickle_file.facebook_nllb-200-${nllb}.{direction}.${nllb_l1}.${nllb_l2}.teacher_forcing_{teacher_forcing}.ignore_attention_{ignore_attention}.pickle" \
python3 ./mtdetect/transformer_mm_explainability/mtdetect_cnn_using_example_translation_nllb.py \
  ./wmt_data/${l}.all.sentences.shuf.{train,dev,test}.${m}.out.all.shuf \
  16 64 64 "$nllb_l1" "$nllb_l2" ${d} encoder+decoder+cross none 1 avg+max+avg '' 5e-3 1 '' yes no "$lm" '' "$lm_frozen" 1e-5 \
  $input_model '1' 2 \
  &> "$inference_output"; \
exit_status="$?"; \
r=$(cat "$inference_output" | egrep -a "^Final $focus eval: "); \
echo "Eval (status: $exit_status): $inference_output: $r") &
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

if [[ "$njobs" -gt "0" ]]; then
    echo "Waiting..."
    wait
    njobs="0"
fi

# prepare data for sigf

for l in $(echo "de-en-mtpaper zh-en-mtpaper ru-en-mtpaper"); do
tmp_aggregated_results="./inference.nllb-200-${nllb}/eval_${focus}.nllb-200-${nllb}.${l}.dict_file_acc.all.includes_mono_and_bilingual_${add_mono_and_bilingual}"
rm -f "$tmp_aggregated_results"
touch "$tmp_aggregated_results"
echo "Check aggregated results for $l: $tmp_aggregated_results"
for d in $(echo "src2trg trg2src src2trg+trg2src monolingual bilingual"); do
if [[ "$add_mono_and_bilingual" == "0" ]]; then
    if [[ "$d" == "monolingual" ]] || [[ "$d" == "bilingual" ]]; then
        continue
    fi
fi
for lmdata in $(echo ":: microsoft/mdeberta-v3-base::0 microsoft/mdeberta-v3-base:yes:0 microsoft/mdeberta-v3-base:yes:1"); do
if [[ "$d" == "monolingual" ]] || [[ "$d" == "bilingual" ]]; then
    if [[ "$lmdata" != "microsoft/mdeberta-v3-base:yes:0" ]]; then
        continue # this is the only supported combination
    fi
fi
lm=$(echo "$lmdata" | cut -d: -f1)
lm_str=$(echo "$lm" | tr '/' '_')
lm_input=$(echo "$lmdata" | cut -d: -f2)
lm_frozen=$(echo "$lmdata" | cut -d: -f3)
all_ok="1"
for m in $(echo "deepl google Unbabel_TowerInstruct-7B-v0.2"); do
inference_output="./inference.nllb-200-${nllb}/inference.${l}.${d}.${m}.lm_${lm_str}.lm_input_${lm_input}.lm_frozen_${lm_frozen}"

if [[ ! -f "$inference_output" ]]; then
    echo "ERROR: does not exist: $inference_output"
    all_ok="0"
    continue
fi
done
if [[ "$all_ok" == "0" ]]; then
    echo "ERROR: some file does not exist, so can't prepare files for $tmp_aggregated_results"
    continue
fi

inference_output="./inference.nllb-200-${nllb}/inference.${l}.${d}.{deepl,google,Unbabel_TowerInstruct-7B-v0.2}.lm_${lm_str}.lm_input_${lm_input}.lm_frozen_${lm_frozen}"

if [[ "$d" == "monolingual" ]] || [[ "$d" == "bilingual" ]]; then
sanity_check=$(eval cat "$inference_output" | egrep -a "Inference metrics: " | wc -l)
else
sanity_check=$(eval cat "$inference_output" | egrep -a "^Final $focus eval: " | wc -l)
fi

if [[ "$sanity_check" != "3" ]]; then
    echo "ERROR: sanity check not passed: $inference_output"
    continue
fi

output="./inference.nllb-200-${nllb}/eval_${focus}.tp_and_tn_are_1s_otherwise_0.${l}.${d}.all_mt.lm_${lm_str}.lm_input_${lm_input}.lm_frozen_${lm_frozen}"

if [[ "$d" == "monolingual" ]] || [[ "$d" == "bilingual" ]]; then
eval cat "$inference_output" | fgrep -a "inference: stdin" | sed 's/^.*inference: stdin/inference: stdin/' | cut -f5 | sed -e 's/tp/1/' -e 's/tn/1/' -e 's/fp/0/' -e 's/fn/0/' \
  &> "${output}"
else
eval cat "$inference_output" | fgrep -a "inference: $focus" | cut -f5 | sed -e 's/tp/1/' -e 's/tn/1/' -e 's/fp/0/' -e 's/fn/0/' \
  &> "${output}"
fi

acc_num=$(cat "$output" | python3 -c 'import sys; print(sum([int(l.strip()) for l in sys.stdin]))')
acc_den=$(cat "$output" | wc -l)

if [[ "$acc_den" == "0" ]]; then
    echo "ERROR: why denominator is 0? $output"
    continue
fi

acc=$(python3 -c "print($acc_num / $acc_den)")

echo "$acc $output" >> "$tmp_aggregated_results"

done
done
cat "$tmp_aggregated_results" | sort -k1,1gr > "${tmp_aggregated_results}.sort"

# sigf test
sigf_output="${tmp_aggregated_results}.sort.sigf"

echo "Check SIGF test results: ${sigf_output}.{out,log}"

rm -f "${sigf_output}.out"
touch "${sigf_output}.out"
rm -f "${sigf_output}.log"
touch "${sigf_output}.log"

n=$(cat "${tmp_aggregated_results}.sort" | wc -l)
i="1"
j=$((i+1))
group="1"
f1=$(cat "${tmp_aggregated_results}.sort" | head -n "$i" | tail -1 | awk '{print $2}')
s1=$(cat "${tmp_aggregated_results}.sort" | head -n "$i" | tail -1 | awk '{print $1}')

echo "$group $s1 $f1" >> "${sigf_output}.out"

while [[ "$i" -lt "$n" ]] && [[ "$j" -le "$n" ]]; do
    f1=$(cat "${tmp_aggregated_results}.sort" | head -n "$i" | tail -1 | awk '{print $2}')
    f2=$(cat "${tmp_aggregated_results}.sort" | head -n "$j" | tail -1 | awk '{print $2}')
    s1=$(cat "${tmp_aggregated_results}.sort" | head -n "$i" | tail -1 | awk '{print $1}')
    s2=$(cat "${tmp_aggregated_results}.sort" | head -n "$j" | tail -1 | awk '{print $1}')

    echo "sigf: $f1 $f2" >> "${sigf_output}.log"

    p=$(python3 ./mtdetect/scripts/sigf.py --score accuracy_mean -n 100000 "$f1" "$f2" 2>> "${sigf_output}.log" | awk '{print $3}')
    statistically_significant=$(python3 -c "print(1 if $p <= $p_value_statistically_significant else 0)")

    echo "p-value: $p (statistically significant: $statistically_significant)" >> "${sigf_output}.log"

    if [[ "$statistically_significant" == "1" ]]; then
        i=$((j+0))
        j=$((i+1))
        group=$((group+1))
        echo "$group $s2 $f2" >> "${sigf_output}.out"
    else
        echo "$group $s2 $f2" >> "${sigf_output}.out"
        j=$((j+1))
    fi
done
done
