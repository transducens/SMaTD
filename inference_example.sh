#!/bin/bash

pickle_train="de-es-mtpaper.all.sentences.shuf.train.Helsinki-NLP_opus-mt-de-es.out.all.shuf.pm_facebook_nllb-200-3.3B.layer_-15.direction_src2trg.out.pickle.gz"
pickle_dev="de-es-mtpaper.all.sentences.shuf.dev.Helsinki-NLP_opus-mt-de-es.out.all.shuf.pm_facebook_nllb-200-3.3B.layer_-15.direction_src2trg.out.pickle.gz"
pickle_test="de-es-mtpaper.all.sentences.shuf.test.Helsinki-NLP_opus-mt-de-es.out.all.shuf.pm_facebook_nllb-200-3.3B.layer_-15.direction_src2trg.out.pickle.gz"
model_input="exp_de-es-mtpaper_nllb-200-3.3B.Helsinki-NLP_opus-mt-de-es.out.all.shuf.direction_src2trg.layer_-15.lm_.lm_input_.lm_frozen_params_.model_output"

smatd \
  de-es-mtpaper.all.sentences.shuf.{train,dev,test}.Helsinki-NLP_opus-mt-de-es.out.all.shuf \
  --verbose --num-layers 3 --num-attention-heads 4 --source-lang deu_Latn --target-lang eng_Latn \
  --direction src2trg --pretrained-model "facebook/nllb-200-3.3B" --batch-size 8 \
  --pickle-train-filename "$pickle_train" --pickle-dev-filename "$pickle_dev" --pickle-test-filename "$pickle_test" \
  --model-input "$model_input" --inference --skip-train-set-eval
