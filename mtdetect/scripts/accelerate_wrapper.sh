#!/bin/bash

GPUS="$1"
ARGS="${@:2}"

if [[ -z "$GPUS" ]]; then
  >&2 echo "You did not specify the number of GPUs to use"

  exit 1
fi

echo "GPUs: $GPUS; Args: $ARGS"

srun --gres=gpu:$GPUS \
accelerate launch --no_python --num_processes="$GPUS" mtdetect $ARGS
