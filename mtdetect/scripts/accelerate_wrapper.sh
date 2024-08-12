#!/bin/bash

GPUS="$1"
ARGS="${@:2}"

if [[ -z "$GPUS" ]]; then
  >&2 echo "You did not specify the number of GPUs to use"

  exit 1
fi

if [[ -z "$MTD_ACCELERATE_PORT" ]]; then
  MTD_ACCELERATE_PORT="29500"
fi

echo "MTD_ACCELERATE_PORT: $MTD_ACCELERATE_PORT; GPUs: $GPUS; Args: $ARGS"

srun --gres=gpu:$GPUS \
accelerate launch --main_process_port "$MTD_ACCELERATE_PORT" --no_python --num_processes="$GPUS" mtdetect $ARGS
