#!/bin/bash

SCENE=scene4_dark
EXPERIMENT=LLUW/"$SCENE"
DATA_ROOT=/path/to/okinawa
DATA_DIR="$DATA_ROOT"/"$SCENE"

export CUDA_VISIBLE_DEVICES=0

# accelerate launch train.py \
python train.py \
  --gin_configs=/home/mil/s-liu/media-nerf/configs/Okinawa_lluw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}_1'"