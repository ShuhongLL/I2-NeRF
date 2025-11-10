#!/bin/bash

SCENE=scene4_dark
EXPERIMENT=LLUW/"$SCENE"
DATA_ROOT=/data/umihebi0/users/shuhong/okinawa
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
# export CUDA_VISIBLE_DEVICES=6

# accelerate launch train.py \
python train.py \
  --gin_configs=/home/mil/s-liu/media-nerf/configs/Okinawa_lluw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}_1'"