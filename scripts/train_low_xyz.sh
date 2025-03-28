#!/bin/bash

SCENE=bike
EXPERIMENT=LOW/"$SCENE"
DATA_ROOT=/data/umihebi0/users/shuhong/LOM_full
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
# export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch train.py \
  --gin_configs=configs/llff_low.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"