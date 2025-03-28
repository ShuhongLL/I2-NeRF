#!/bin/bash

SCENE=Curasao
EXPERIMENT=UW_xyz/"$SCENE"
DATA_ROOT=/data/umihebi0/users/shuhong/SeathruNeRF_dataset
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
# export CUDA_VISIBLE_DEVICES=3,4,5,6

accelerate launch train.py \
  --gin_configs=configs/llff_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"