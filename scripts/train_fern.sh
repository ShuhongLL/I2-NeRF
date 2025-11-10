#!/bin/bash

SCENE=fern
EXPERIMENT=fern_test/"$SCENE"
DATA_ROOT=/data/umihebi0/users/shuhong/NeRF_fern
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
export CUDA_VISIBLE_DEVICES=1

# accelerate launch train.py \
python train.py \
  --gin_configs=configs/Fern_test.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"