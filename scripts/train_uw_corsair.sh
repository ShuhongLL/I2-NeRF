#!/bin/bash

SCENE=Corsair_3
EXPERIMENT=UW/"$SCENE"
DATA_ROOT=/data/funa0/shuhong/NationalPark_dataset
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
export CUDA_VISIBLE_DEVICES=0

# accelerate launch train.py \
python train.py \
  --gin_configs=configs/Corsair_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"