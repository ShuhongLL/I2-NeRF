#!/bin/bash

SCENE=scene2
EXPERIMENT=UW/"$SCENE"
# DATA_ROOT=/data/umihebi0/users/shuhong/SeathruNeRF_dataset
DATA_ROOT=/data/umihebi0/users/shuhong/okinawa
DATA_DIR="$DATA_ROOT"/"$SCENE"

export CUDA_VISIBLE_DEVICES=4

# accelerate launch train.py \
python train.py \
  --gin_configs=configs/llff_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"