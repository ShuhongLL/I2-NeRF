#!/bin/bash

SCENE=fern
EXPERIMENT=fern_test/"$SCENE"
DATA_ROOT=/path/to/NeRF_data
DATA_DIR="$DATA_ROOT"/"$SCENE"

export CUDA_VISIBLE_DEVICES=0

# accelerate launch train.py \
python train.py \
  --gin_configs=configs/Fern_test.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"