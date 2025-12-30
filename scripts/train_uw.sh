#!/bin/bash

SCENE=Curasao
EXPERIMENT=SeaThru/"$SCENE"
DATA_ROOT=/path/to/SeathruNeRF_dataset
DATA_DIR="$DATA_ROOT"/"$SCENE"

# accelerate launch train.py \
python train.py \
  --gin_configs=configs/SeaThru/llff_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
