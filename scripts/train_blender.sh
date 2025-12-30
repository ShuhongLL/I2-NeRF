#!/bin/bash

SCENE=bike
EXPERIMENT=blender/"$SCENE"
DATA_ROOT=/path/to/LOM
DATA_DIR="$DATA_ROOT"/"$SCENE"

export CUDA_VISIBLE_DEVICES=0

# accelerate launch train.py \
python train.py \
  --gin_configs="configs/LOM/'${SCENE}'.gin" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
