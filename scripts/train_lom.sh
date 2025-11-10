#!/bin/bash

SCENE=bike
EXPERIMENT=blender/"$SCENE"
DATA_ROOT=/data/umihebi0/users/shuhong/LOM_full
DATA_DIR="$DATA_ROOT"/"$SCENE"

# rm exp/"$EXPERIMENT"/*
python train.py --gin_configs=configs/LOM/bike.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
