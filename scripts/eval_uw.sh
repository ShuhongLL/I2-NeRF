#!/bin/bash

SCENE=IUI3-RedSea
EXPERIMENT=SeaThru/"$SCENE"
DATA_ROOT=/data/umihebi0/users/shuhong/SeathruNeRF_dataset
DATA_DIR="$DATA_ROOT"/"$SCENE"

python eval.py \
  --gin_configs=configs/SeaThru/llff_uw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
