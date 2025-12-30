#!/bin/bash

SCENE=bike
EXPERIMENT=LOM/"$SCENE"
DATA_ROOT=/path/to/SeathruNeRF_dataset
DATA_DIR="$DATA_ROOT"/"$SCENE"

python eval_lom.py \
  --gin_configs="configs/LOM/${SCENE}.gin" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
