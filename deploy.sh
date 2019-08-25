#!/bin/bash

set -e

GCP_BASE_PATH="gs://gsoc-tfjs/weights/psenet/custom"
GCP_SOURCE_PATH="$GCP_BASE_PATH/psenet_rc185"
MODEL_VERSION="psenet_48"
EXPORTED_VERSION="psenet-rc185-v1"
ASSETS_DIR="./scratchpad"
mkdir -p $ASSETS_DIR
cd $ASSETS_DIR
mkdir $MODEL_VERSION
gsutil cp -r "$GCP_SOURCE_PATH/$MODEL_VERSION*" $MODEL_VERSION
gsutil cp -r "$GCP_SOURCE_PATH/checkpoint" $MODEL_VERSION
PYTHONPATH=.. python ../psenet/serve.py \
    --source-dir $MODEL_VERSION \
    --target-dir $EXPORTED_VERSION
zip -r $EXPORTED_VERSION.zip $EXPORTED_VERSION
gsutil cp $EXPORTED_VERSION.zip $GCP_BASE_PATH
