#!/bin/bash

JOB_ID=psenet_rc185
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir gs://gsoc-tfjs/weights/psenet/custom/$JOB_ID \
    --module-name psenet.train \
    --package-path psenet/ \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --region us-east1 \
    --config config.yaml \
    -- \
    --num-epochs 600 \
    --steps-per-epoch 80 \
    --kernels-num 7 \
    --batch-size 128 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/2019/preprocessed/train \
    --backbone-name mobilenetv2 \
    --learning-rate 0.001 \
    --decay-steps 16000 \
    --decay-rate 0.01 \
    --num-readers 4 \
    --resize-length 320 \
    --num-gpus 8 \
    --prefetch 4 \
    --dataset preprocessed
