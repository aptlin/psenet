#!/bin/bash

# the checkpoint with mobilenet weights:
#gs://gsoc-tfjs/weights/psenet/warm/segmentation_filters_128

JOB_ID=psenet_rc168
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
    --kernels-num 7 \
    --batch-size 16 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/2019/preprocessed/train \
    --backbone-name mobilenetv2 \
    --learning-rate 0.0001 \
    --decay-steps 2500 \
    --decay-rate 0.01 \
    --readers-num 4 \
    --resize-length 320 \
    --gpu-per-worker 8 \
    --prefetch 4 \
    --distribution-strategy multi-worker-mirrored \
    --regularization-weight-decay 0.0005 \
    --augment-training-data False \
    --dataset preprocessed
