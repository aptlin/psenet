#!/bin/bash

JOB_ID=psenet_rc75
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir gs://gsoc-tfjs/weights/psenet/custom/$JOB_ID \
    --module-name psenet.train \
    --package-path psenet/ \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --region us-central1 \
    --config config.yaml \
    -- \
    --train-steps 36000 \
    --eval-steps 10 \
    --kernels-num 7 \
    --batch-size 4 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/train \
    --eval-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/eval \
    --warm-ckpt gs://weights/psenet/warm \
    --backbone-name mobilenetv2 \
    --learning-rate 0.001 \
    --decay-steps 12000 \
    --decay-rate 0.01 \
    --eval-start-delay-secs 600 \
    --eval-throttle-secs 1800 \
    --save-checkpoints-secs 120 \
    --save-summary-steps 4 \
    --readers-num 4 \
    --resize-length 640 \
    --crop-size 320 \
    --gpus-num 4 \
    --regularization-weight-decay 0.0005
