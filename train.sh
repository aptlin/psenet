#!/bin/bash

# the checkpoint with mobilenet weights:
#gs://gsoc-tfjs/weights/psenet/warm/segmentation_filters_128

JOB_ID=psenet_rc146
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir gs://gsoc-tfjs/weights/psenet/custom/$JOB_ID \
    --module-name psenet.train \
    --package-path psenet/ \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --region us-east1 \
    --config config.yaml \
    -- \
    --train-steps 36000 \
    --eval-steps 10 \
    --kernels-num 7 \
    --batch-size 8 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/2019/tfrecords/train \
    --eval-data-dir gs://gsoc-tfjs/data/icdar/mlt/2019/tfrecords/eval \
    --warm-ckpt gs://gsoc-tfjs/weights/psenet/custom/psenet_rc145 \
    --backbone-name mobilenetv2 \
    --learning-rate 0.00001 \
    --decay-steps 12000 \
    --decay-rate 0.01 \
    --eval-start-delay-secs 120 \
    --eval-throttle-secs 72000 \
    --save-checkpoints-steps 10 \
    --save-summary-steps 5 \
    --readers-num 4 \
    --resize-length 320 \
    --gpu-per-worker 4 \
    --prefetch 4 \
    --distribution-strategy multi-worker-mirrored \
    --regularization-weight-decay 0.0005 \
    --augment-training-data False
