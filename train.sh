#!/bin/bash

JOB_ID=psenet_rc169
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir gs://gsoc-tfjs/weights/psenet/custom/$JOB_ID \
    --module-name psenet.train \
    --package-path psenet/ \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --region us-central1 \
    --config config.yaml \
    -- \
    --train-steps 10000 \
    --eval-steps 10 \
    --kernels-num 7 \
    --batch-size 16 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/train \
    --eval-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/eval \
    --backbone-name mobilenetv2 \
    --learning-rate 0.00001 \
    --decay-steps 10000 \
    --decay-rate 0.94 \
    --eval-start-delay-secs 120 \
    --eval-throttle-secs 300 \
    --save-checkpoints-secs 120 \
    --save-summary-steps 100 \
    --readers-num 16 \
    --resize-length 640 \
    --gpus-num 2
