#!/bin/bash

JOB_ID=psenet_rc30
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir gs://gsoc-tfjs/weights/psenet/custom/$JOB_ID \
    --module-name psenet.train \
    --package-path psenet/ \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --region us-central1 \
    --scale-tier custom \
    --master-machine-type complex_model_l_gpu \
    -- \
    --train-steps 600 \
    --eval-steps 10 \
    --kernels-num 7 \
    --batch-size 16 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/train \
    --eval-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/eval \
    --backbone-name mobilenetv2 \
    --learning-rate 0.0001 \
    --decay-steps 200 \
    --eval-start-delay-secs 600 \
    --eval-throttle-secs 1800 \
    --save-checkpoints-secs 90 \
    --save-summary-steps 1 \
    --readers-num 32 \
    --gpus-num 8 \
    --cpus-num 32
