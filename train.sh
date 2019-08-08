JOB_ID=psenet_rc7
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir gs://gsoc-tfjs/weights/psenet/custom \
    --module-name psenet.train \
    --package-path psenet/ \
    --packages segmentation_models/dist/segmentation_models-1.0.0b1.tar.gz \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --region us-central1 \
    --scale-tier STANDARD_1 \
    -- \
    --train-steps 600 \
    --eval-steps 100 \
    --kernels-num 7 \
    --batch-size 32 \
    --training-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/train \
    --eval-data-dir gs://gsoc-tfjs/data/icdar/mlt/tfrecords/eval \
    --backbone-name mobilenetv2 \
    --learning-rate 0.0001 \
    --decay-steps 200 \
    --eval-start-delay-secs 120 \
    --eval-throttle-secs 1800 \
    --save_checkpoints_secs 600 \
    --save_summary_steps 1
