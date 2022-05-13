#!/bin/bash

TASK_NAME=qqp

pass=3
alpha=0.5
MODEL=bert-base-uncased

echo TASK_NAME ${TASK_NAME} num_iter ${num_iter} alpha ${alpha}

CUDA_VISIBLE_DEVICES=0 python ./run_glue_no_trainer.py \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --alpha ${alpha} \
  --temperature_p 6.67 \
  --temperature_q 10 \
  --num_train_epochs 20 \
  --intra_distillation \
  --num_iter ${pass} \
  --output_dir ./models/$TASK_NAME/

