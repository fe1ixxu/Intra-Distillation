#!/bin/bash

lang=de # de->en
pass=3 # Number of time we pass the model
alpha=5 # Alpha value

SAVE_DIR=./models/${lang}-${pass}-${alpha}/

fairseq-train ./data/data-bin-${lang}/ --arch transformer_iwslt_de_en --task translation_intra_distillation \
--alpha ${alpha}  --adaptive-alpha 1 --max-updates-train 50000 --max-update 50000 --num-iter ${pass} \
--temperature-q 10 --temperature-p 5  \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 8000  --dropout 0.3 --attention-dropout 0.1 \
--weight-decay 0.0001 --max-tokens 4096 --update-freq 1 --keep-interval-updates 1 --patience 40 \
--no-epoch-checkpoints --log-format simple --log-interval 100 \
--ddp-backend no_c10d --fp16  --fp16-init-scale 16 --seed 1 \
--save-dir ${SAVE_DIR} --max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/