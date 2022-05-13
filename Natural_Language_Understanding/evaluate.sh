#!/bin/bash
task=qqp # task name (cola sst2 mrpc stsb qqp mnli qnli rte)
name=QQP # output file name (CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE)

OUT=./prediction/

CUDA_VISIBLE_DEVICES=0 python ./run_glue.py \
--model_name_or_path ./models/${task}/ \
--task_name ${task} \
--do_predict \
--max_seq_length 128 \
--output_dir ${OUT} \
--per_device_eval_batch_size 32

if [ $task == "mnli" ]; then
mv ${OUT}/predict_results_mnli-mm.txt ${OUT}/${name}-mm.tsv
mv ${OUT}/predict_results_mnli.txt ${OUT}/${name}-m.tsv
else
mv ${OUT}/*.txt ${OUT}/${name}.tsv
fi
