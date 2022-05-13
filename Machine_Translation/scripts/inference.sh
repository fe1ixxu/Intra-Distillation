#!/bin/bash
MODEL_PATH=${1}
lang=${2}


DATA_DIR=./data/data-bin-${lang}
FSRC=./data/tok-${lang}/test.${lang}
FTGT=./data/raw-${lang}/test.en
FOUT=${MODEL_PATH}/results/test.en
mkdir -p ${MODEL_PATH}/results

cat $FSRC | \
fairseq-interactive ${DATA_DIR} \
    --path $MODEL_PATH/checkpoint_best.pt \
    --buffer-size 1024 --batch-size 100 \
    --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece | \
grep -P "^H" | cut -f 3- > $FOUT

cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu
head ${FOUT}.bleu