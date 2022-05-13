
WORKSPACE=workspace
mkdir ${WORKSPACE}
cd ${WORKSPACE}
git clone https://github.com/moses-smt/mosesdecoder.git
cd ..

SCRIPTS=${WORKSPACE}/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

prep=data
tmp=$prep/tmp
mkdir -p $prep $tmp

## Download data from https://drive.google.com/uc?id=1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed
gdown 1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed
tar zxvf 2014-01.tgz

for src in ar de es fa he it nl pl; do
    tgt=en
    pair=${src}-${tgt}
    lprep=${prep}/$src/
    orig=2014-01/texts/$src/en/

    mkdir -p $lprep
    echo "data location: $orig"

    tar zxvf $orig/${pair}.tgz -C $orig/

    # preprocess train data
    orig=$orig/$pair/
    for l in $src $tgt; do
        f=${orig}/train.tags.$pair.$l
        tok=${tmp}/train.tags.$pair.tok.$l

        cat $f | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' | \
        perl $TOKENIZER -threads 8 -l $l > $tok
        echo ""
    done

    perl $CLEAN -ratio 1.5 $tmp/train.tags.$pair.tok $src $tgt $tmp/train.tags.$pair.clean 1 175

    for l in $src $tgt; do
        perl $LC < $tmp/train.tags.$pair.clean.$l > $tmp/train.tags.$pair.lc.$l
        cat $tmp/train.tags.$pair.lc.${l} | perl $DETOKENIZER -threads 8 -l $l > $tmp/train.tags.$pair.$l
        
    done

    echo "Preprocess valid/test data"
    # preprocess valid/test data:
    for l in $src $tgt; do
        for o in `ls $orig/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$tmp/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
        perl $LC > $f
        echo ""
        done
    done

    # Creating train, valid, test data...
    echo "Creating train, valid, test data..."
    for l in $src $tgt; do
        if [ ${l} == 'pt-br' ]
        then
            awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$pair.$l > $lprep/valid.pt
            awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$pair.$l > $lprep/train.pt
            cat $tmp/IWSLT14.T*.${l} > $lprep/test.pt
        else
            awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$pair.$l > $lprep/valid.$l
            awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$pair.$l > $lprep/train.$l
        cat $tmp/IWSLT14.TED.dev2010.$src-$tgt.$l \
            $tmp/IWSLT14.TEDX.dev2012.$src-$tgt.$l \
            $tmp/IWSLT14.TED.tst2010.$src-$tgt.$l \
            $tmp/IWSLT14.TED.tst2011.$src-$tgt.$l \
            $tmp/IWSLT14.TED.tst2012.$src-$tgt.$l \
            > $lprep/test.$l
        fi  
    done

    if [ ${src} == 'pt-br' ]
    then
        mv ${lprep} ${lprep}/../pt
    fi
    echo "Finished!"

    rm $tmp/*
done

rm -rf $tmp
rm -rf ${WORKSPACE}
rm -rf 2014-01*


for lg in ar de es fa he it nl pl; do
    mv ${prep}/${lg} ${prep}/raw-${lg}
    mkdir -p ${prep}/data-bin-${lg}
    mkdir -p ${prep}/tok-${lg}

    python ./scripts/spm_train.py \
    --input=$(echo $(ls ${prep}/raw-${lg}/*) | sed 's/ /,/g') \
    --model_prefix=${prep}/data-bin-${lg}/spm_12k --vocab_size=12000 --character_coverage=0.99999995 \
    --input_sentence_size=1000000

    cut -f 1 ${prep}/data-bin-${lg}/spm_12k.vocab | tail -n +4 | sed "s/$/ 100/g" > ${prep}/data-bin-${lg}/dict.txt

    for mode in train valid test; do
        for l in $lg en; do
            python ./scripts/spm_encode.py \
            --model ${prep}/data-bin-${lg}/spm_12k.model \
            --input ${prep}/raw-${lg}/${mode}.${l} \
            --outputs ${prep}/tok-${lg}/${mode}.${l}
        done
    done

    fairseq-preprocess --task "translation" --source-lang $lg --target-lang en \
    --trainpref ${prep}/tok-${lg}/train --validpref ${prep}/tok-${lg}/valid --testpref ${prep}/tok-${lg}/test \
    --destdir ${prep}/data-bin-${lg}/ --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
    --srcdict ${prep}/data-bin-${lg}/dict.txt --tgtdict ${prep}/data-bin-${lg}/dict.txt
done