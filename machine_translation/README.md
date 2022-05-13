## Prerequisites
```
conda create -n better-mt python=3.8
conda activate ID
pip install -e ./
pip install sentencepiece
pip install tensorboardX
pip install gdown
```

## Data Download and Preprocessing
We download and preprocess IWSLT'14 dataset for 8 language pairs here (XX->En):
```
bash scripts/preprocess_iwslt14.sh
```
## Train
To train the model, run:
```
bash scripts/train.sh
```
## Evaluate
To evaluate the model on the test set, run:
```
bash scripts/inference.sh ${MODEL_PATH} ${LANGUAGE}
```
where `${MODEL_PATH}` is your model path and `${LANGUAGE}` is the source language. SacreBLEU will be printed out at the end.
