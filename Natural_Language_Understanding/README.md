## Prerequisites
```
conda create -n nlu python=3.8
conda activate nlu
pip install -r requirements.txt
pip install transformers
```

## Train the Model
Taking QQP task as an example (Note that the dataset will be automatically loaded):
```
bash run.sh
```

## To Generate Test Results
```
bash evaluate.sh
```