## Prerequisites
```
conda create --name xtreme --file conda-env.txt
conda activate xtreme
bash install_tools.sh
```

## Download the Data
There are two ways to download the data. The first way is easier: directly downloading our zipped data from google drive:
```
gdown https://drive.google.com/uc?id=1uA6QHGQ9iBqhYe45A54EPBUwsFTmHxec
unzip download.zip
rm download.zip
```
All data will be located at the folder `download/`

The second is following the command to download the data:
```
bash scripts/download_data.sh
```

## Training
To conduct intra-distillation on various tasks, we run (It may take a while to load data for the first run):
```
bash scripts/train.sh [MODEL] [TASK] [SEED] [ALPHA] [GPU]
```
The model and results will be stored at `outputs/seed-${SEED}/"}`

For example, we fine-tune xlm-roberta-large on the NER task with random seed 1, alpha 1, and training on the GPU 0:
```
bash scripts/train.sh xlm-roberta-large panx 1 1 0
```

Note that our codebase only support intra-distillation for 2 tasks presented in the paper, i.e. `panx` and `tydiqa`. The 5 seeds used in the paper are 1,2,3,4,5. 

## Results
You can find your results on the test sets for all target languages in the `test_results.txt` file. The reader can find them under `outputs` folder. For example, they are located at:
```
for panx: outputs/seed-1/panx/xlm-roberta-large-LR2e-5-epoch10-MaxLen128/test_results.txt
for tydiqa: outputs/seed-1/xlm-roberta-base_LR3e-5_EPOCH30_maxlen384_batchsize4_gradacc8/predictions/test_results.txt
```



