# Mixed-Gradient-Few-Shot
This is the repository of the Findings of NAACL 2022 paper "[Por Qué Não Utiliser Alla Språk? Mixed Training with Gradient Optimization in Few-Shot Cross-Lingual Transfer](https://arxiv.org/pdf/2204.13869.pdf)".

We modified the code based on the [Xtreme benchmark](https://github.com/google-research/xtreme).

## Model Card
We show an example of how we code the stochastic gradient surgery function at `sgs_card.py` for easier locating the implementation of the method.

## Prerequisites
The first step is to build the virtual environment.
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
To conduct few-shot learning on various tasks, we run:
```
bash scripts/train.sh [MODEL] [TASK] [SEED] [FEW_SHOT] [GPU]
```
The model and results will be stored at `outputs/seed-${SEED}/"}`

For example, we run a 5-shot learning by fine-tuning xlm-roberta-large on the NER task with random seed 1, which is trained on the GPU 0:
```
bash scripts/train.sh xlm-roberta-large panx 1 5 0
```

Note that our codebase only support stochastic gradient surgery for 4 tasks presented in the paper, i.e. `panx`, `udpos`, `xnli`, `tydiqa`. The 5 seeds used in the paper are 1,2,3,4,5. 

## Results
You can find your results on the test sets for all target languages in the `test_results.txt` file. The reader can find them under `outputs` folder. For example, they are located at:
```
for panx: outputs/seed-1/panx/xlm-roberta-large-LR2e-5-epoch10-MaxLen128/test_results.txt
for udpos: outputs/seed-1/udpos/xlm-roberta-large-LR2e-5-epoch10-MaxLen128/test_results.txt
for tydiqa: outputs/seed-1/xlm-roberta-base_LR3e-5_EPOCH30_maxlen384_batchsize4_gradacc8/predictions/test_results.txt
for xnli: outputs/seed-1/xlm-roberta-base-LR2e-5-epoch5-MaxLen128/test_results.txt
```



