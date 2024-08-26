# LabelPrompt

code for LabelPrompt: Effective Prompt-based Learning for Relation Classification


## Setup

To set up the environment, follow these steps:

```sh
conda create -n labelprompt python=3.8
conda activate labelprompt

pip install -r requirements.txt
```

## Dataset

The dataset can be found at [KnowPrompt](https://github.com/zjunlp/KnowPrompt).

### for few-shot

To generate few-shot datasets, run the following command:

```sh
python genetate_k_shot.py --dataset retacred --data_file train.txt
```

## Run

To run the model, use the following command:

```sh
python main.py --gpu 0 --dataset retacred --data_path ./dataset/retacred/k-shot/8-1
```