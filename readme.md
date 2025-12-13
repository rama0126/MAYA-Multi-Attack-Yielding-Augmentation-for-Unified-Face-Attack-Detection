# SIREN SHIELD  
Chung-Ang University

## Dataset Directory

    FAS-ICCV
    ├── Data-train
    │ ├── 00000.png
    │ ├── ...
    ├── Data-val
    ├── Data-test
    └── phase1
    ├── Protocol-train.txt
    ├── Protocol-val.txt
    └── Protocol-test.txt


# Getting Started


## Environment
Pyton 3.10.x
NVIDIA GeForce RTX A6000


    pip install -r requirements.txt

# Workflow
## 1. Data preprocessing

    python data_preprocessing.py --base_dir [DATASET_PATH]

DATASET_PATH is dataset directory path, such as "/workspace/datasets/FAS-ICCV"


Preprocessed outputs will be automatically saved to:

    FAS-ICCV/Data-train_preprocessed

    FAS-ICCV/Data-val_preprocessed

    FAS-ICCV/Data-test_preprocessed

## 2. Model Training

    python train.py --train_base_dir [TRAIN_DATASET_PATH]

TRAIN_DATASET_PATH is train dataset directory path, such as "/workspace/datasets/FAS-ICCV/Data-train_preprocessed"

## 3. Predcition

    python predict_logit.py --model_path [MODEL_PATH] --base_dir [DATASET_PATH]

DATASET_PATH is dataset directory path, such as "/workspace/datasets/FAS-ICCV"
MODEL_PATH is trained model path, such as "./OUR_BEST_MODEL.pth"

## Evaluation
Best Model Score: 0.1765962804 (phase 2) (our 5th submission)

Weights File: OUR_BEST_MODEL.pth