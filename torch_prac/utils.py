import os
import pandas as pd
from transformers import AutoTokenizer

from torch.optim import Adam, AdamW
from torch_prac.optimizer.optimizer import AdamP
from torch_prac.model.models import (
    ROBERTA_KLUE_Classifier,
    ROBERTA_S_Multi_Classifier,
)

from torch_prac.config import CHPath
from torch_prac.config import parser_args
from torch_prac.logger import get_logger

config = parser_args()
path = CHPath(config)
LOGGER = get_logger()


def get_tokenizer(config):
    if config.model == "klue_roberta":
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    elif config.model == "s_multi_roberta":
        tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")

    else:
        raise NotImplementedError("Tokenizer not avaliable")

    return tokenizer


def get_model(config):
    if config.model == "klue_roberta":
        model = ROBERTA_KLUE_Classifier(config)

    elif config.model == "s_multi_roberta":
        model = ROBERTA_S_Multi_Classifier(config)

    else:
        raise NotImplementedError("Model not available")

    return model


def get_optimizer(model, config):
    if config.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.01)
    elif config.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    elif config.optimizer == "AdamP":
        optimizer = AdamP(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            delta=0.1,
            wd_ratio=0.1,
            nesterov=False,
        )
    else:
        raise NotImplementedError("Optimizer not available")

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer


def save_tokenizer(tokenizer, save_path):
    tokenizer.save_pretrained(save_path)


def load_dataset(config, dataset_file_path):
    if os.path.isfile(path.preprocessed_dataset_file_path):
        LOGGER.info("[+] Preprocessed Dataset already exist")
        dataset = pd.read_feather(path.preprocessed_dataset_file_path)
    else:
        LOGGER.info("[+] Dataset DownLoading")
        dataset = pd.read_csv(dataset_file_path)
        dataset.columns = ["index", "text", "label"]

        LOGGER.info("[+] Dataset Preprocessing")
        dataset = preprocessing_dataset(config, dataset)

    return dataset


def preprocessing_dataset(config, dataset):

    # label 기준 결측치 행 제거, 중복 제거
    dataset = dataset.loc[dataset["label"].isnull() == False, :]
    dataset = dataset.drop_duplicates(["text", "label"])

    ## Label Encoding
    # label_to_num_dict = {
    #     "entailment": 0,
    #     "contradiction": 1,
    #     "neutral": 2,
    # }
    # dataset["labels"] = dataset.label.map(label_to_num_dict)

    # 전처리 데이터셋 저장
    dataset.to_feather(path.preprocessed_dataset_file_path)

    return dataset.reset_index(drop=True)


def get_train_valid_loader(
    config, train_data, valid_data, train_label, valid_label, tokenizer
):
    tokenized_train = tokenized_dataset(config, train_data, tokenizer)
    tokenized_valid = tokenized_dataset(config, valid_data, tokenizer)

    train_dataset = CHDataset(tokenized_train, train_label)
    valid_dataset = CHDataset(tokenized_valid, valid_label)

    trainloader = CHDataloader(config, train_dataset).get_dataloader()
    validloader = CHDataloader(config, valid_dataset).get_dataloader()

    return trainloader, validloader
