from transformers import AutoTokenizer

from torch.optim import Adam, AdamW
from torch_prac.optimizer.optimizer import AdamP
from torch_prac.model.models import (
    ROBERTA_KLUE_Classifier,
    ROBERTA_S_Multi_Classifier,
)


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
