import torch

from torch_prac.logger import get_logger
from torch_prac.config import parser_args, CHPath
from torch_prac.dataloader.dataset import CHDataset
from torch_prac.dataloader.dataloader import CHDataloader

from torch_prac.utils import (
    get_model,
    get_tokenizer,
    get_optimizer,
    save_tokenizer,
    load_dataset,
    get_train_valid_loader,
)

LOGGER = get_logger()
config = parser_args()
path = CHPath(config)
LOGGER.info(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(device)

tokenizer = get_tokenizer(config)
save_tokenizer(tokenizer, path.save_model_folder_path)
# TODO
# save_tokenizer

dataset = load_dataset(config)
# print(dataset.head())
train_dataset = CHDataset(dataset)
print(train_dataset.__getitem__(1))
trainloader = CHDataloader(config, train_dataset, tokenizer)
# print(trainloader)
trainloader = trainloader.get_dataloader()
print(trainloader)
for i, data in enumerate(trainloader):
    print(data)
    if i == 1:
        break


# LOGGER.info(train_dataset.__getitem__(10))
# # trainloader, validloader = get_train_valid_loader(config, )
# model = get_model(config)
# model = model.classifier
# optimizer = get_optimizer(model, config)
# LOGGER.info(optimizer)


# LOGGER.info(model)
