import torch
import torch.nn.functional as F

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

from tqdm import tqdm

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

# dataset = load_dataset(config)
dataset = load_dataset(
    config,
    dataset_file_path=str(path.raw_dataset_file_path),
)
# print(dataset.head())
train_dataset = CHDataset(dataset)
print(train_dataset.__getitem__(1))
trainloader = CHDataloader(config, train_dataset, tokenizer)
# print(trainloader)
trainloader = trainloader.get_dataloader()
# print(trainloader)
# for i, data in enumerate(trainloader):
#     print(data)
#     if i == 1:
#         break
model = get_model(config)
model = model.classifier
print(model)
optimizer = get_optimizer(model, config)
losses = []
accuracies = []
total_loss = 0.0
correct = 0
total = 0
# optimizer = AdamW(model.parameters(), lr=1e-5)
model.to(device)
for i in range(config.epochs):

    model.train()

    # for input_ids_batch, attention_masks_batch, batch['labels'] in tqdm(trainloader):
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        # print("batch type")
        # print(batch)
        # print(type(batch))

        # batch["labels"] = batch["labels"].to(device)
        batch["labels"].to(device)
        y_pred = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )[0]
        loss = F.cross_entropy(y_pred, batch["labels"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == batch["labels"]).sum()
        total += len(batch["labels"])

    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print(
        "Train Loss:", total_loss / total, "Accuracy:", correct.float() / total
    )


# LOGGER.info(train_dataset.__getitem__(10))
# # trainloader, validloader = get_train_valid_loader(config, )
# model = get_model(config)
# model = model.classifier
# optimizer = get_optimizer(model, config)
# LOGGER.info(optimizer)


# LOGGER.info(model)
