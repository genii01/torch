import torch

from torch_prac.logger import get_logger
from torch_prac.config import parser_args, CHPath

from torch_prac.utils import (
    get_model,
    get_tokenizer,
    get_optimizer,
    save_tokenizer,
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


model = get_model(config)
model = model.classifier
optimizer = get_optimizer(model, config)
LOGGER.info(optimizer)
# LOGGER.info(model)
