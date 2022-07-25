from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple


def parser_args() -> ArgumentParser:
    main_parser = ArgumentParser()
    main_parser.add_argument("--seed", default=1004, type=int)
    main_parser.add_argument(
        "--project_name", default="sentence theme classification", type=str
    )
    main_parser.add_argument("--task", default="multiclassification", type=str)
    main_parser.add_argument("--root_path", default="./torch_prac", type=str)
    main_parser.add_argument(
        "--save_model_folder_path", default="save_model", type=str
    )
    main_parser.add_argument(
        "--raw_dataset_folder_path", default="data", type=str
    )
    main_parser.add_argument("--train_file", default="train_data.csv", type=str)

    main_parser.add_argument("--model", default="s_multi_roberta", type=str)
    main_parser.add_argument("--epochs", default=10, type=int)
    main_parser.add_argument("--optimizer", default="AdamW", type=str)
    main_parser.add_argument("--num_classes", default=7, type=int)
    main_parser.add_argument("--lr", default=1e-5, type=float)
    main_parser.add_argument("--batch_size", default=32, type=int)

    main_parser.add_argument("--max_length", default=25, type=int)

    return main_parser.parse_args()


class CHPath:
    def __init__(self, config: Namespace):
        self.config = config
        self.root_path = Path(self.config.root_path)
        self.setup_directory()

    def setup_directory(self):
        self.raw_dataset_folder_path.mkdir(parents=True, exist_ok=True)
        self.save_model_folder_path.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dataset_folder_path(self) -> Path:
        return self.root_path / self.config.raw_dataset_folder_path

    @property
    def raw_dataset_file_path(self) -> Path:
        return self.raw_dataset_folder_path / self.config.train_file

    @property
    def preprocessed_dataset_file_path(self) -> Path:
        return self.raw_dataset_folder_path / str(
            "preprocessed_" + self.config.train_file.split(".")[0] + ".ftr"
        )

    @property
    def save_model_folder_path(self) -> Path:
        return self.root_path / self.config.save_model_folder_path
