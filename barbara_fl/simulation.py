from enum import Enum
from typing import Optional, TypedDict, Union

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class Algorithm(Enum):
    FedAVG = "fedavg"


class Paths(TypedDict):
    train: list[str]
    test: list[str]


class Config(TypedDict):
    num_rounds: int
    num_clients: int
    data_paths: Paths
    algorithm: Algorithm


class ConfigOptional(TypedDict, total=False):
    columns_to_remove: list[str]
    batch_size: int


Scaler = Union[MinMaxScaler]


def get_df(path: str, columns_to_remove: list[str] = []) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow").drop(columns=columns_to_remove)


def standardize_df(df: pd.DataFrame, scaler: Scaler):
    df[df.columns] = scaler.transform(df[df.columns])


def get_x_y_values(df: pd.DataFrame):
    return df.iloc[:, :-1].values, df.iloc[:, -1:].values


def get_data_for_loader(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> dict:
    data = {}
    if train_df is not None:
        x, y = get_x_y_values(train_df)
        data.update({"x_train": x, "y_train": y})
    if test_df is not None:
        x, y = get_x_y_values(test_df)
        data.update({"x_test": x, "y_test": y})
    return data


def get_train_and_test_loaders(
    data: dict, batch_size
) -> tuple[DataLoader, DataLoader]:
    x_train_tensor = torch.tensor(data["x_train"], dtype=torch.float32)
    y_train_tensor = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test_tensor = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader


class Simulation:
    def __init__(
        self, config: Config, config_optional: Optional[ConfigOptional] = None
    ) -> None:
        self.num_clients = config["num_clients"]
        self.num_rounds = config["num_rounds"]
        self.train_paths = config["data_paths"]["train"]
        self.test_paths = config["data_paths"]["test"]
        self.config_optional = config_optional
        self.scaler = MinMaxScaler
        self.columns_to_remove = (config_optional or {}).get(
            "columns_to_remove", []
        )
        self.batch_size = (config_optional or {}).get("batch_size", 128)

    def start(self):
        return self.__get_data_loaders()

    def __get_data_loaders(self):
        # 1. load both train and test dataframe from a client
        # 2. standardize them
        # 3. get the values
        # 4. load them into tensors
        # 5. create data loaders
        loaders: list[tuple[DataLoader, DataLoader]] = []
        for train_path, test_path in zip(self.train_paths, self.test_paths):
            train_df = get_df(train_path, self.columns_to_remove)
            test_df = get_df(test_path, self.columns_to_remove)
            scaler = self.scaler().fit(train_df)
            standardize_df(train_df, scaler)
            standardize_df(test_df, scaler)
            data = get_data_for_loader(train_df, test_df)
            loaders.append(get_train_and_test_loaders(data, self.batch_size))
        return loaders

    def read_files(self): ...
