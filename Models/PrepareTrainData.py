import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path


class RadarDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class PrepareTrainData:
    def __init__(self, window_size=5, fs=2000, batch_size=16, is_shuffle=False):
        self.window_size = window_size
        self.fs = fs
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def get_trainval_data(self, is_train=True):
        parent_dir = str(Path.cwd().parent.parent)

        df_dataset = None
        sub_folder = "train/" if is_train is True else "val/"
        dataset_directory = parent_dir + "/publicdata/dataset/" + sub_folder
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith("raw_")]

        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df], ignore_index=True)
            print(f'file_name-{i} = {file_path}')

            if i > 0:
                break

        last_column = "f_590"
        df_X= df_dataset.loc[:, "f_1":last_column]

        X_data = df_X.to_numpy()
        y_data = df_dataset["hr"].to_numpy()

        return X_data, y_data

    def get_test_data(self, participant):
        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/dataset/test/"
        file_name = "raw_" + str(participant) + "_Resting.csv"
        file_path = os.path.join(dataset_directory, file_name)
        df = pd.read_csv(file_path)
        last_column = "f_590"
        df_X_val = df.loc[:, "f_1":last_column]

        X_val = df_X_val.to_numpy()
        y_val = df["hr"].to_numpy()

        return X_val, y_val

    def train_dataloader(self):
        X_train, y_train = self.get_trainval_data()
        return self.get_dataloader(X_train, y_train)

    def val_dataloader(self):
        X_val, y_val = self.get_trainval_data(is_train=False)
        return self.get_dataloader(X_val, y_val)

    def test_dataloader(self, participant):
        X_test, y_test = self.get_test_data(participant)
        return self.get_dataloader(X_test, y_test)

    def get_dataloader(self, X_data, y_data):
        X_data = X_data.reshape(len(X_data), self.window_size, 118)
        y_data = y_data.reshape(len(y_data), 1)
        X_data = torch.tensor(X_data).float()
        y_data = torch.tensor(y_data).float()
        dataset = RadarDataset(X_data, y_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)
        return dataloader



   # X_data, y_data = PrepareTrainData(is_shuffle=False).get_data(show=True, isEval=False, is_complex=True)