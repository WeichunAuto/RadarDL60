import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import re


class RadarDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class PrepareTrainData:
    def __init__(self, seq_length=5, n_features=118, batch_size=16, is_shuffle=False):
        self.seq_length = seq_length
        self.n_features = n_features
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def get_trainval_data(self, is_train=True):
        parent_dir = str(Path.cwd().parent.parent)

        df_dataset = None
        sub_folder = "train/" if is_train is True else "val/"
        dataset_directory = parent_dir + "/publicdata/Dataset/" + sub_folder
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith("raw_")]

        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df], ignore_index=True)
            print(f'file_name-{i} = {file_path}')

            if i > 3:
                break

        last_column = "f_590"
        df_X= df_dataset.loc[:, "f_1":last_column]

        X_data = df_X.to_numpy()
        y_data = df_dataset["hr"].to_numpy()
        print(f'x_data length = {X_data.shape}')

        return X_data, y_data

    def get_val_data(self, participant):
        # parent_dir = str(Path.cwd().parent)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dataset_directory = parent_dir + "/publicdata/Dataset/cross_train/"
        file_name = "raw_" + str(participant) + "_Resting.csv"
        file_path = os.path.join(dataset_directory, file_name)
        df = pd.read_csv(file_path)
        last_column = "f_590"
        df_X_val = df.loc[:, "f_1":last_column]

        X_val = df_X_val.to_numpy()
        y_val = df["hr"].to_numpy()

        return X_val, y_val

    def get_test_data(self, participant):
        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/Dataset/test/"
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

    def val_dataloader(self, participant):
        X_val, y_val = self.get_val_data(participant)
        return self.get_dataloader(X_val, y_val)

    def test_dataloader(self, participant):
        X_test, y_test = self.get_test_data(participant)
        return self.get_dataloader(X_test, y_test)

    def get_dataloader(self, X_data, y_data):
        X_data = X_data.reshape(len(X_data), self.seq_length, self.n_features)
        y_data = y_data.reshape(len(y_data), 1)
        X_data = torch.tensor(X_data).float()
        y_data = torch.tensor(y_data).float()
        dataset = RadarDataset(X_data, y_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)
        return dataloader

    def get_cross_dataloaders(self, participant_id):
        '''
        Prepare dataloader for cross validation.
        :param participant_id:
        :return:
        '''
        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/Dataset/cross_train/"
        val_file_name = "raw_" + str(participant_id) + "_Resting.csv"
        df_dataset = None

        # prepare train dataloader
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith("raw_") and file_name != val_file_name]
        file_names = sorted(file_names, key=lambda x: int(re.findall(r'\d+', x)[0]))
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            # print(f'training file_name-{i} = {file_path}')
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df], ignore_index=True)

            if i>1:
                break

        last_column = "f_590"
        df_train_X = df_dataset.loc[:, "f_1":last_column]

        Train_X_data = df_train_X.to_numpy()
        Train_y_data = df_dataset["hr"].to_numpy()
        # print(f'Train_X_data shape = {Train_X_data.shape}, Train_y_data shape = {Train_y_data.shape}')
        train_X = Train_X_data.reshape(len(Train_X_data), self.seq_length, self.n_features)
        train_y = Train_y_data.reshape(len(Train_y_data), 1)
        train_X = torch.tensor(train_X).float()
        train_y = torch.tensor(train_y).float()
        train_dataset = RadarDataset(train_X, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        # prepare validation train dataloader
        df_val = pd.read_csv(os.path.join(dataset_directory, val_file_name))

        df_val_X = df_val.loc[:, "f_1":last_column]
        val_X_data = df_val_X.to_numpy()
        val_y_data = df_val["hr"].to_numpy()
        # print(f'val_X_data shape = {val_X_data.shape}, val_y_data shape = {val_y_data.shape}')
        val_X = val_X_data.reshape(len(val_X_data), self.seq_length, self.n_features)
        val_y = val_y_data.reshape(len(val_y_data), 1)
        val_X = torch.tensor(val_X).float()
        val_y = torch.tensor(val_y).float()
        val_dataset = RadarDataset(val_X, val_y)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        return train_dataloader, val_dataloader


# PrepareTrainData().process_data_for_idea_noise()
