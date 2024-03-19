import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
import shutil

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

    def get_data(self, isEval=False):
        is_raw = "raw/"
        parent_dir = str(Path.cwd().parent.parent) if isEval is False else str(Path.cwd().parent)

        if isEval is True:
            file_name = "test_dataset.csv"
            parent_dir = parent_dir + "/publicdata/dataset/" + is_raw
            file_path = os.path.join(parent_dir, file_name)

            df = pd.read_csv(file_path)
            last_column = "fs_" + str(int(self.window_size * (self.fs / 10)))
            df_X_train = df.loc[:, "fs_1":last_column]

            X_train = df_X_train.to_numpy()
            y_train = df["hr"].to_numpy()

            return X_train, y_train

        df_dataset = None
        dataset_directory = parent_dir + "/publicdata/dataset/" + is_raw + "window_size_" + str(self.window_size)
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith('Train_raw_000')]
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df], ignore_index=True)
            # df_dataset = df_dataset.append(df, ignore_index=True) if df_dataset is not None else df
            print(f'file_name-{i} = {file_path}')

            if i > 0:
                break

        last_column = "fs_" + str(self.window_size*self.fs)
        df_X_train= df_dataset.loc[:, "fs_1":last_column]

        X_train = df_X_train.to_numpy()

        y_train = np.complex64(df_dataset["hr"]).real

        X_train_comp = np.complex64(X_train)
        X_train_r = X_train_comp.real
        X_train_i = X_train_comp.imag
        X_train = np.dstack((X_train_r, X_train_i)).reshape(X_train_r.shape[0], X_train_r.shape[1], 2)

        return X_train, y_train

    def load_data(self, isEval=False, is_normalize=False):
        X_data, y_data = self.get_data(isEval=isEval)

        split_index = int(len(X_data) * 0.8)
        y_data = y_data.reshape(len(y_data), 1)

        # Data dimensionality reduction
        X_flattened = X_data.reshape(len(X_data), -1)
        X_length = int((self.window_size * self.fs)/10)
        pca = PCA(n_components=X_length)
        X_data = pca.fit_transform(X_flattened)

        if is_normalize is True:
            flattened_data = X_data.flatten()
            flattened_data = flattened_data.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(-10, 10))
            normalized_data = scaler.fit_transform(flattened_data)

            X_data = normalized_data.reshape(X_data.shape)

        X_data = X_data.reshape(len(X_data), self.window_size, int(self.fs/10))

        X_train = X_data[:split_index]
        y_train = y_data[:split_index]

        X_test = X_data[split_index:]
        y_test = y_data[split_index:]
        self.y_test = y_test

        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()

        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        train_dataset = RadarDataset(X_train, y_train)
        test_dataset = RadarDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        if isEval is False:
            self.cache_test_data(test_loader)

        return train_loader, test_loader

    def cache_test_data(self, test_loader):
        new_data_loader = DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))

        X_batch, y_batch = next(iter(new_data_loader))
        pd_columns = ["fs_" + str(i) for i in range(1, X_batch.size(1) * X_batch.size(2) + 1)]
        pd_columns.append("hr")
        X_batch = X_batch.reshape(len(X_batch), X_batch.size(1) * X_batch.size(2))

        test_data = np.concatenate((X_batch, y_batch), axis=1)

        is_raw = "raw/"
        parent_dir = str(Path.cwd().parent.parent)
        cache_directory = parent_dir + "/publicdata/dataset/" + is_raw
        file_name = "test_dataset.csv"
        file_path = os.path.join(cache_directory, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)

        saved_file_path = os.path.join(cache_directory, file_name)
        df = pd.DataFrame(test_data, columns=pd_columns)

        df.to_csv(saved_file_path)


# X_data, y_data = PrepareTrainData(is_shuffle=False).get_data(show=True, isEval=False, is_complex=True)