import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

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

    def get_data(self, show=True, is_complex=False):
        df_dataset = None
        is_raw = "" if is_complex is False else "raw/"
        dataset_directory = str(Path.cwd().parent.parent) + "/publicdata/dataset/" + is_raw + "window_size_" + str(self.window_size)
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith('Train_raw_000')]
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df])
            print(f'file_name-{i} = {file_path}')

            if i > 0:
                break

        last_column = "fs_" + str(self.window_size*self.fs)
        df_X_train= df_dataset.loc[:, "fs_1":last_column]

        X_train = df_X_train.to_numpy()

        if is_complex is False:
            df_y_train = df_dataset["hr"].astype("int32")
            y_train = df_y_train.to_numpy()
        else:
            y_train = np.complex64(df_dataset["hr"]).real

            X_train_comp = np.complex64(X_train)
            X_train_r = X_train_comp.real
            X_train_i = X_train_comp.imag
            X_train = np.dstack((X_train_r, X_train_i)).reshape(X_train_r.shape[0], X_train_r.shape[1], 2)

        if show is True:
            df_y_train.value_counts().plot(kind="bar")
            plt.xticks(rotation=90)
            plt.show()

        return X_train, y_train

    def load_data(self, is_complex=False):
        X_data, y_data = self.get_data(show=False, is_complex=is_complex)

        split_index = int(len(X_data) * 0.8)
        y_data = y_data.reshape(len(y_data), 1)

        # Data dimensionality reduction
        X_flattened = X_data.reshape(len(X_data), -1)
        pca = PCA(n_components=int((self.window_size * self.fs)/10))
        X_data = pca.fit_transform(X_flattened)

        X_data = X_data.reshape(len(X_data), self.window_size, int(self.fs/10)) if is_complex is True else X_data.reshape(len(X_data), self.window_size, self.fs)

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

        return train_loader, test_loader