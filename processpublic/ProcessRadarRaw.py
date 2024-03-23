
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import os
import scipy
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA



root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def plot_radar_raw_signal(participant):
    fs = 2000
    radar_phase = __get_radar_phase(participant)

    x_seconds = len(radar_phase) / fs
    x_times = np.array([i for i in np.arange(0, x_seconds, x_seconds / len(radar_phase))])

    plt.figure(figsize=(12, 6))
    plt.plot(x_times, radar_phase, color='blue')
    plt.title('Radar Raw Signal Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def __get_radar_phase(participant):
    mat_data = __get_mat_data(participant)
    radar_i = None
    radar_q = None
    # Access variables in the loaded data
    name_i = 'radar_i'
    name_q = "radar_q"
    if name_i in mat_data and name_q in mat_data:
        radar_i = mat_data[name_i]
        radar_q = mat_data[name_q]
    radar_i = radar_i.squeeze()
    radar_q = radar_q.squeeze()
    radar_complex = radar_i + 1j * radar_q
    radar_phase = radar_complex

    return radar_phase


def __get_ecg_phase(participant):
    ecg_phase = None
    mat_data = __get_mat_data(participant)

    # Access variables in the loaded data
    variable_name = 'tfm_ecg1'
    if variable_name in mat_data:
        ecg_phase = mat_data[variable_name]

    ecg_phase = ecg_phase.squeeze()
    return ecg_phase


def __get_mat_data(participant):
    folder = "GDN000" if participant < 10 else "GDN00"
    mat_file_path = os.path.join(root_path, "publicdata/" + folder + str(participant) + "/" + folder + str(
        participant) + "_1_Resting.mat")
    mat_data = scipy.io.loadmat(mat_file_path)
    return mat_data


def process_radar_raw(participant):
    fs = 2000
    window_size = 5

    radar_phase = __get_radar_phase(participant)
    ecg_phase = __get_ecg_phase(participant)

    x_seconds = len(radar_phase) / fs
    pd_columns = ["fs_" + str(i) for i in range(1, fs * window_size + 1)]
    pd_columns.append("hr")
    pd_X_values = []

    for i in range(int(x_seconds) - window_size + 1):
        X_ecg_phase = ecg_phase[i * fs: (fs * window_size + i * fs)]
        ecg_result = ecg.ecg(signal=X_ecg_phase, sampling_rate=fs, show=False)  # show=True, plot the img.

        heart_rate = ecg_result['heart_rate']
        y_hr = round(np.mean(heart_rate))

        X_radar_phase = radar_phase[i * fs: (fs * window_size + i * fs)]
        row_data = X_radar_phase

        row_data = np.append(row_data, y_hr)

        pd_X_values.append(row_data)

    saved_directory = os.path.join("../publicdata/dataset/raw", "window_size_" + str(window_size))
    if not os.path.exists(saved_directory):
        os.makedirs(saved_directory)

    saved_file_path = os.path.join(saved_directory, "Train_raw_000" + str(participant) + "_Resting.csv")
    df = pd.DataFrame(pd_X_values, columns=pd_columns)

    df.to_csv(saved_file_path)

def prepare_train_val_data():
    parent_dir = str(Path.cwd().parent)
    dataset_directory = parent_dir + "/publicdata/dataset/"
    raw_dir = dataset_directory + "raw/window_size_5/"
    train_dir = dataset_directory + "train/individual"
    val_dir = dataset_directory + "val/individual"

    file_names = [file_name for file_name in os.listdir(raw_dir) if file_name.startswith('Train_raw_000')]
    file_names = sorted(file_names, key=lambda x: int(re.findall(r'\d+', x)[0]))
    num_val = 6

    # create and save data for validation
    for i in range(1, num_val+1):
        last_file_name = file_names.pop()
        _pca_to_csv(last_file_name, val_dir)
        print(f"validation done: {last_file_name}")

    # create and save data for training
    for index, file_name in enumerate(file_names):
        _pca_to_csv(file_name, train_dir)
        print(f"training done: {file_name}")



    print(file_names)


def _pca_to_csv(file_name, save_dir):
    parent_dir = str(Path.cwd().parent)
    dataset_directory = parent_dir + "/publicdata/dataset/"
    raw_dir = dataset_directory + "raw/window_size_5/"

    df_dataset = pd.read_csv(raw_dir + file_name)
    last_column = "fs_10000"
    df_X = df_dataset.loc[:, "fs_1":last_column]
    X = df_X.to_numpy()
    X_comp = np.complex64(X)
    X_train_r = X_comp.real
    X_train_i = X_comp.imag
    X = np.dstack((X_train_r, X_train_i)).reshape(X_train_r.shape[0], X_train_r.shape[1], 2)
    y = np.complex64(df_dataset["hr"]).real
    # Data dimensionality reduction
    X_flattened = X.reshape(len(X), -1)
    X_length = 590
    pca = PCA(n_components=X_length)
    X = pca.fit_transform(X_flattened)
    df_values = np.concatenate((X, y.reshape(len(y), 1)), axis=1)
    pd_columns = ["f_" + str(i) for i in range(1, X_length + 1)]
    pd_columns.append("hr")
    num_participant = int(re.findall(r'\d+', file_name)[0])
    saved_file_path = os.path.join(save_dir, "raw_" + str(num_participant) + "_Resting.csv")
    df = pd.DataFrame(df_values, columns=pd_columns)
    df.to_csv(saved_file_path)


# plot_radar_raw_signal(3)
# for i in range(4, 31):
#     process_radar_raw(i)
#     print(f'{i} done...')
prepare_train_val_data()