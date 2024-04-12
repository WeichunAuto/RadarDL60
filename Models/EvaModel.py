import os

import numpy as np
import torch
from pathlib import Path

from Models.BiLSTM.RadarBiLSTM import RadarBiLSTM
from Models.GRU.RadarGRU import RadarGRU
from Models.ModelNames import ModelNames
from Models.PrepareTrainData import PrepareTrainData
from Models.LSTM.RadarLSTM import RadarLSTM

import matplotlib.pyplot as plt

from Models.TPALSTM.RadarTpaLSTM import RadarTpaLSTM


class EvaModel:

    @staticmethod
    def preds_all_mases(model_name):
        # all participants ID
        participant_ids = [i for i in range(1, 31) if i != 3]
        MSEs = []
        MAEs = []
        for participant in participant_ids:
            MSE, MAE = EvaModel.preds_of_mase(model_name, participant)
            MSEs.append(MSE)
            MAEs.append(MAE)
        return MSEs, MAEs

    @staticmethod
    def preds_of_mase(model_name, participant):
        y_preds, y_real = EvaModel.get_model_preds(model_name, participant)

        # y_real_max = np.max(y_real)
        # y_real_min = np.min(y_real)
        # y_real_mean = np.mean(y_real)
        # print(f"y_real_max = {y_real_max}, y_real_min = {y_real_min}, y_real_mean = {y_real_mean}")

        MSE = round(np.mean((y_preds - y_real) ** 2), 2)
        RMSE = round(np.sqrt(MSE), 2)
        MAE = round(np.mean(np.abs(y_preds - y_real)), 2)
        # print(f"MSE = {MSE}, RMSE = {RMSE}, MAE = {MAE}")
        return MSE, MAE

    @staticmethod
    def plot_model_preds(model_name, participant):
        y_preds, y_real = EvaModel.get_model_preds(model_name, participant)

        plt.plot(y_real, color='green', label='Hr Reference')
        plt.plot(y_preds, color='orange', label='Hr Prediction', alpha=0.8)
        plt.xlabel("Seconds")
        plt.ylabel("HR")
        plt.grid()
        plt.legend()
        # plt.savefig(model_path + '.png')
        plt.show()

    @staticmethod
    def get_model_preds(model_name, participant):
        model = None
        base_directory = os.path.dirname(__file__)
        models_directory = os.path.join(base_directory, model_name, "trained_models")
        if model_name == ModelNames.LSTM.value:
            model = RadarLSTM()
        elif model_name == ModelNames.BiLSTM.value:
            model = RadarBiLSTM()
        model_file_name = [file_name for file_name in os.listdir(models_directory) if
                           file_name.startswith(model_name + "_model") and file_name.endswith(
                               "val_" + str(participant) + ".tar")][0]
        model_path = os.path.join(models_directory, model_file_name)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        val_loader = PrepareTrainData().val_dataloader(participant)
        y_real = None
        y_preds = None
        for index, batch in enumerate(val_loader):
            X_batch, y_batch = batch[0], batch[1]

            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

            with torch.inference_mode():  # 关闭 gradient
                preds_batch = model(X_batch)

            preds_batch = torch.round(preds_batch)

            if y_real is None:
                y_real = y_batch
                y_preds = preds_batch
            else:
                y_real = torch.cat((y_real, y_batch), dim=0)
                y_preds = torch.cat((y_preds, preds_batch), dim=0)
        y_real = y_real.numpy()
        y_preds = y_preds.numpy()
        y_real = y_real.reshape(1, len(y_real)).squeeze()
        y_preds = y_preds.reshape(1, len(y_preds)).squeeze()
        return y_preds, y_real


# model = RadarLSTM(n_features=118)
# model_path = "LSTM/lstm_best_t_model_20240328-00:18_0.0_.tar"

# model = RadarTpaLSTM(n_features=118)
# model_path = "TPALSTM/tpa-lstm_best_t_model_20240326-22:08.tar"

# model = RadarGRU(n_features=118)
# model_path = "GRU/gru_best_t_model_20240401-16:37_0.03_.tar"
model_name = ModelNames.LSTM.value
print(EvaModel.preds_all_mases(model_name))
