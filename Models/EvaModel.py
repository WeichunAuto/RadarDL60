import os

import numpy as np
import torch
import pandas as pd

from Models.BiLSTM.RadarBiLSTM import RadarBiLSTM
from Models.DILATE.NetGRU import NetGRU
from Models.GRU.RadarGRU import RadarGRU
from Models.ModelNames import ModelNames
from Models.NBEATS.NBeats import NBeats
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
        print(f"MSE = {MSE}, RMSE = {RMSE}, MAE = {MAE}")
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
        elif model_name == ModelNames.GRU.value:
            model = RadarGRU()
        elif model_name == ModelNames.DILATE.value:
            model = NetGRU()
        elif model_name == ModelNames.NBEATS.value:
            model = NBeats()
        elif model_name == ModelNames.TPALSTM.value:
            model = RadarTpaLSTM()

        model_file_name = [file_name for file_name in os.listdir(models_directory) if file_name.startswith(model_name + "_model") and file_name.endswith("val_" + str(participant) + ".tar")][0]

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

    @staticmethod
    def plot_mase_of_all_models(method):
        fig, ax = plt.subplots()
        # plt.figure(figsize=(10, 3))
        participant_ids = [i for i in range(1, 31) if i != 3]
        model_name_all = list(ModelNames)
        for i in range(len(model_name_all)):
            model_name = model_name_all[i].value

            # if model_name == ModelNames.LSTM.value or model_name == ModelNames.BiLSTM.value or model_name == ModelNames.GRU.value:
            #     continue

            MSEs, MAEs = EvaModel.preds_all_mases(model_name)
            if method.lower() == "mae":
                ax.plot(participant_ids, MAEs, label=model_name, marker='o')
            else:
                ax.plot(participant_ids, MSEs, label=model_name, marker='o')
            print(f"{model_name} done...")

        ax.set_xlabel("Participants ID")
        ax.set_ylabel(method.upper() + " values")
        ax.set_title(method.upper() + " Values for Each Participant.")
        ax.legend()
        plt.xticks(participant_ids)
        plt.show()

    @staticmethod
    def plot_loss_of_all_models(type, participant):
        epoches = [i for i in range(500)]
        fig, ax = plt.subplots()
        base_directory = os.path.dirname(__file__)

        column_plot = "t_loss" if type == "t" else "v_loss"

        model_name_all = list(ModelNames)

        for i in range(len(model_name_all)):
            model_name = model_name_all[i].value

            # if model_name != ModelNames.LSTM.value and model_name != ModelNames.GRU.value:
            #     continue

            models_directory = os.path.join(base_directory, model_name, "trained_models")
            loss_file_name = "loss_" + model_name + ".csv"
            loss_file_path = os.path.join(models_directory, loss_file_name)
            df = pd.read_csv(loss_file_path)
            filtered_participant = (df["p_id"] == participant)
            y_values = df.loc[filtered_participant, [column_plot]].to_numpy()

            ax.plot(epoches, y_values, label=model_name)

        label_name = "training loss" if type == "t" else "validation loss"
        ax.set_xlabel("epoches")
        ax.set_ylabel(label_name)
        ax.set_title("The " + label_name + "of different models for participant " + str(participant))
        ax.legend()
        plt.show()


# model = RadarLSTM(n_features=118)
# model_path = "LSTM/lstm_best_t_model_20240328-00:18_0.0_.tar"

# model = RadarTpaLSTM(n_features=118)
# model_path = "TPALSTM/tpa-lstm_best_t_model_20240326-22:08.tar"

# model = RadarGRU(n_features=118)
# model_path = "GRU/gru_best_t_model_20240401-16:37_0.03_.tar"

EvaModel.plot_mase_of_all_models("mse")
# EvaModel.plot_loss_of_all_models("t", 1)


# model_name = ModelNames.NBEATS.value
# EvaModel.plot_model_preds(model_name, 1)
# EvaModel.preds_of_mase(model_name, 1)