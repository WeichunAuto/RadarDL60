import os

import numpy as np
import torch
import pandas as pd

from Models.BiLSTM.RadarBiLSTM import RadarBiLSTM
from Models.DILATE.NetGRU import NetGRU
from Models.GRU.RadarGRU import RadarGRU
from Models.ModelNames import ModelNames, PlotMarker
from Models.NBEATS.NBeats import NBeats
from Models.PrepareTrainData import PrepareTrainData
from Models.LSTM.RadarLSTM import RadarLSTM
from Models.CnnLSTM.CnnLSTM import CnnLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from Models.TPALSTM.RadarTpaLSTM import RadarTpaLSTM


class EvaModel:

    @staticmethod
    def plot_average_metric_for_each_model():
        model_name_all = list(ModelNames)
        mse_mean_array = []
        rmse_mean_array = []
        mae_mean_array = []
        r2_mean_array = []

        size = len(model_name_all)
        MSE_mean_str = ""
        RMSE_mean_str = ""
        MAE_mean_str = ""
        R2_mean_str = ""

        for i in range(size):
            model_name = model_name_all[i].value
            MSEs, MAEs, RMSEs, R2s = EvaModel.preds_all_mases(model_name)

            MSE_mean = sum(MSEs) / len(MSEs)
            RMSE_mean = sum(RMSEs) / len(RMSEs)
            MAE_mean = sum(MAEs) / len(MAEs)
            R2_mean = sum(R2s) / len(R2s)

            MSE_mean_str = MSE_mean_str + " & " + str(round(MSE_mean, 4))
            RMSE_mean_str = RMSE_mean_str + " & " + str(round(RMSE_mean, 4))
            MAE_mean_str = MAE_mean_str + " & " + str(round(MAE_mean, 4))
            R2_mean_str = R2_mean_str + " & " + str(round(R2_mean, 4))

            mse_mean_array.append(MSE_mean)
            rmse_mean_array.append(RMSE_mean)
            mae_mean_array.append(MAE_mean)
            r2_mean_array.append(R2_mean)

        print(f"\nMSE_mean_str = {MSE_mean_str}")
        print(f"RMSE_mean_str = {RMSE_mean_str}")
        print(f"MAE_mean_str = {MAE_mean_str}")
        print(f"R2_mean_str = {R2_mean_str}")

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8), sharex=False)
        fig.subplots_adjust(wspace=0.2, hspace=0.4)
        x = np.arange(-0.2, size - 1)
        total_width, n = 1.4, 2
        width = total_width / n

        ax[0, 0].bar(x, mse_mean_array, width=width, label="MSE Mean", color='orange')  #
        ax[0, 1].bar(x, rmse_mean_array, width=width, label="RMSE Mean", color='green')
        ax[1, 0].bar(x, mae_mean_array, width=width, label="MAE Mean", color='blue')
        ax[1, 1].bar(x, r2_mean_array, width=width, label=r"$R^2$ Mean", color='purple')

        xticks = [model_name_all[i].value for i in range(size)]
        xline = [i for i in np.arange(size)]

        ax[0, 0].set_xlabel("Model Name")
        ax[0, 0].set_ylabel("MSE Average Values")
        ax[0, 0].set_title("MSE Average Values for All Participants.")
        # ax[0,0].tick_params(axis='x', rotation=45)
        ax[0, 0].legend(loc='lower right')
        ax[0, 0].set_xticks(xline, xticks)

        ax[0, 1].set_xlabel("Model Names")
        ax[0, 1].set_ylabel("RMSE Average Values")
        ax[0, 1].set_title("RMSE Average Values for All Participants.")
        ax[0, 1].legend(loc='lower right')
        ax[0, 1].set_xticks(xline, xticks)

        ax[1, 0].set_xlabel("Model Names")
        ax[1, 0].set_ylabel("MAE Average Values")
        ax[1, 0].set_title("MAE Average Values for All Participants.")
        ax[1, 0].legend(loc='lower right')
        ax[1, 0].set_xticks(xline, xticks)

        ax[1, 1].set_xlabel("Model Names")
        ax[1, 1].set_ylabel(r"$R^2$ Average Values")
        ax[1, 1].set_title(r"$R^2$ Average Values for All Participants.")
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_xticks(xline, xticks)

        plt.show()

    @staticmethod
    def preds_all_mases(model_name):
        # all participants ID
        participant_ids = [i for i in range(1, 31) if i != 3]
        # start_id = 30
        # participant_ids = [i for i in range(start_id, start_id+1) if i != 3]
        MSEs = []
        MAEs = []
        RMSEs = []
        R2s = []
        for participant in participant_ids:
            MSE, MAE, RMSE, R2 = EvaModel.preds_of_mase(model_name, participant)
            MSE = np.round(MSE, 4)
            RMSE = np.round(RMSE, 4)
            MAE = np.round(MAE, 4)
            R2 = np.round(R2, 4)

            MSEs.append(MSE)
            MAEs.append(MAE)
            RMSEs.append(RMSE)
            R2s.append(R2)

            print(f"model_name = {model_name}, participant = {participant}")
            # print(f"participant = {participant}, model = {model_name}, MSE = " + str(MSE))
            # print(f"participant = {participant}, model = {model_name}, RMSE = " + str(RMSE))
            # print(f"participant = {participant}, model = {model_name}, MAE = " + str(MAE))
            # print(f"participant = {participant}, model = {model_name}, R2 = " + str(R2))

        # print(f"{model_name} MSE mean: {sum(MSEs)/len(MSEs)}")
        # print(f"{model_name} MAE mean: {sum(MAEs) / len(MAEs)}")
        # print(f"{model_name} RMSE mean: {sum(RMSEs) / len(RMSEs)}")
        # print(f"{model_name} R2 mean: {sum(R2s) / len(R2s)}")
        return MSEs, MAEs, RMSEs, R2s

    @staticmethod
    def preds_of_mase(model_name, participant):
        y_preds, y_real = EvaModel.get_model_preds(model_name, participant)

        # MSE = round(np.mean((y_preds - y_real) ** 2), 2)
        MSE = mean_squared_error(y_real, y_preds)
        RMSE = np.sqrt(MSE)
        # MAE = round(np.mean(np.abs(y_preds - y_real)), 2)
        MAE = mean_absolute_error(y_real, y_preds)
        R2 = r2_score(y_real, y_preds)
        return MSE, MAE, RMSE, R2

    @staticmethod
    def plot_model_preds(model_name, participant):
        y_preds, y_real = EvaModel.get_model_preds(model_name, participant)

        MSE = round(np.mean((y_preds - y_real) ** 2), 2)
        print(f'MSE = {MSE}')

        plt.plot(y_real, color='green', label='Hr Reference')
        plt.plot(y_preds, color='orange', label='Hr Prediction', alpha=0.8)
        plt.xlabel("Seconds")
        plt.ylabel("HR")
        plt.grid()
        plt.legend()
        # plt.savefig(model_path + '.png')
        plt.show()

    @staticmethod
    def plot_model_preds_for_participant_on_traval(model_name, participant):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4), sharey=True)
        fig.subplots_adjust(wspace=0.2, hspace=0.4)

        y_preds, y_real = EvaModel.get_model_preds(model_name, participant, type='t')
        plot_period = 200
        y_preds = y_preds[0:plot_period]
        y_real = y_real[0:plot_period]
        ax[0].plot(y_real, color='green', label='HR Reference Wave')
        ax[0].plot(y_preds, color='orange', label='HR Prediction Wave', alpha=0.8)
        ax[0].set_xlabel("Seconds")
        ax[0].set_ylabel("HR Values")
        ax[0].set_title("Prediction Results Using Participant 12 \nas the Training Dataset")
        ax[0].legend(loc='upper right')

        y_preds, y_real = EvaModel.get_model_preds(model_name, participant)
        y_preds = y_preds[0:plot_period]
        y_real = y_real[0:plot_period]
        ax[1].plot(y_real, color='green', label='HR Reference Wave')
        ax[1].plot(y_preds, color='orange', label='HR Prediction Wave', alpha=0.8)
        ax[1].set_xlabel("Seconds")
        ax[1].set_ylabel("HR Values")
        ax[1].set_title("Prediction Results Using Participant 12 \nas the Validation Dataset")
        ax[1].legend(loc='upper right')

        plt.show()


    @staticmethod
    def get_idea_model_preds(model_name):
        model = None
        base_directory = os.path.dirname(__file__)
        models_directory = os.path.join(base_directory, model_name, "trained_models", "idea_noise")
        if model_name == ModelNames.LSTM.value:
            model = RadarLSTM()
        elif model_name == ModelNames.BiLSTM.value:
            model = RadarBiLSTM()
        model_file_name = [file_name for file_name in os.listdir(models_directory) if
                           file_name.startswith(model_name + "_model") and file_name.endswith(
                               "val_-1.tar")][0]
        model_path = os.path.join(models_directory, model_file_name)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        val_loader = PrepareTrainData().get_idea_test_dataloader()
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

        MSE = mean_squared_error(y_real, y_preds)
        RMSE = np.sqrt(MSE)
        # MAE = round(np.mean(np.abs(y_preds - y_real)), 2)
        MAE = mean_absolute_error(y_real, y_preds)
        R2 = r2_score(y_real, y_preds)
        print(f"MSE = {MSE}, RMSE = {RMSE}, MAE = {MAE}, R2= {R2}")

        # return y_preds, y_real

    @staticmethod
    def get_model_preds(model_name, participant, type="c"):
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
        # elif model_name == ModelNames.HARHN.value:
        #     model = HARHN()
        elif model_name == ModelNames.CnnLSTM.value:
            model = CnnLSTM()

        model_file_name = [file_name for file_name in os.listdir(models_directory) if
                           file_name.startswith(model_name + "_model") and file_name.endswith(
                               "val_" + str(participant if type == 'c' else (participant + 1)) + ".tar")][0]

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
        if method.lower() == "all":
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8), sharex=False)
            fig.subplots_adjust(wspace=0.2, hspace=0.4)
        else:
            fig, ax = plt.subplots()
        # plt.figure(figsize=(10, 3))
        participant_ids = [i for i in range(1, 31) if i != 3]
        model_name_all = list(ModelNames)
        for i in range(len(model_name_all)):
            model_name = model_name_all[i].value

            MSEs, MAEs, RMSEs, R2s = EvaModel.preds_all_mases(model_name)
            marker = PlotMarker[model_name].value
            if method.lower() == "mae":
                ax.plot(participant_ids, MAEs, label=model_name, linewidth=1, marker=marker)
            elif method.lower() == "mse":
                ax.plot(participant_ids, MSEs, label=model_name, linewidth=1, marker=marker)
            elif method.lower() == "rmse":
                ax.plot(participant_ids, RMSEs, label=model_name, linewidth=1, marker=marker)
            elif method.lower() == "r2":
                ax.plot(participant_ids, R2s, label=model_name, linewidth=1, marker=marker)
            elif method.lower() == "all":
                ax[0, 0].plot(participant_ids, MSEs, label=model_name, linewidth=0.8, marker=marker)
                ax[0, 1].plot(participant_ids, RMSEs, label=model_name, linewidth=0.8, marker=marker)
                ax[1, 0].plot(participant_ids, MAEs, label=model_name, linewidth=0.8, marker=marker)
                ax[1, 1].plot(participant_ids, R2s, label=model_name, linewidth=0.8, marker=marker)

            print(f"{model_name} done...")

        if method.lower() == "all":
            me_name = "MSE"
            ax[0, 0].set_xlabel("Participants' IDs")
            ax[0, 0].set_ylabel(me_name + " Values")
            ax[0, 0].set_title(me_name + " Metrics for Each Participant.")
            # ax[0,0].tick_params(axis='x', rotation=45)
            ax[0, 0].legend(loc='upper right')
            ax[0, 0].set_xticks(participant_ids)

            me_name = "RMSE"
            ax[0, 1].set_xlabel("Participants' IDs")
            ax[0, 1].set_ylabel(me_name + " Values")
            ax[0, 1].set_title(me_name + " Metrics for Each Participant.")
            # ax[0,1].tick_params(axis='x', rotation=45)
            ax[0, 1].legend(loc='upper right')
            ax[0, 1].set_xticks(participant_ids)

            me_name = "MAE"
            ax[1, 0].set_xlabel("Participants' IDs")
            ax[1, 0].set_ylabel(me_name + " Values")
            ax[1, 0].set_title(me_name + " Metrics for Each Participant.")
            # ax[1, 0].tick_params(axis='x', rotation=45)
            ax[1, 0].legend(loc='upper right')
            ax[1, 0].set_xticks(participant_ids)

            me_name = "$R^{2}$"
            ax[1, 1].set_xlabel("Participants' IDs")
            ax[1, 1].set_ylabel(r"" + me_name + " Values")
            ax[1, 1].set_title(r"" + me_name + " Metrics for Each Participant.")
            # ax[1,1].tick_params(axis='x', rotation=45)
            ax[1, 1].legend(loc='lower right')
            ax[1, 1].set_xticks(participant_ids)

        else:
            ax.set_xlabel("Participants' IDs")
            ax.set_ylabel(method.upper() + " Values")
            ax.set_title(method.upper() + " Metrics for Each Participant.")
            ax.legend()
        # font = fm.FontProperties(size=12)
        # plt.xticks(participant_ids)
        plt.show()
        # plt.plot()

    @staticmethod
    def plot_loss_of_all_models(type, participant):
        epoches = [i for i in range(500)]
        fig, ax = plt.subplots()
        base_directory = os.path.dirname(__file__)

        column_plot = "t_loss" if type == "t" else "v_loss"

        model_name_all = list(ModelNames)

        for i in range(len(model_name_all)):
            model_name = model_name_all[i].value

            models_directory = os.path.join(base_directory, model_name, "trained_models")
            loss_file_name = "loss_" + model_name + ".csv"
            loss_file_path = os.path.join(models_directory, loss_file_name)
            df = pd.read_csv(loss_file_path)
            filtered_participant = (df["p_id"] == participant)
            y_values = df.loc[filtered_participant, [column_plot]].to_numpy()

            ax.plot(epoches, y_values, label=model_name, linewidth=.8)

        label_name = "Training loss" if type == "t" else "validation loss"
        ax.set_xlabel("Epoches")
        ax.set_ylabel(label_name)
        ax.set_title("The " + label_name + "of different models for participant " + str(participant))
        ax.legend()
        plt.show()

    @staticmethod
    def plot_mean_loss_of_all_models(type):
        epoches = [i for i in range(500)]
        participant_ids = [i for i in range(1, 31) if i != 3]

        base_directory = os.path.dirname(__file__)

        if type == "t":
            column_plot = ["t_loss"]
            fig, ax = plt.subplots(figsize=(10, 5))
        elif type == "v":
            column_plot = ["v_loss"]
            fig, ax = plt.subplots(figsize=(10, 5))
        elif type == "a":
            column_plot = ["t_loss", "v_loss"]
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=False)
            fig.subplots_adjust(wspace=0.4)

        model_name_all = list(ModelNames)

        for i in range(len(model_name_all)):
            model_name = model_name_all[i].value

            loss_values = np.array([])
            t_loss_values = np.array([])
            v_loss_values = np.array([])

            models_directory = os.path.join(base_directory, model_name, "trained_models")
            loss_file_name = "loss_" + model_name + ".csv"
            loss_file_path = os.path.join(models_directory, loss_file_name)
            df = pd.read_csv(loss_file_path)

            for idx, participant in enumerate(participant_ids):
                filtered_participant = (df["p_id"] == participant)
                y_values = df.loc[filtered_participant, [column_plot[0]]].to_numpy()
                if len(y_values) > 500:
                    y_values = y_values[0:500]

                # Boolean indexing to identify elements greater than 10
                mask = y_values > 20000
                y_values[mask] = 20000
                loss_values = y_values if loss_values.size == 0 else loss_values + y_values

                if type == "a":
                    t_y_values = df.loc[filtered_participant, [column_plot[0]]].to_numpy()
                    v_y_values = df.loc[filtered_participant, [column_plot[1]]].to_numpy()

                    if len(t_y_values) > 500:
                        t_y_values = t_y_values[0:500]
                        v_y_values = v_y_values[0:500]

                    mask_t = t_y_values > 20000
                    t_y_values[mask_t] = 20000

                    mask_v = v_y_values > 20000
                    v_y_values[mask_v] = 20000

                    t_loss_values = t_y_values if t_loss_values.size == 0 else t_loss_values + t_y_values
                    v_loss_values = v_y_values if v_loss_values.size == 0 else v_loss_values + v_y_values

            if type == "t" or type == "v":
                ax.plot(epoches, loss_values / len(participant_ids), label=model_name, linewidth=1)
            elif type == "a":
                # ax1 = plt.subplot(121)
                ax[0].plot(epoches, t_loss_values / len(participant_ids), label=model_name, linewidth=1)
                # ax2 = plt.subplot(122)
                ax[1].plot(epoches, v_loss_values / len(participant_ids), label=model_name, linewidth=1)

        if type == "t" or type == "v":
            label_name = "The Average of Training Loss" if type == "t" else "The Average of Validation Loss"
            ax.set_xlabel("Epoches")
            ax.set_ylabel(label_name)
            ax.set_title(label_name + " for Different Models.")
            ax.legend()
        elif type == "a":
            ax[0].set_xlabel("Epoches")
            ax[0].set_ylabel("The Average of Training MSE Loss")
            ax[0].set_title("Average Training MSE Loss for Different Models.")
            # ax[0].text(1, 5.4, 'A', va='bottom', ha='right')
            ax[0].legend()
            ax[1].set_xlabel("Epoches")
            ax[1].set_ylabel("The Average of Validation MSE Loss")
            ax[1].set_title("Average Validation MSE Loss for Different Models.")
            # ax[0].text(1, 9.7, 'B', va='bottom', ha='right')
            ax[1].legend()
        plt.show()


# model = RadarLSTM(n_features=118)
# model_path = "LSTM/lstm_best_t_model_20240328-00:18_0.0_.tar"

# model = RadarTpaLSTM(n_features=118)
# model_path = "TPALSTM/tpa-lstm_best_t_model_20240326-22:08.tar"

# model = RadarGRU(n_features=118)
# model_path = "GRU/gru_best_t_model_20240401-16:37_0.03_.tar"

# EvaModel.plot_mase_of_all_models("all")
# EvaModel.plot_loss_of_all_models("v", 30)
# EvaModel.plot_mean_loss_of_all_models("a")

# model_name = ModelNames.LSTM.value
# EvaModel.plot_model_preds(model_name, 5)
# EvaModel.preds_of_mase(model_name, 1)


# model_name_all = list(ModelNames)
# MSE_str = None
# RMSE_str = None
# MAE_str = None
# R2_str = None
# for i in range(len(model_name_all)):
#     model_name = model_name_all[i].value
#     MSEs, MAEs, RMSEs, R2s = EvaModel.preds_all_mases(model_name)
#     if MSE_str is None:
#         MSE_str = "& MSE & " + str(MSEs[0])
#         RMSE_str = "& RMSE & " + str(RMSEs[0])
#         MAE_str = "& MAE & " + str(MAEs[0])
#         R2_str = "& $R^2$ & " + str(R2s[0])
#     else:
#         MSE_str = MSE_str + " & " + str(MSEs[0])
#         RMSE_str = RMSE_str + " & " + str(RMSEs[0])
#         MAE_str = MAE_str + " & " + str(MAEs[0])
#         R2_str = R2_str + " & " + str(R2s[0])
#
# MSE_str = MSE_str + " \\\\"
# RMSE_str = RMSE_str + " \\\\"
# MAE_str = MAE_str + " \\\\"
# R2_str = R2_str + " \\\\"
# print(f"MSE_str = {MSE_str}")
# print(f"RMSE_str = {RMSE_str}")
# print(f"MAE_str = {MAE_str}")
# print(f"R2_str = {R2_str}")
# print("")
# print(MSE_str + "\n" + RMSE_str + "\n" + MAE_str + "\n" + R2_str)

# EvaModel.plot_average_metric_for_each_model()

model_name = ModelNames.BiLSTM.value
EvaModel.plot_model_preds_for_participant_on_traval(model_name, 12)
