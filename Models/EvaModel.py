import numpy as np
import torch
from pathlib import Path
from Models.PrepareTrainData import PrepareTrainData
from Models.LSTM.RadarLSTM import RadarLSTM

import matplotlib.pyplot as plt


class EvaModel:

    @staticmethod
    def eva_lstm_preds(model_path, isEval=False, is_complex=False):
        trained_model = RadarLSTM(n_features=200)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        trained_model.load_state_dict(state_dict)

        if torch.cuda.is_available():
            trained_model = trained_model.cuda()

        _, test_loader = PrepareTrainData(is_shuffle=False).load_data(isEval=isEval, is_complex=is_complex)
        y_real = None
        y_preds = None

        for index, batch in enumerate(test_loader):
            X_batch, y_batch = batch[0], batch[1]

            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

            with torch.inference_mode():  # 关闭 gradient
                preds_batch = trained_model(X_batch)

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

        y_real_max = np.max(y_real)
        y_real_min = np.min(y_real)
        y_real_mean = np.mean(y_real)
        print(f"y_real_max = {y_real_max}, y_real_min = {y_real_min}, y_real_mean = {y_real_mean}")

        y_preds_max = np.max(y_preds)
        y_preds_min = np.min(y_preds)
        y_preds_mean = np.mean(y_preds)
        print(f"y_preds_max = {y_preds_max}, y_preds_min = {y_preds_min}, y_preds_mean = {y_preds_mean}")

        MSE = np.mean((y_preds - y_real) ** 2)
        RMSE = np.sqrt(MSE)
        MAE = np.mean(np.abs(y_preds - y_real))
        print(f"MSE = {MSE}, RMSE = {RMSE}, MAE = {MAE}")

        plt.plot(y_real, color='green', label='Hr Reference')
        plt.plot(y_preds, color='orange', label='Hr Prediction', alpha=0.8)
        plt.xlabel("Numbers")
        plt.ylabel("HR")
        plt.legend()
        # plt.savefig(model_path + '.png')
        plt.show()


# print(Path.cwd().parent)

# EvaModel.eva_lstm_preds("LSTM/lstm_best_t_model_20240315-21:06.tar", isEval=True, is_complex=True, is_normalize=True)
