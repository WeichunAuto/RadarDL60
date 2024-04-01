import random

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import sys

sys.path.append('/home/syt0722/Weichun/60pts')
from Models.PrepareTrainData import PrepareTrainData
from Models.LSTM.RadarLSTM import RadarLSTM

import matplotlib.pyplot as plt
from datetime import datetime

seed=42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子


class ExecuteTrain:
    def __init__(self, window_size=5, fs=2000, is_shuffle=False, epochs=500):

        self.window_size = window_size
        self.fs = fs
        self.epochs = epochs

        self.train_loader, self.val_loader = self.initialize_dataloader(is_shuffle=is_shuffle)

        self.lr, self.loss_fun, self.model, self.optimizer = self.initialize_model()

        self.formatted_time = datetime.now().strftime("%Y%m%d-%H:%M")

    def initialize_dataloader(self, is_shuffle=False):
        ptd = PrepareTrainData(is_shuffle=is_shuffle)
        return ptd.train_dataloader(), ptd.val_dataloader()

    def initialize_model(self):
        lr = 0.001
        loss_fun = nn.MSELoss()
        model = RadarLSTM(n_features=118)

        if torch.cuda.is_available():
            model = model.cuda()
            loss_fun = loss_fun.cuda()
        optimizer = optim.ASGD(model.parameters(), lr=lr)

        return lr, loss_fun, model, optimizer

    def start_training(self):
        train_losses = []
        validate_losses = []
        epoch_counter = []
        best_v_loss = float('inf')
        best_t_loss = float('inf')

        best_t_epoch = 0
        best_v_epoch = 0

        for epoch in tqdm(range(self.epochs)):
            t_loss = self.train_per_epoch()
            v_loss = self.validate_per_epoch()

            if v_loss < best_v_loss:
                best_v_loss = v_loss
                best_v_epoch = epoch
                torch.save(self.model.state_dict(), "lstm_best_v_model_" + self.formatted_time + ".tar")  # 保存训练后的模型
            if t_loss < best_t_loss:
                best_t_loss = t_loss
                best_t_epoch = epoch
                torch.save(self.model.state_dict(), "lstm_best_t_model_" + self.formatted_time + ".tar")  # 保存训练后的模型

            if (epoch + 1) % 10 == 0:
                print("t_loss: " + str(t_loss) + ", v_loss: " + str(v_loss))

            train_losses.append(t_loss)
            validate_losses.append(v_loss)
            epoch_counter.append(epoch)
        print(f"best_t_epoch = {best_t_epoch}, best_v_epoch = {best_v_epoch}")
        return train_losses, validate_losses, epoch_counter

    def train_per_epoch(self):
        self.model.train()
        loss_batch_sum = 0.
        for index, batch in enumerate(self.train_loader):
            X_batch, y_batch = batch[0], batch[1]

            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

            preds_batch = self.model(X_batch)  # 2. 预测
            loss = self.loss_fun(preds_batch, y_batch)  # 3. 计算 loss
            self.optimizer.zero_grad()  # 4. 每一次 loop, 都重置 gradient
            loss.backward()  # 5. 反向传播，计算并更新 gradient 为 True 的参数值
            self.optimizer.step()  # 6. 更新 参数值

            loss_batch_sum += loss.item()
            print(f'loss_batch_sum-{index} = {loss_batch_sum}')

        return loss_batch_sum

    def validate_per_epoch(self):
        self.model.eval()
        loss_batch_sum = 0.
        for index, batch in enumerate(self.val_loader):
            X_batch, y_batch = batch[0], batch[1]

            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

            with torch.inference_mode():  # 关闭 gradient tracking
                preds_batch = self.model(X_batch)  # 2. 预测

                loss = self.loss_fun(preds_batch, y_batch)  # 3. 计算 loss
                loss_batch_sum += loss.item()

        return loss_batch_sum

    def visualize_loss(self, train_loss, validation_loss):
        plt.figure(figsize=(10, 3))
        plt.plot(train_loss, color='green', label='train_loss')
        plt.plot(validation_loss, color='blue', label='validation_loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('loss_plot' + self.formatted_time + '.png')
        plt.show()


et = ExecuteTrain()
t_loss, v_loss, _ = et.start_training()
et.visualize_loss(t_loss, v_loss)
# et.evaluate_preds()

# X_data, y_data = PrepareTrainData(is_shuffle=True).load_data(isEval=False)
