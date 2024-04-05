from torch import nn
from torch import Tensor
import torch


class CnnLSTM(nn.Module):

    def __init__(self, n_features=118, n_hidden=128, n_layers=3, dropout=0.):
        super(CnnLSTM, self).__init__()

        self.n_layers = n_layers
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.conv = nn.Conv1d(in_channels=self.n_features, out_channels=self.n_features,
                              kernel_size=(3,), padding=2,
                              padding_mode='replicate')
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=self.dropout)
        self.batchnorm = nn.BatchNorm1d(n_hidden)
        # self.fc = nn.Linear(n_hidden, out_features=num_classes)
        self.fc = nn.Linear(n_hidden, out_features=1)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)
        conv_out = self.conv(x)
        maxpool_out = self.maxpool(conv_out)
        maxpool_out = maxpool_out.permute(0,2,1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, requires_grad=True)
        c_0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, requires_grad=True)

        output_out, (h_out, c_out) = self.lstm(maxpool_out, (h_0, c_0))

        out = output_out[:, -1, :]
        batchnorm_output = self.batchnorm(out)
        linear_output = self.fc(batchnorm_output)
        # sigmoid_output = self.sig(linear_output)

        return linear_output