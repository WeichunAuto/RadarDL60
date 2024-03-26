import torch
from torch import nn, optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RadarTpaLSTM(nn.Module):

    def __init__(self, input_size=118, output_horizon=1, num_filters=3, hidden_size=64, obs_len=5, n_layers=3):
        super(RadarTpaLSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bias=True,
                            batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = num_filters
        self.filter_size = 1  # Don't change this - otherwise CNN filters no longer 1D
        self.output_horizon = output_horizon
        self.attention = TemporalPatternAttention(self.filter_size, self.filter_num, obs_len - 1, hidden_size)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            self.relu,
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size // 2, output_horizon)
        )
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers

        # self.lr = lr
        self.criterion = nn.MSELoss()

        # self.save_hyperparameters()

    def forward(self, x):
        batch_size, obs_len, f_dim = x.size()
        xconcat = self.relu(self.hidden(x))

        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(device)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        ct = ht.clone()
        for t in range(obs_len):
            xt = x[:, t, :].view(batch_size, 1, -1)
            # xt = xconcat[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)

        # reshape hidden states H
        H = H.view(-1, 1, obs_len - 1, self.hidden_size)
        new_ht = self.attention(H, htt)
        ypred = self.linear(new_ht).unsqueeze(-1)
        #         ypred = self.mlp_out(new_ht).unsqueeze(-1)

        return ypred



class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()

    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht)  # batch_size, 1, filter_num
        conv_vecs = self.conv(H)

        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)

        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return new_ht
