from enum import Enum


class ModelNames(Enum):
    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    GRU = "GRU"
    DILATE = "DILATE"
    TPALSTM = "TPALSTM"
    NBEATS = "NBEATS"
    CnnLSTM = "CnnLSTM"

# all_v = list(ModelNames)
# print(all_v[0].value)
