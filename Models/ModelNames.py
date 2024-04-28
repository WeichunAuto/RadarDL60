from enum import Enum


class ModelNames(Enum):
    TPALSTM = "TPALSTM"
    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    GRU = "GRU"
    DILATE = "DILATE"
    NBEATS = "NBEATS"
    # CnnLSTM = "CnnLSTM"

# all_v = list(ModelNames)
# print(all_v[0].value)
