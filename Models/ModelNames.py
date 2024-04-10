from enum import Enum


class ModelNames(Enum):
    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    GRU = "GRU"
    DILATE = "DILATE"
    TPALSTM = "TPALSTM"
    HARHN = "HARHN"
    CnnLSTM = "CnnLSTM"


# print(ModelNames.LSTM.value)
