from enum import Enum


class ModelNames(Enum):

    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    GRU = "GRU"
    CnnLSTM = "CNNLSTM"
    DILATE = "RNNED"
    TPALSTM = "TPALSTM"
    NBEATS = "NBEATS"
    # Informer = "Informer"
    Transformer = "Transformer"


class PlotMarker(Enum):
    TPALSTM = "o"
    LSTM = "s"
    BiLSTM = "*"
    GRU = "h"
    RNNED = "d"
    CNNLSTM = "8"
    NBEATS = "p"

# all_v = list(PlotMarker)
# s = ModelNames.NBEATS.value
# print(PlotMarker[s].value)