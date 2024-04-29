from enum import Enum


class ModelNames(Enum):

    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    TPALSTM = "TPALSTM"
    GRU = "GRU"
    DILATE = "DILATE"
    NBEATS = "NBEATS"
    # HARHN = "HARHN"
    # CnnLSTM = "CnnLSTM"


class PlotMarker(Enum):
    TPALSTM = "o"
    LSTM = "s"
    BiLSTM = "*"
    GRU = "h"
    DILATE = "d"
    NBEATS = "p"

# all_v = list(PlotMarker)
# s = ModelNames.NBEATS.value
# print(PlotMarker[s].value)