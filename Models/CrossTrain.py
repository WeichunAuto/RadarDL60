import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/syt0722/Weichun/60pts')
from Models.ExecuteTrainModels import ExecuteTrainModels
from Models.ModelNames import ModelNames
from Models.LSTM.RadarLSTM import RadarLSTM
from Models.BiLSTM.RadarBiLSTM import RadarBiLSTM
from Models.GRU.RadarGRU import RadarGRU
from Models.TPALSTM.RadarTpaLSTM import RadarTpaLSTM


def get_model(model_name):
    model = None
    if model_name == ModelNames.LSTM.value:
        model = RadarLSTM()
    elif model_name == ModelNames.BiLSTM.value:
        model = RadarBiLSTM()
    elif model_name == ModelNames.GRU.value:
        model = RadarGRU()
    elif model_name == ModelNames.TPALSTM.value:
        model = RadarTpaLSTM()

    return model


epochs = 2
model_name = ModelNames.TPALSTM.value
my_model = get_model(model_name)

pd_columns = ["t_loss", "v_loss", "p_id"]
pd_values = None
current_dir = str(Path.cwd())
save_path = os.path.join(current_dir, model_name, "trained_models")
saved_file_path = os.path.join(save_path, "loss_" + model_name + ".csv")

# all participants ID
participant_ids = [i for i in range(1, 31) if i != 3]

if not os.path.exists(save_path):
    os.makedirs(save_path)

for participant_id in participant_ids:
    ids = [participant_id for i in range(epochs)]
    et = ExecuteTrainModels(my_model, model_name, participant_id, epochs=epochs)
    t_loss, v_loss, _ = et.start_training()

    if pd_values is None:
        pd_values = np.column_stack((t_loss, v_loss, ids))
    else:
        pd_values = np.vstack([pd_values, np.column_stack((t_loss, v_loss, ids))])
    print(f'participant {participant_id} done.')

df = pd.DataFrame(pd_values, columns=pd_columns)
df.to_csv(saved_file_path)
