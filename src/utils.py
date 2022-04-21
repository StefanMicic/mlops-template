import datetime
import os
from typing import Optional

import pandas as pd
from numpy import loadtxt


def load_dataset(data_folder: str) -> (pd.DataFrame, pd.DataFrame):
    """Loads dataset prepared for model training.
    Args:
        data_folder: Path to preprocessed data.
    Returns:
        Features and labels for training.
    """
    dataset = loadtxt(f"{data_folder}/data.csv", delimiter=',')
    return dataset[:, 0:8], dataset[:, 8]


def check_if_there_is_new_data(data_folder: str) -> Optional[str]:
    for filename in os.listdir(data_folder):
        m_time = os.path.getmtime(f"{data_folder}/{filename}")
        dt_m = datetime.datetime.fromtimestamp(m_time)
        duration = datetime.datetime.now() - dt_m
        if duration.days == 0:
            return f"{data_folder}/{filename}"
    return None
