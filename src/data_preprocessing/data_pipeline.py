import os.path

import pandas as pd


class DataPipeline:
    """A class for preprocessing data and for data versioning using lakeFS."""

    def __init__(self, raw_data_path: str, data_path: str):
        """
        Creates an instance with desired image shapes and path to the dictionary containing dataset.
        Args:
            raw_data_path: Path to raw data ready to be prepared for training
            data_path: Path to prepared data
        """
        self.__raw_data_path = raw_data_path
        self.__data_path = data_path

    def prepare_dataset(self) -> None:
        """Prepares data for training."""
        data = pd.read_csv(self.__raw_data_path)
        normalized_df = (data - data.mean()) / data.std()
        if not os.path.exists(self.__data_path):
            os.mkdir(self.__data_path)
        normalized_df.to_csv(f"{self.__data_path}/data.csv", index=False)
