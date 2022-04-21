from typing import Dict

import mlflow
from loguru import logger
from tensorflow import keras

from data_preprocessing.data_pipeline import DataPipeline
from models.ml_model import Model
from utils import load_dataset


def preprocessing_step(raw_data_path, check_new_data, prepared_data_path):
    logger.info("Preprocessing step")
    raw_data_path = raw_data_path if check_new_data else f"{raw_data_path}/data.csv"
    data_pipeline = DataPipeline(raw_data_path, prepared_data_path)
    data_pipeline.prepare_dataset()


def log_model(parameters: Dict, metrics: Dict, model: keras.Model) -> None:
    """
    Logs models metrics and parameters to MLFlow.
    Args:
        parameters: Parameters for logging.
        metrics: Metrics for logging.
        model: Model to be saved before converting for tensorflow serving.
    """
    with mlflow.start_run(run_name='model subclassing'):
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.keras.log_model(model, "my_model")


def train_step(experiment_name, cfg):
    logger.info("Training step")
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)
    model = Model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    x, y = load_dataset(cfg.get('common', 'prepared_data_path'))
    hist = model.fit(x,
                     y,
                     batch_size=int(cfg.get('train', 'batch_size')),
                     epochs=int(cfg.get('train', 'epochs')))

    log_model(parameters={"epochs": cfg.get('train', 'epochs'), "batch_size": cfg.get('train', 'batch_size')},
              metrics={'accuracy': hist.history['accuracy'][-1], 'loss': hist.history['loss'][-1]},
              model=model)

    model.save(cfg.get('train', 'model_path'))
