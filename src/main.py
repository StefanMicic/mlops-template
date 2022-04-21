import argparse
import time
from configparser import ConfigParser

import schedule
from loguru import logger

from steps import preprocessing_step, train_step
from utils import check_if_there_is_new_data


def pipeline():
    parser = argparse.ArgumentParser(description="MLOps")
    parser.add_argument("--preprocessing", type=str, default="yes")
    parser.add_argument("--training", type=str, default="yes")
    parser.add_argument("--check_new_data", type=str, default="no")
    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read('config.ini')
    raw_data_path = cfg.get('preprocessing', 'raw_data_path')
    if args.check_new_data == "yes":
        logger.info("Checking for new data...")
        new_data = check_if_there_is_new_data(cfg.get('preprocessing', 'raw_data_path'))
        if new_data is not None:
            logger.info("New data found!")
            raw_data_path = new_data
        else:
            logger.info("No new data!")
            return
    if args.preprocessing == "yes":
        preprocessing_step(raw_data_path, args.check_new_data == "yes", cfg.get('common', 'prepared_data_path'))
    if args.training == "yes":
        train_step(cfg.get('train', 'experiment_name'), cfg)


if __name__ == "__main__":
    schedule.every(1).days.do(pipeline)
    while True:
        schedule.run_pending()
        time.sleep(1)
