
# from mock_data import gen_mock_data
import data.raw.gen_mock_data
from config import raw_data_output_dir
from pathlib import Path
import os
from gen_features import run_queries, gen_profile, upload_raw_data, train_test_split
from training.xgb_experiment import XGBExperiment

def run():

    # 1. ----- Generate mock data -----

    # generate raw data if it doesn't already exist
    if not os.path.exists(raw_data_output_dir):
        data.raw.gen_mock_data.run()

    # 2. ----- Generate features -----

    # load data into BigQuery
    upload_raw_data.run()

    # generate transactions table
    run_queries.run("./gen_features/sql/format")

    # generate customer profile table
    gen_profile.run()

    # generate features table
    run_queries.run("./gen_features/sql/features/stage_1")
    run_queries.run("./gen_features/sql/features/stage_2")

    # generate train/validation/test data, and download to local disk
    train_test_split.run()

    # 3. ----- Train model -----
    
    # train & evaluate model
    config_path = "./training/config.yaml"
    exp = XGBExperiment(config_path)
    exp.run()





if __name__ == '__main__':
    run()

