
# from mock_data import gen_mock_data
import mock_data.gen_mock_data
from config import data_output_dir
from pathlib import Path
import os
from gen_features import load_data, run_queries, gen_profile


def run():

    # generate data if it doesn't already exist
    if not os.path.exists(data_output_dir):
        mock_data.gen_mock_data.run()

    # load data into BigQuery
    load_data.run()

    # generate transactions table
    run_queries.run("./gen_features/sql/format")

    # generate customer profile table
    gen_profile.run()

    # generate features
    run_queries.run("./gen_features/sql/features/stage_1")
    run_queries.run("./gen_features/sql/features/stage_2")




if __name__ == '__main__':
    run()

