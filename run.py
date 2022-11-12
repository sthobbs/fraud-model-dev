
# from mock_data import gen_mock_data
import mock_data.gen_mock_data
from mock_data.config import data_output_dir
from pathlib import Path
import os




def run():

    # generate data if it doesn't already exist
    data_file_path = Path(data_output_dir) / "actions.json"
    if not os.path.exists(data_file_path):
        mock_data.gen_mock_data.run()





if __name__ == '__main__':
    run()

