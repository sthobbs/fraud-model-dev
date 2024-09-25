
# This script uploads score event data, generated from both
# training (Python/BigQuery) and serving (Python/MongoDB),
# to GCS and BigQuery.

from gcp_helpers.bigquery import BigQuery
from gcp_helpers.storage import Storage
from gcp_helpers.logger import Logger
from config import project_id, dataset_id, bucket_name, model_exp_dir, scored_data_dir
from utils.parallel import parallelize_threads

# setup logger
logger = Logger(project_id).logger


def load_to_BQ(local_path, gcs_path, table_id, schema_path):
    """
    Upload data to GCS, then load into BQ

    Parameters
    ----------
    local_path : str
        Path to local file
    GCS_path : str
        Path to GCS file to upload to
    table_id : str
        Name of table to load data into
    schema_path : str
        Path to schema file
    """

    # upload data to GCS
    s = Storage(project_id=project_id, bucket_name=bucket_name, logger=logger)
    s.upload_file(local_path, gcs_path)

    # instantiate BigQuery API helper
    t = BigQuery(project_id=project_id,
                 dataset_id=dataset_id,
                 table_id=table_id,
                 schema_json_path=schema_path,
                 logger=logger)

    # move data from GCS to BQ
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    source_format = local_path.split('.')[-1].upper()
    t.load_from_gcs(gcs_uri, source_format=source_format)


def run():
   
    # get params for parallelization
    params = [
        { # Training
            'local_path': f"{model_exp_dir}/scores/test_scores.csv",
            'gcs_path': 'BQ_scores_raw_test_data.csv',
            'table_id': 'bq_scores_raw',
            'schema_path': './validate_serving/schemas/bq_scores_raw.json'
        },
        { # Serving
            'local_path': f"{scored_data_dir}/serving_scores.json",
            'gcs_path': 'serving_scores_raw.json',
            'table_id': 'serving_scores_raw',
            'schema_path': './validate_serving/schemas/serving_scores_raw.json'
        }
    ]

    # Upload to GCS and BQ in parallel
    parallelize_threads(load_to_BQ, params)

if __name__ == '__main__':
    run()
