
# from gcp_helpers.storage import Storage
from gcp_helpers.storage import Storage
from gcp_helpers.bigquery import BigQuery
from gcp_helpers.logger import Logger
from config import raw_data_output_dir, project_id, dataset_id, bucket_name
from pathlib import Path
from gen_features.utils import parallelize_threads

# setup logger
logger = Logger(project_id).logger


def load_to_BQ(table_id):
    """
    Upload data to GCS, then load into BQ

    Parameters
    ----------
    table_id : str
        Name of table to load data into
    """

    # upload data to GCS
    s = Storage(project_id=project_id, bucket_name=bucket_name, logger=logger)
    local_src_path = Path(raw_data_output_dir) / f"{table_id}.json"
    gcs_dest_path = f'raw_data/{table_id}.json'
    s.upload_file(local_src_path, gcs_dest_path)

    # instantiate BigQuery API helper
    t = BigQuery(project_id=project_id,
                 dataset_id=dataset_id,
                 table_id=table_id,
                 schema_json_path=f'./gen_features/schemas/{table_id}.json',
                 logger=logger)

    # create dataset if it doesn't exist
    if not t.dataset_exists():
        t.create_dataset()

    # move data from GCS to BQ
    gcs_uri = f"gs://{bucket_name}/{gcs_dest_path}"
    t.load_from_gcs(gcs_uri, source_format='JSON')


def run():

    # Upload data to GCS, then load into BQ, in parallel
    params = [
        {'table_id': 'events'},
        {'table_id': 'customer_info'},
    ]
    parallelize_threads(load_to_BQ, params)


if __name__ == '__main__':
    run()
