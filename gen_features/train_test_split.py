
# Split BigQuery data into train/validation/test,
# then move to GCS and local disk


from pathlib import Path

from tqdm import tqdm
from gcp_helpers.storage import Storage
from gcp_helpers.bigquery import BigQuery
import utils.run_queries
from config import project_id, dataset_id, bucket_name


def run():
    
    # Split feature data into out-of-time train/validation/test datasets
    utils.run_queries.run("./gen_features/sql/train_test_split")

    # Move data from BigQuery to GCS
    for name in ['train', 'validation', 'test']:
        t = BigQuery(project_id=project_id, dataset_id=dataset_id, table_id=f"features_{name}")
        gcs_uri = f"gs://{bucket_name}/feature_data/{name}_*.csv"
        t.extract_to_gcs(gcs_uri)

    # Move data from GCS to local disk
    s = Storage(project_id=project_id, bucket_name=bucket_name)
    blobs = s.list_blobs(prefix="feature_data/")
    for blob in tqdm(blobs):
        file_name = Path(blob).name
        local_dest_path = f"./data/features/{file_name}"
        s.download_blob(blob, local_dest_path)



