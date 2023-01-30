
from gcp_helpers.bigquery import BigQuery
from gcp_helpers.logger import Logger
from config import project_id, dataset_id, query_params
from pathlib import Path
from utils.parallel import parallelize_threads
from glob import glob

# setup logger
logger = Logger(project_id).logger


def run_query(query_path):
    """
    Run a query and output results to a BigQuery table

    Parameters
    ----------
    query_path : str
        Path to the query file
    """

    # get destination table name from query file name
    query_path = Path(query_path)
    dest_table_id = query_path.name.replace(".sql", "")

    # get query from file and pass in query parameters
    with open(query_path, 'r') as file:
        query = file.read().format(**query_params)

    # instantiate BigQuery API helper
    t = BigQuery(project_id=project_id, dataset_id=dataset_id, logger=logger)
    
    # run queries
    t.query(query, dest_table_id=dest_table_id)


def run(query_dir):
    """
    Run all queries in a directory in parallel

    Parameters
    ----------
    query_dir : str
        Path to the directory containing the queries
    """

    # get query file paths
    paths = [{'query_path': p} for p in glob(f"{query_dir}/*.sql")]

    # run queries in parallel
    parallelize_threads(run_query, paths)
