
from gcp_helpers.bigquery import BigQuery
from gcp_helpers.logger import Logger
from config import project_id, dataset_id
from utils.parallel import parallelize_threads
import pandas as pd
from pathlib import Path


# setup logger
logger = Logger(project_id).logger


def get_join_rates(output_path):
    """
    get the join rates for the two datasets
    
    Parameters
    ----------
    output_path : str
        path to write the join rates to
    """
    
    t = BigQuery(project_id=project_id, dataset_id=dataset_id, table_id=f"temp")
    
    # detemine join rate between the two sets of scores
    query = f"""SELECT 1 - COUNTIF(score_serving is NULL) / COUNT(score) AS bq_join_rate
                FROM {project_id}.{dataset_id}.bq_serving_scores"""
    bq_join_rate = t.query(query)["bq_join_rate"][0]
    query = f"""SELECT 1 - COUNTIF(score is NULL) / COUNT(score_serving) AS serving_join_rate
                FROM {project_id}.{dataset_id}.bq_serving_scores"""
    serving_join_rate = t.query(query)["serving_join_rate"][0]

    # write to file
    with open(output_path, "w") as f:
        f.write(f"BQ join rate: {bq_join_rate}\n")
        f.write(f"DF join rate: {serving_join_rate}")


def feature_mismatch_rate(feature, tolerance=0.01):
    """
    determine the mismatch rate for a given feature
    
    Parameters
    ----------
    feature : str
        feature to compare
    """
    
    # initialize bigquery client
    t = BigQuery(project_id=project_id,
                 dataset_id=dataset_id,
                 table_id=f"temp_{feature}",
                 logger=logger)
    
    # get mismatch rate dataframe
    query = f"""
    SELECT 
        '{feature}' AS feature,
        COUNTIF(ABS({feature} - {feature}_serving) > {tolerance}) / COUNT(1) AS mismatch_rate
    FROM `{project_id}.{dataset_id}.bq_serving_scores`
    """
    df = t.query(query)
    return df


def all_feature_mismatch_rates(output_path):
    """
    determine the mismatch rate for all features
    
    Parameters
    ----------
    output_path : str
        path to write the mismatch rates to
    """

    # initialize bigquery client
    t = BigQuery(project_id=project_id, dataset_id=dataset_id, table_id=f"temp")
    
    # get list of features
    query = f"SELECT featureNamesStr FROM {project_id}.{dataset_id}.df_scores_raw LIMIT 1"
    features = t.query(query)["featureNamesStr"][0].split(", ")

    # get mismatch rate for each feature in parallel
    params = [{'feature': f} for f in features]
    dfs = parallelize_threads(feature_mismatch_rate, params)
    df = pd.concat(dfs)

    # sort and save to file
    df.sort_values("mismatch_rate", ascending=False, inplace=True)
    df.to_csv(output_path, index=False)


def score_difference_distribution(output_path):
    """
    determine the distribution of score differences

    Parameters
    ----------
    output_path : str
        path to write the distribution to
    """
    
    # initialize bigquery client
    t = BigQuery(project_id=project_id, dataset_id=dataset_id, table_id=f"temp")
    
    # get score difference distribution
    query = f"""
    WITH quantile_array AS (
        SELECT APPROX_QUANTILES(ABS(score - score_serving), 100) AS quantile
        FROM `{dataset_id}.bq_serving_scores`
    )
    SELECT quantile
    FROM quantile_array, UNNEST(quantile) AS quantile
    """
    df = t.query(query)
    df.to_csv(output_path)


def run():

    # make directory for analysis
    analysis_dir = Path("./validate_serving/analysis")
    analysis_dir.mkdir(exist_ok=True)

    # get join rates
    output_path = analysis_dir / "join_rates.txt"
    get_join_rates(output_path)

    # get feature mismatch rates
    output_path = analysis_dir / "feature_mismatch_rates.csv"
    all_feature_mismatch_rates(output_path)

    # get score difference distribution
    output_path = analysis_dir / "score_difference_quantiles.csv"
    score_difference_distribution(output_path)


if __name__ == "__main__":
    run()

