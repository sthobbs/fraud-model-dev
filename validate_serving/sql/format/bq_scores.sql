
/*
Generate table for test data features and scores from the BigQuery and Python processes

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
*/


SELECT
    S.score,
    F.*,
FROM `{project_id}.{dataset_id}.bq_scores_raw` S
INNER JOIN `{project_id}.{dataset_id}.features_test` F
ON S.uniqueId = F.uniqueId
