#standardSQL

/*
Generate table for test data features and scores from both the
the BigQuery/Python and Dataflow processes

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
*/


SELECT
    BQ.*,
    S.* EXCEPT (
        fraudLabel,
        uniqueId,
        customerId,
        sessionId,
        timestamp,
        action,
        amount,
        modelId
    ),
FROM `{project_id}.{dataset_id}.bq_scores` BQ
FULL OUTER JOIN `{project_id}.{dataset_id}.serving_scores` S
ON BQ.uniqueId = S.uniqueId
