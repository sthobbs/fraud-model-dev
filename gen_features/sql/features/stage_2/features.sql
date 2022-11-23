#standardSQL

/*
Combine all feature tables into a single table.

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {start_date} = Start date for transaction records
    {end_date} = End date for transaction records
*/


SELECT
    A.*,
    B.* EXCEPT (uniqueId),
    C.* EXCEPT (uniqueId),
FROM `{project_id}.{dataset_id}.features_txn` A
LEFT JOIN `{project_id}.{dataset_id}.features_profile` B
    ON A.uniqueId = B.uniqueId
LEFT JOIN `{project_id}.{dataset_id}.features_cust_info` C
    ON A.uniqueId = C.uniqueId
WHERE A.date >= '{start_date}'
    AND A.date < '{end_date}'
