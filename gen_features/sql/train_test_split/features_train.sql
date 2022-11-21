
/*
Generate training set

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {train_start_date} = Start date for training set
    {train_end_date} = End date for for training set
*/


SELECT *
FROM `{project_id}.{dataset_id}.features` T
WHERE T.date >= '{train_start_date}'
    AND T.date < '{train_end_date}'
