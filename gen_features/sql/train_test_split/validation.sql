
/*
Generate validation set

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {valid_start_date} = Start date for validation set
    {valid_end_date} = End date for for validation set
*/


SELECT *
FROM `{project_id}.{dataset_id}.features` T
WHERE T.date >= '{valid_start_date}'
    AND T.date < '{valid_end_date}'
