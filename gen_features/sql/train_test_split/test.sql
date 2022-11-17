
/*
Generate test set

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {test_start_date} = Start date for test set
    {test_end_date} = End date for for test set
*/


SELECT *
FROM `{project_id}.{dataset_id}.features` T
WHERE T.date >= '{test_start_date}'
    AND T.date < '{test_end_date}'
