
/*
Generate feature that use the customer information table.

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {start_date} = Start date for transaction records
    {end_date} = End date for transaction records
*/


SELECT
    T.uniqueId,
    C.age,
    CASE WHEN C.gender = 'M' THEN 1 ELSE 0 END AS genderMale,
    CASE WHEN C.maritalStatus = 'single' THEN 1 ELSE 0 END AS maritalStatusSingle,
    CASE WHEN C.maritalStatus = 'married' THEN 1 ELSE 0 END AS maritalStatusMarried,
    CASE WHEN C.maritalStatus = 'divorced' THEN 1 ELSE 0 END AS maritalStatusDivorced,
    C.homeLongitude,
    C.homeLatitude,
    SQRT(POW(C.homeLongitude - T.longitude, 2) + POW(C.homeLatitude - T.latitude, 2)) AS distanceFromHome,
FROM `{project_id}.{dataset_id}.transactions` T
LEFT JOIN `{project_id}.{dataset_id}.customer_info` C
ON T.customerId = C.customerId
WHERE T.date >= '{start_date}'
    AND T.date < '{end_date}'
