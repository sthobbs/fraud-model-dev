
/*
Generate transaction records with session aggregated into an array

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {start_date} = Start date for transaction records
    {end_date} = End date for transaction records
*/


WITH

-- Remove Duplicates (although there shouldn't be any on this dataset)
events_deduped AS (
    SELECT *
    FROM `{project_id}.{dataset_id}.events`
    WHERE timestamp >= '{start_date}'
        AND timestamp < '{end_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY uniqueId) = 1
)

-- Generate transaction records
SELECT
    fraudLabel,
    uniqueId,
    customerId,
    sessionId,
    timestamp,
    CAST(timestamp AS DATE) AS date,
    action,
    FIRST_VALUE(longitude) OVER W AS longitude,
    FIRST_VALUE(latitude) OVER W AS latitude,
    amount,
    accountType,
    recipient,

    -- Aggregate session events into an array
    ARRAY_AGG(
        STRUCT(
            uniqueId,
            timestamp,
            action,
            longitude,
            latitude,
            amount,
            accountType,
            recipient
        )
    ) OVER W AS session
FROM events_deduped e
QUALIFY action = 'transaction'
WINDOW W AS (
    PARTITION BY sessionId
    ORDER BY timestamp
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
