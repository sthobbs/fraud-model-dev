#standardSQL

/*
Generate customer profile table

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {start_date} = Start date for transaction records
    {end_date} = End date for transaction records
*/


WITH

-- Customer level aggregations
customer_agg AS (
    SELECT
        T.customerId,
        
        -- Amounts of transactions
        MIN(T.amount) AS amountMin,
        MAX(T.amount) AS amountMax,
        AVG(T.amount) AS amountAvg,
        STDDEV(T.amount) AS amountStd,
        APPROX_QUANTILES(T.amount, 100)[OFFSET(10)] AS amountPercentile10,
        APPROX_QUANTILES(T.amount, 100)[OFFSET(25)] AS amountPercentile25, -- q1
        APPROX_QUANTILES(T.amount, 100)[OFFSET(50)] AS amountPercentile50, -- median
        APPROX_QUANTILES(T.amount, 100)[OFFSET(75)] AS amountPercentile75, -- q3
        APPROX_QUANTILES(T.amount, 100)[OFFSET(90)] AS amountPercentile90,

        -- Time between start of the session and the transaction
        AVG(TIMESTAMP_DIFF(T.timestamp, T.session[OFFSET(0)].timestamp, SECOND)) AS meanSecondsToTransaction,
        STDDEV_SAMP(TIMESTAMP_DIFF(T.timestamp, T.session[OFFSET(0)].timestamp, SECOND)) AS stdSecondsToTransaction,
        
        -- Number of sessions with transactions
        COUNT(DISTINCT T.sessionId) AS sessionCount,
        
        -- Number of transactions
        COUNT(1) AS transactionCount,
        
        -- Session action count averages (before the transaction)
        AVG(ARRAY_LENGTH(T.session)) AS meanSessionActionCount,
        AVG((SELECT COUNTIF(S.action = 'action_0') FROM UNNEST(T.session) S)) AS meanSessionAction0Count,
        AVG((SELECT COUNTIF(S.action = 'action_1') FROM UNNEST(T.session) S)) AS meanSessionAction1Count,
        AVG((SELECT COUNTIF(S.action = 'action_2') FROM UNNEST(T.session) S)) AS meanSessionAction2Count,
        AVG((SELECT COUNTIF(S.action = 'action_3') FROM UNNEST(T.session) S)) AS meanSessionAction3Count,
        AVG((SELECT COUNTIF(S.action = 'action_4') FROM UNNEST(T.session) S)) AS meanSessionAction4Count,
        AVG((SELECT COUNTIF(S.action = 'action_5') FROM UNNEST(T.session) S)) AS meanSessionAction5Count,
        AVG((SELECT COUNTIF(S.action = 'action_6') FROM UNNEST(T.session) S)) AS meanSessionAction6Count,
        AVG((SELECT COUNTIF(S.action = 'action_7') FROM UNNEST(T.session) S)) AS meanSessionAction7Count,
        AVG((SELECT COUNTIF(S.action = 'action_8') FROM UNNEST(T.session) S)) AS meanSessionAction8Count,
        AVG((SELECT COUNTIF(S.action = 'action_9') FROM UNNEST(T.session) S)) AS meanSessionAction9Count,

        -- Session action count standard deviations (before the transaction)
        STDDEV_SAMP(ARRAY_LENGTH(T.session)) AS stdSessionActionCount,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_0') FROM UNNEST(T.session) S)) AS stdSessionAction0Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_1') FROM UNNEST(T.session) S)) AS stdSessionAction1Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_2') FROM UNNEST(T.session) S)) AS stdSessionAction2Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_3') FROM UNNEST(T.session) S)) AS stdSessionAction3Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_4') FROM UNNEST(T.session) S)) AS stdSessionAction4Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_5') FROM UNNEST(T.session) S)) AS stdSessionAction5Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_6') FROM UNNEST(T.session) S)) AS stdSessionAction6Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_7') FROM UNNEST(T.session) S)) AS stdSessionAction7Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_8') FROM UNNEST(T.session) S)) AS stdSessionAction8Count,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'action_9') FROM UNNEST(T.session) S)) AS stdSessionAction9Count,

        -- Session transaction count averages (before and including the current transaction)
        AVG((SELECT COUNTIF(S.action = 'transaction')
            FROM UNNEST(T.session) S)) AS meanSessionTransactionCount,
        AVG((SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'checking')
            FROM UNNEST(T.session) S)) AS meanSessionTransactionFromCheckingCount,
        AVG((SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'savings')
            FROM UNNEST(T.session) S)) AS meanSessionTransactionFromSavingsCount,
        AVG((SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'credit_card')
            FROM UNNEST(T.session) S)) AS meanSessionTransactionFromCreditCardCount,

        -- Session transaction count averages (before and including the current transaction)
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'transaction')
            FROM UNNEST(T.session) S)) AS stdSessionTransactionCount,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'checking')
            FROM UNNEST(T.session) S)) AS stdSessionTransactionFromCheckingCount,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'savings')
            FROM UNNEST(T.session) S)) AS stdSessionTransactionFromSavingsCount,
        STDDEV_SAMP((SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'credit_card')
            FROM UNNEST(T.session) S)) AS stdSessionTransactionFromCreditCardCount,

    FROM `{project_id}.{dataset_id}.transactions` T
    WHERE T.date >= '{start_date}'
        AND T.date < '{end_date}'
    GROUP BY T.customerId
),


-- Customer, recipient level aggregations
recipient_pre_agg AS (
    SELECT
        customerId,
        recipient,
        COUNT(1) AS txnCnt,
        MIN(timestamp) AS minTimestamp,
    FROM `{project_id}.{dataset_id}.transactions`
    WHERE date >= '{start_date}'
        AND date < '{end_date}'
    GROUP BY customerId, recipient
),

-- Recipient arrays
recipient_agg AS (
    SELECT 
        customerId,
        ARRAY_AGG(
            STRUCT(
                recipient,
                txnCnt,
                minTimestamp
            )
        ) AS recipients
    FROM recipient_pre_agg
    GROUP BY customerId
)


SELECT
    CAST('{end_date}' AS DATE) AS profileDate,
    A.*,
    B.recipients
FROM customer_agg A
LEFT JOIN recipient_agg B
    ON A.customerId = B.customerId
