#standardSQL

/*
Generate feature that only use the transactions table.

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {start_date} = Start date for transaction records
    {end_date} = End date for transaction records
*/


SELECT
    
    -- Non-feature fields
    CURRENT_DATE() AS generatedDate,
    IFNULL(T.fraudLabel, 0) AS fraudLabel,
    T.uniqueId,
    T.customerId,
    T.sessionId,
    T.timestamp,
    T.date,
    T.action,
    
    -- Location    
    IFNULL(T.longitude, 0.0) AS longitude,
    IFNULL(T.latitude, 0.0) AS latitude,

    -- Amount features
    IFNULL(T.amount, -1.0) AS amount,
    IFNULL(MOD(CAST(100 * T.amount AS INT64), 100 * 1   ) / 100, -1.0) AS amountMod1,
    IFNULL(MOD(CAST(100 * T.amount AS INT64), 100 * 100 ) / 100, -1.0) AS amountMod100,
    IFNULL(MOD(CAST(100 * T.amount AS INT64), 100 * 250 ) / 100, -1.0) AS amountMod250,
    IFNULL(MOD(CAST(100 * T.amount AS INT64), 100 * 500 ) / 100, -1.0) AS amountMod500,
    IFNULL(MOD(CAST(100 * T.amount AS INT64), 100 * 1000) / 100, -1.0) AS amountMod1000,

    -- Transaction time features
    IFNULL(EXTRACT(HOUR      FROM T.timestamp), -1) AS hour,
    IFNULL(EXTRACT(DAYOFWEEK FROM T.timestamp), -1) AS dayOfWeek,
    IFNULL(EXTRACT(DAY       FROM T.timestamp), -1) AS dayOfMonth,

    -- Account type features
    CASE WHEN T.accountType = 'checking'    THEN 1 ELSE 0 END AS accountTypeChecking,
    CASE WHEN T.accountType = 'savings'     THEN 1 ELSE 0 END AS accountTypeSavings,
    CASE WHEN T.accountType = 'credit_card' THEN 1 ELSE 0 END AS accountTypeCreditCard,

    -- Count of each type of action
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'transaction') AS transactionCount,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_0') AS action0Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_1') AS action1Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_2') AS action2Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_3') AS action3Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_4') AS action4Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_5') AS action5Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_6') AS action6Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_7') AS action7Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_8') AS action8Count,
    (SELECT COUNT(1) FROM UNNEST(T.session) S WHERE S.action = 'action_9') AS action9Count,
    ARRAY_LENGTH(T.session) AS actionCount,

    -- Total duration and average duration per action
    DATETIME_DIFF(T.timestamp, T.session[OFFSET(0)].timestamp, SECOND) AS secondsToTransaction,
    (SELECT DATETIME_DIFF(T.timestamp, T.session[OFFSET(0)].timestamp, SECOND))
        / ARRAY_LENGTH(T.session) AS avgActionDuration,

    -- Sum/avg/min/max amounts for transactions in session
    (SELECT SUM(S.amount) FROM UNNEST(T.session) S WHERE S.action = 'transaction') AS amountSum,
    (SELECT AVG(S.amount) FROM UNNEST(T.session) S WHERE S.action = 'transaction') AS amountAvg,
    (SELECT MIN(S.amount) FROM UNNEST(T.session) S WHERE S.action = 'transaction') AS amountMin,
    (SELECT MAX(S.amount) FROM UNNEST(T.session) S WHERE S.action = 'transaction') AS amountMax,

    -- Count transactions to the current recipient in session
    (SELECT COUNT(1) FROM UNNEST(T.session) S
        WHERE S.action = 'transaction'
        AND S.recipient = T.recipient) AS recipientTransactionCount,

    -- Number of distinct recipients
    (SELECT COUNT(DISTINCT S.recipient) FROM UNNEST(T.session) S
        WHERE S.action = 'transaction') AS distinctRecipientCount,

    -- Number of repeated recipients (# txns - # distinct recipients)
    (SELECT COUNT(1) - COUNT(DISTINCT S.recipient) FROM UNNEST(T.session) S
        WHERE S.action = 'transaction') AS repeatedRecipientCount,

FROM `{project_id}.{dataset_id}.transactions` T
WHERE T.date >= '{start_date}'
    AND T.date < '{end_date}'
