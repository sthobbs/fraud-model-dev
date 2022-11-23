#standardSQL

/*
Generate feature that use the monthly-updated customer profile table.

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {start_date} = Start date for transaction records
    {end_date} = End date for transaction records

By convention, fields are named with camelCase to match their corresponding
name in the java Apache Beam pipeline. All profile features begin with
"profile", and features that only use the profile (i.e. not current transaction
data) are prefixed with "profileRaw". 
*/


WITH

-- Get valid profile dates
profile_dates AS (
    SELECT DISTINCT profileDate
    FROM `{project_id}.{dataset_id}.profile`
    WHERE profileDate < '{end_date}'
),


-- Get the most recent profile for each transaction
txn_profile_dates AS (
    SELECT
        T.uniqueId,
        P.profileDate,
    FROM `{project_id}.{dataset_id}.transactions` T
    LEFT JOIN profile_dates P
        ON T.date >= P.profileDate
    WHERE T.date >= '{start_date}'
        AND T.date < '{end_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY T.uniqueId ORDER BY profileDate DESC) = 1
)


SELECT
    T.uniqueId,

    -- Indicator of whether or not we have a profle for this customer
    CASE WHEN P.customerId IS NULL THEN 0 ELSE 1 END AS profileRawInd,

    -- Amount features
    IFNULL(P.amountMin, -1.0) AS profileRawAmountMin,
    IFNULL(P.amountMax, -1.0) AS profileRawAmountMax,
    IFNULL(P.amountAvg, -1.0) AS profileRawAmountAvg,
    IFNULL(P.amountStd, -1.0) AS profileRawAmountStd,
    IFNULL(P.amountPercentile10, -1.0) AS profileRawAmountPercentile10,
    IFNULL(P.amountPercentile25, -1.0) AS profileRawAmountPercentile25,
    IFNULL(P.amountPercentile50, -1.0) AS profileRawAmountPercentile50,
    IFNULL(P.amountPercentile75, -1.0) AS profileRawAmountPercentile75,
    IFNULL(P.amountPercentile90, -1.0) AS profileRawAmountPercentile90,
    IFNULL(SAFE_DIVIDE(T.amount - P.amountAvg, P.amountStd), -1.0) AS profileAmountZScore,

    -- Time between start of session and first transaction
    IFNULL(P.meanSecondsToTransaction, -1.0) AS profileRawMeanSecondsToTransaction,
    IFNULL(P.stdSecondsToTransaction, -1.0) AS profileRawStdSecondsToTransaction,
    IFNULL(SAFE_DIVIDE(
        DATETIME_DIFF(T.timestamp, T.session[OFFSET(0)].timestamp, SECOND) - P.meanSecondsToTransaction,
        P.stdSecondsToTransaction
    ), -1.0) AS profileSecondsToTransactionZScore,

    -- Number of sessions with transactions
    IFNULL(P.sessionCount, -1) AS profileRawSessionCount,

    -- Number of transactions
    IFNULL(P.transactionCount, -1) AS profileRawTransactionCount,

    -- Session action count averages
    IFNULL(P.meanSessionActionCount, -1.0) AS profileRawMeanSessionActionCount,
    IFNULL(P.meanSessionAction0Count, -1.0) AS profileRawMeanSessionAction0Count,
    IFNULL(P.meanSessionAction1Count, -1.0) AS profileRawMeanSessionAction1Count,
    IFNULL(P.meanSessionAction2Count, -1.0) AS profileRawMeanSessionAction2Count,
    IFNULL(P.meanSessionAction3Count, -1.0) AS profileRawMeanSessionAction3Count,
    IFNULL(P.meanSessionAction4Count, -1.0) AS profileRawMeanSessionAction4Count,
    IFNULL(P.meanSessionAction5Count, -1.0) AS profileRawMeanSessionAction5Count,
    IFNULL(P.meanSessionAction6Count, -1.0) AS profileRawMeanSessionAction6Count,
    IFNULL(P.meanSessionAction7Count, -1.0) AS profileRawMeanSessionAction7Count,
    IFNULL(P.meanSessionAction8Count, -1.0) AS profileRawMeanSessionAction8Count,
    IFNULL(P.meanSessionAction9Count, -1.0) AS profileRawMeanSessionAction9Count,

    -- Session action count standard deviations
    IFNULL(P.stdSessionActionCount, -1.0) AS profileRawStdSessionActionCount,
    IFNULL(P.stdSessionAction0Count, -1.0) AS profileRawStdSessionAction0Count,
    IFNULL(P.stdSessionAction1Count, -1.0) AS profileRawStdSessionAction1Count,
    IFNULL(P.stdSessionAction2Count, -1.0) AS profileRawStdSessionAction2Count,
    IFNULL(P.stdSessionAction3Count, -1.0) AS profileRawStdSessionAction3Count,
    IFNULL(P.stdSessionAction4Count, -1.0) AS profileRawStdSessionAction4Count,
    IFNULL(P.stdSessionAction5Count, -1.0) AS profileRawStdSessionAction5Count,
    IFNULL(P.stdSessionAction6Count, -1.0) AS profileRawStdSessionAction6Count,
    IFNULL(P.stdSessionAction7Count, -1.0) AS profileRawStdSessionAction7Count,
    IFNULL(P.stdSessionAction8Count, -1.0) AS profileRawStdSessionAction8Count,
    IFNULL(P.stdSessionAction9Count, -1.0) AS profileRawStdSessionAction9Count,
    
    -- Session action count z-scores
    IFNULL(SAFE_DIVIDE(
        ARRAY_LENGTH(T.session) - P.meanSessionActionCount,
        P.stdSessionActionCount
    ), -1.0) AS profileSessionActionCountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_0') FROM UNNEST(T.session) S) - P.meanSessionAction0Count,
        P.stdSessionAction0Count
    ), -1.0) AS profileSessionAction0CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_1') FROM UNNEST(T.session) S) - P.meanSessionAction1Count,
        P.stdSessionAction1Count
    ), -1.0) AS profileSessionAction1CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_2') FROM UNNEST(T.session) S) - P.meanSessionAction2Count,
        P.stdSessionAction2Count
    ), -1.0) AS profileSessionAction2CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_3') FROM UNNEST(T.session) S) - P.meanSessionAction3Count,
        P.stdSessionAction3Count
    ), -1.0) AS profileSessionAction3CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_4') FROM UNNEST(T.session) S) - P.meanSessionAction4Count,
        P.stdSessionAction4Count
    ), -1.0) AS profileSessionAction4CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_5') FROM UNNEST(T.session) S) - P.meanSessionAction5Count,
        P.stdSessionAction5Count
    ), -1.0) AS profileSessionAction5CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_6') FROM UNNEST(T.session) S) - P.meanSessionAction6Count,
        P.stdSessionAction6Count
    ), -1.0) AS profileSessionAction6CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_7') FROM UNNEST(T.session) S) - P.meanSessionAction7Count,
        P.stdSessionAction7Count
    ), -1.0) AS profileSessionAction7CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_8') FROM UNNEST(T.session) S) - P.meanSessionAction8Count,
        P.stdSessionAction8Count
    ), -1.0) AS profileSessionAction8CountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action_9') FROM UNNEST(T.session) S) - P.meanSessionAction9Count,
        P.stdSessionAction9Count
    ), -1.0) AS profileSessionAction9CountZScore,

    -- Session transaction count averages
    IFNULL(P.meanSessionTransactionCount, -1.0) AS profileRawMeanSessionTransactionCount,
    IFNULL(P.meanSessionTransactionFromCheckingCount, -1.0) AS profileRawMeanSessionTransactionFromCheckingCount,
    IFNULL(P.meanSessionTransactionFromSavingsCount, -1.0) AS profileRawMeanSessionTransactionFromSavingsCount,
    IFNULL(P.meanSessionTransactionFromCreditCardCount, -1.0) AS profileRawMeanSessionTransactionFromCreditCardCount,

    -- Session transaction count standard deviations
    IFNULL(P.stdSessionTransactionCount, -1.0) AS profileRawStdSessionTransactionCount,
    IFNULL(P.stdSessionTransactionFromCheckingCount, -1.0) AS profileRawStdSessionTransactionFromCheckingCount,
    IFNULL(P.stdSessionTransactionFromSavingsCount, -1.0) AS profileRawStdSessionTransactionFromSavingsCount,
    IFNULL(P.stdSessionTransactionFromCreditCardCount, -1.0) AS profileRawStdSessionTransactionFromCreditCardCount,

    -- Session transaction count z-scores
    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionCount,
        P.stdSessionTransactionCount
    ), -1.0) AS profileSessionTransactionCountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'checking')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionFromCheckingCount,
        P.stdSessionTransactionFromCheckingCount
    ), -1.0) AS profileSessionTransactionFromCheckingCountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'savings')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionFromSavingsCount,
        P.stdSessionTransactionFromSavingsCount
    ), -1.0) AS profileSessionTransactionFromSavingsCountZScore,

    IFNULL(SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'credit_card')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionFromCreditCardCount,
        P.stdSessionTransactionFromCreditCardCount
    ), -1.0) AS profileSessionTransactionFromCreditCardCountZScore,

    -- Number of times they previously sent money to this recipient
    IFNULL((SELECT R.txnCnt FROM UNNEST(P.recipients) R WHERE R.recipient = T.recipient LIMIT 1), 0) AS profileRecipientTxnCount,

    -- Number of distinct recipients they previously sent money to
    IFNULL((SELECT COUNT(DISTINCT R.recipient) FROM UNNEST(P.recipients) R), 0) AS profileDistinctRecipientCount,

FROM `{project_id}.{dataset_id}.transactions` T
LEFT JOIN txn_profile_dates D
    ON T.uniqueId = D.uniqueId
LEFT JOIN `{project_id}.{dataset_id}.profile` P
    ON T.customerId = P.customerId
    AND P.profileDate = D.profileDate
WHERE T.date >= '{start_date}'
    AND T.date < '{end_date}'
