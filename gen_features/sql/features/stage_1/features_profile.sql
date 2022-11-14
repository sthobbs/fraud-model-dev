
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
    P.amountMin AS profileRawAmountMin,
    P.amountMax AS profileRawAmountMax,
    P.amountAvg AS profileRawAmountAvg,
    P.amountStd AS profileRawAmountStd,
    P.amountPercentile10 AS profileRawAmountPercentile10,
    P.amountPercentile25 AS profileRawAmountPercentile25,
    P.amountPercentile50 AS profileRawAmountPercentile50,
    P.amountPercentile75 AS profileRawAmountPercentile75,
    P.amountPercentile90 AS profileRawAmountPercentile90,
    SAFE_DIVIDE(T.amount - P.amountAvg, P.amountStd) AS profileAmountZScore,

    -- Time between start of session and first transaction
    P.meanSecondsToTransaction AS profileRawMeanSecondsToTransaction,
    P.stdSecondsToTransaction AS profileRawStdSecondsToTransaction,
    SAFE_DIVIDE(
        DATETIME_DIFF(T.timestamp, T.session[OFFSET(0)].timestamp, SECOND) - P.meanSecondsToTransaction,
        P.stdSecondsToTransaction
    ) AS profileSecondsToTransactionZScore,

    -- Number of sessions with transactions
    P.sessionCount AS profileRawSessionCount,

    -- Number of transactions
    P.transactionCount AS profileRawTransactionCount,

    -- Session action count averages
    P.meanSessionActionCount AS profileRawMeanSessionActionCount,
    P.meanSessionAction0Count AS profileRawMeanSessionAction0Count,
    P.meanSessionAction1Count AS profileRawMeanSessionAction1Count,
    P.meanSessionAction2Count AS profileRawMeanSessionAction2Count,
    P.meanSessionAction3Count AS profileRawMeanSessionAction3Count,
    P.meanSessionAction4Count AS profileRawMeanSessionAction4Count,
    P.meanSessionAction5Count AS profileRawMeanSessionAction5Count,
    P.meanSessionAction6Count AS profileRawMeanSessionAction6Count,
    P.meanSessionAction7Count AS profileRawMeanSessionAction7Count,
    P.meanSessionAction8Count AS profileRawMeanSessionAction8Count,
    P.meanSessionAction9Count AS profileRawMeanSessionAction9Count,

    -- Session action count standard deviations
    P.stdSessionActionCount AS profileRawStdSessionActionCount,
    P.stdSessionAction0Count AS profileRawStdSessionAction0Count,
    P.stdSessionAction1Count AS profileRawStdSessionAction1Count,
    P.stdSessionAction2Count AS profileRawStdSessionAction2Count,
    P.stdSessionAction3Count AS profileRawStdSessionAction3Count,
    P.stdSessionAction4Count AS profileRawStdSessionAction4Count,
    P.stdSessionAction5Count AS profileRawStdSessionAction5Count,
    P.stdSessionAction6Count AS profileRawStdSessionAction6Count,
    P.stdSessionAction7Count AS profileRawStdSessionAction7Count,
    P.stdSessionAction8Count AS profileRawStdSessionAction8Count,
    P.stdSessionAction9Count AS profileRawStdSessionAction9Count,
    
    -- Session action count z-scores
    SAFE_DIVIDE(
        ARRAY_LENGTH(T.session) - P.meanSessionActionCount,
        P.stdSessionActionCount
    ) AS profileSessionActionCountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action0') FROM UNNEST(T.session) S) - P.meanSessionAction0Count,
        P.stdSessionAction0Count
    ) AS profileSessionAction0CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action1') FROM UNNEST(T.session) S) - P.meanSessionAction1Count,
        P.stdSessionAction1Count
    ) AS profileSessionAction1CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action2') FROM UNNEST(T.session) S) - P.meanSessionAction2Count,
        P.stdSessionAction2Count
    ) AS profileSessionAction2CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action3') FROM UNNEST(T.session) S) - P.meanSessionAction3Count,
        P.stdSessionAction3Count
    ) AS profileSessionAction3CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action4') FROM UNNEST(T.session) S) - P.meanSessionAction4Count,
        P.stdSessionAction4Count
    ) AS profileSessionAction4CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action5') FROM UNNEST(T.session) S) - P.meanSessionAction5Count,
        P.stdSessionAction5Count
    ) AS profileSessionAction5CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action6') FROM UNNEST(T.session) S) - P.meanSessionAction6Count,
        P.stdSessionAction6Count
    ) AS profileSessionAction6CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action7') FROM UNNEST(T.session) S) - P.meanSessionAction7Count,
        P.stdSessionAction7Count
    ) AS profileSessionAction7CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action8') FROM UNNEST(T.session) S) - P.meanSessionAction8Count,
        P.stdSessionAction8Count
    ) AS profileSessionAction8CountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'action9') FROM UNNEST(T.session) S) - P.meanSessionAction9Count,
        P.stdSessionAction9Count
    ) AS profileSessionAction9CountZScore,

    -- Session transaction count averages
    P.meanSessionTransactionCount AS profileRawMeanSessionTransactionCount,
    P.meanSessionTransactionFromCheckingCount AS profileRawMeanSessionTransactionFromCheckingCount,
    P.meanSessionTransactionFromSavingsCount AS profileRawMeanSessionTransactionFromSavingsCount,
    P.meanSessionTransactionFromCreditCardCount AS profileRawMeanSessionTransactionFromCreditCardCount,

    -- Session transaction count standard deviations
    P.stdSessionTransactionCount AS profileRawStdSessionTransactionCount,
    P.stdSessionTransactionFromCheckingCount AS profileRawStdSessionTransactionFromCheckingCount,
    P.stdSessionTransactionFromSavingsCount AS profileRawStdSessionTransactionFromSavingsCount,
    P.stdSessionTransactionFromCreditCardCount AS profileRawStdSessionTransactionFromCreditCardCount,

    -- Session transaction count z-scores
    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionCount,
        P.stdSessionTransactionCount
    ) AS profileSessionTransactionCountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'checking')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionFromCheckingCount,
        P.stdSessionTransactionFromCheckingCount
    ) AS profileSessionTransactionFromCheckingCountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'savings')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionFromSavingsCount,
        P.stdSessionTransactionFromSavingsCount
    ) AS profileSessionTransactionFromSavingsCountZScore,

    SAFE_DIVIDE(
        (SELECT COUNTIF(S.action = 'transaction' AND S.accountType = 'credit_card')
            FROM UNNEST(T.session) S) - P.meanSessionTransactionFromCreditCardCount,
        P.stdSessionTransactionFromCreditCardCount
    ) AS profileSessionTransactionFromCreditCardCountZScore,

    -- Number of times they previously sent money to this recipient
    (SELECT COUNT(1) FROM UNNEST(P.recipients) R WHERE R.recipient = T.recipient) AS profileRecipientTxnCount,

    -- Number of distinct recipients they previously sent money to
    (SELECT COUNT(DISTINCT R.recipient) FROM UNNEST(P.recipients) R) AS profileDistinctRecipientCount,

FROM `{project_id}.{dataset_id}.transactions` T
LEFT JOIN txn_profile_dates D
    ON T.uniqueId = D.uniqueId
LEFT JOIN `{project_id}.{dataset_id}.profile` P
    ON T.customerId = P.customerId
    AND P.profileDate = D.profileDate
WHERE T.date >= '{start_date}'
    AND T.date < '{end_date}'
