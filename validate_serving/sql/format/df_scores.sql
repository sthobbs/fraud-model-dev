#standardSQL

/*
Generate table for test data features and scores from the Dataflow pipeline processes

Query parameters:
    {project_id} = BigQuery project ID
    {dataset_id} = BigQuery dataset ID
    {test_start_date} = Start date for test set
    {test_end_date} = End date for for test set
*/


WITH

-- Remove duplicates and filter date range
dedupe AS (
    SELECT
        fraudLabel,
        uniqueId,
        customerId,
        sessionId,
        timestamp,
        scoreTimestamp,
        action,
        amount,
        modelId,
        score AS score_df,
        SPLIT(featureValuesStr, ', ') AS featureValues,
    FROM `{project_id}.{dataset_id}.df_scores_raw`
    -- Filter on test data to match BQ data we're comparing to
    WHERE timestamp >= '{test_start_date}'
        AND timestamp < '{test_end_date}' 
    -- Remove duplicates (since transactions are scored repeatedly as new data comes in)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY uniqueId ORDER BY scoreTimestamp DESC) = 1
)

SELECT
    * EXCEPT (featureValues),
    -- Parse feature values into separate columns
    CAST(featureValues[ORDINAL(1)] AS FLOAT64) AS longitude_df,
    CAST(featureValues[ORDINAL(2)] AS FLOAT64) AS latitude_df,
    CAST(featureValues[ORDINAL(3)] AS FLOAT64) AS amount_df,
    CAST(featureValues[ORDINAL(4)] AS FLOAT64) AS amountMod1_df,
    CAST(featureValues[ORDINAL(5)] AS FLOAT64) AS amountMod100_df,
    CAST(featureValues[ORDINAL(6)] AS FLOAT64) AS amountMod250_df,
    CAST(featureValues[ORDINAL(7)] AS FLOAT64) AS amountMod500_df,
    CAST(featureValues[ORDINAL(8)] AS FLOAT64) AS amountMod1000_df,
    CAST(featureValues[ORDINAL(9)] AS FLOAT64) AS hour_df,
    CAST(featureValues[ORDINAL(10)] AS FLOAT64) AS dayOfWeek_df,
    CAST(featureValues[ORDINAL(11)] AS FLOAT64) AS dayOfMonth_df,
    CAST(featureValues[ORDINAL(12)] AS FLOAT64) AS accountTypeChecking_df,
    CAST(featureValues[ORDINAL(13)] AS FLOAT64) AS accountTypeSavings_df,
    CAST(featureValues[ORDINAL(14)] AS FLOAT64) AS accountTypeCreditCard_df,
    CAST(featureValues[ORDINAL(15)] AS FLOAT64) AS transactionCount_df,
    CAST(featureValues[ORDINAL(16)] AS FLOAT64) AS action0Count_df,
    CAST(featureValues[ORDINAL(17)] AS FLOAT64) AS action1Count_df,
    CAST(featureValues[ORDINAL(18)] AS FLOAT64) AS action2Count_df,
    CAST(featureValues[ORDINAL(19)] AS FLOAT64) AS action3Count_df,
    CAST(featureValues[ORDINAL(20)] AS FLOAT64) AS action4Count_df,
    CAST(featureValues[ORDINAL(21)] AS FLOAT64) AS action5Count_df,
    CAST(featureValues[ORDINAL(22)] AS FLOAT64) AS action6Count_df,
    CAST(featureValues[ORDINAL(23)] AS FLOAT64) AS action7Count_df,
    CAST(featureValues[ORDINAL(24)] AS FLOAT64) AS action8Count_df,
    CAST(featureValues[ORDINAL(25)] AS FLOAT64) AS action9Count_df,
    CAST(featureValues[ORDINAL(26)] AS FLOAT64) AS actionCount_df,
    CAST(featureValues[ORDINAL(27)] AS FLOAT64) AS secondsToTransaction_df,
    CAST(featureValues[ORDINAL(28)] AS FLOAT64) AS avgActionDuration_df,
    CAST(featureValues[ORDINAL(29)] AS FLOAT64) AS amountSum_df,
    CAST(featureValues[ORDINAL(30)] AS FLOAT64) AS amountAvg_df,
    CAST(featureValues[ORDINAL(31)] AS FLOAT64) AS amountMin_df,
    CAST(featureValues[ORDINAL(32)] AS FLOAT64) AS amountMax_df,
    CAST(featureValues[ORDINAL(33)] AS FLOAT64) AS recipientTransactionCount_df,
    CAST(featureValues[ORDINAL(34)] AS FLOAT64) AS distinctRecipientCount_df,
    CAST(featureValues[ORDINAL(35)] AS FLOAT64) AS repeatedRecipientCount_df,
    CAST(featureValues[ORDINAL(36)] AS FLOAT64) AS profileRawInd_df,
    CAST(featureValues[ORDINAL(37)] AS FLOAT64) AS profileRawAmountMin_df,
    CAST(featureValues[ORDINAL(38)] AS FLOAT64) AS profileRawAmountMax_df,
    CAST(featureValues[ORDINAL(39)] AS FLOAT64) AS profileRawAmountAvg_df,
    CAST(featureValues[ORDINAL(40)] AS FLOAT64) AS profileRawAmountStd_df,
    CAST(featureValues[ORDINAL(41)] AS FLOAT64) AS profileRawAmountPercentile10_df,
    CAST(featureValues[ORDINAL(42)] AS FLOAT64) AS profileRawAmountPercentile25_df,
    CAST(featureValues[ORDINAL(43)] AS FLOAT64) AS profileRawAmountPercentile50_df,
    CAST(featureValues[ORDINAL(44)] AS FLOAT64) AS profileRawAmountPercentile75_df,
    CAST(featureValues[ORDINAL(45)] AS FLOAT64) AS profileRawAmountPercentile90_df,
    CAST(featureValues[ORDINAL(46)] AS FLOAT64) AS profileAmountZScore_df,
    CAST(featureValues[ORDINAL(47)] AS FLOAT64) AS profileRawMeanSecondsToTransaction_df,
    CAST(featureValues[ORDINAL(48)] AS FLOAT64) AS profileRawStdSecondsToTransaction_df,
    CAST(featureValues[ORDINAL(49)] AS FLOAT64) AS profileSecondsToTransactionZScore_df,
    CAST(featureValues[ORDINAL(50)] AS FLOAT64) AS profileRawSessionCount_df,
    CAST(featureValues[ORDINAL(51)] AS FLOAT64) AS profileRawTransactionCount_df,
    CAST(featureValues[ORDINAL(52)] AS FLOAT64) AS profileRawMeanSessionActionCount_df,
    CAST(featureValues[ORDINAL(53)] AS FLOAT64) AS profileRawMeanSessionAction0Count_df,
    CAST(featureValues[ORDINAL(54)] AS FLOAT64) AS profileRawMeanSessionAction1Count_df,
    CAST(featureValues[ORDINAL(55)] AS FLOAT64) AS profileRawMeanSessionAction2Count_df,
    CAST(featureValues[ORDINAL(56)] AS FLOAT64) AS profileRawMeanSessionAction3Count_df,
    CAST(featureValues[ORDINAL(57)] AS FLOAT64) AS profileRawMeanSessionAction4Count_df,
    CAST(featureValues[ORDINAL(58)] AS FLOAT64) AS profileRawMeanSessionAction5Count_df,
    CAST(featureValues[ORDINAL(59)] AS FLOAT64) AS profileRawMeanSessionAction6Count_df,
    CAST(featureValues[ORDINAL(60)] AS FLOAT64) AS profileRawMeanSessionAction7Count_df,
    CAST(featureValues[ORDINAL(61)] AS FLOAT64) AS profileRawMeanSessionAction8Count_df,
    CAST(featureValues[ORDINAL(62)] AS FLOAT64) AS profileRawMeanSessionAction9Count_df,
    CAST(featureValues[ORDINAL(63)] AS FLOAT64) AS profileRawStdSessionActionCount_df,
    CAST(featureValues[ORDINAL(64)] AS FLOAT64) AS profileRawStdSessionAction0Count_df,
    CAST(featureValues[ORDINAL(65)] AS FLOAT64) AS profileRawStdSessionAction1Count_df,
    CAST(featureValues[ORDINAL(66)] AS FLOAT64) AS profileRawStdSessionAction2Count_df,
    CAST(featureValues[ORDINAL(67)] AS FLOAT64) AS profileRawStdSessionAction3Count_df,
    CAST(featureValues[ORDINAL(68)] AS FLOAT64) AS profileRawStdSessionAction4Count_df,
    CAST(featureValues[ORDINAL(69)] AS FLOAT64) AS profileRawStdSessionAction5Count_df,
    CAST(featureValues[ORDINAL(70)] AS FLOAT64) AS profileRawStdSessionAction6Count_df,
    CAST(featureValues[ORDINAL(71)] AS FLOAT64) AS profileRawStdSessionAction7Count_df,
    CAST(featureValues[ORDINAL(72)] AS FLOAT64) AS profileRawStdSessionAction8Count_df,
    CAST(featureValues[ORDINAL(73)] AS FLOAT64) AS profileRawStdSessionAction9Count_df,
    CAST(featureValues[ORDINAL(74)] AS FLOAT64) AS profileSessionActionCountZScore_df,
    CAST(featureValues[ORDINAL(75)] AS FLOAT64) AS profileSessionAction0CountZScore_df,
    CAST(featureValues[ORDINAL(76)] AS FLOAT64) AS profileSessionAction1CountZScore_df,
    CAST(featureValues[ORDINAL(77)] AS FLOAT64) AS profileSessionAction2CountZScore_df,
    CAST(featureValues[ORDINAL(78)] AS FLOAT64) AS profileSessionAction3CountZScore_df,
    CAST(featureValues[ORDINAL(79)] AS FLOAT64) AS profileSessionAction4CountZScore_df,
    CAST(featureValues[ORDINAL(80)] AS FLOAT64) AS profileSessionAction5CountZScore_df,
    CAST(featureValues[ORDINAL(81)] AS FLOAT64) AS profileSessionAction6CountZScore_df,
    CAST(featureValues[ORDINAL(82)] AS FLOAT64) AS profileSessionAction7CountZScore_df,
    CAST(featureValues[ORDINAL(83)] AS FLOAT64) AS profileSessionAction8CountZScore_df,
    CAST(featureValues[ORDINAL(84)] AS FLOAT64) AS profileSessionAction9CountZScore_df,
    CAST(featureValues[ORDINAL(85)] AS FLOAT64) AS profileRawMeanSessionTransactionCount_df,
    CAST(featureValues[ORDINAL(86)] AS FLOAT64) AS profileRawMeanSessionTransactionFromCheckingCount_df,
    CAST(featureValues[ORDINAL(87)] AS FLOAT64) AS profileRawMeanSessionTransactionFromSavingsCount_df,
    CAST(featureValues[ORDINAL(88)] AS FLOAT64) AS profileRawMeanSessionTransactionFromCreditCardCount_df,
    CAST(featureValues[ORDINAL(89)] AS FLOAT64) AS profileRawStdSessionTransactionCount_df,
    CAST(featureValues[ORDINAL(90)] AS FLOAT64) AS profileRawStdSessionTransactionFromCheckingCount_df,
    CAST(featureValues[ORDINAL(91)] AS FLOAT64) AS profileRawStdSessionTransactionFromSavingsCount_df,
    CAST(featureValues[ORDINAL(92)] AS FLOAT64) AS profileRawStdSessionTransactionFromCreditCardCount_df,
    CAST(featureValues[ORDINAL(93)] AS FLOAT64) AS profileSessionTransactionCountZScore_df,
    CAST(featureValues[ORDINAL(94)] AS FLOAT64) AS profileSessionTransactionFromCheckingCountZScore_df,
    CAST(featureValues[ORDINAL(95)] AS FLOAT64) AS profileSessionTransactionFromSavingsCountZScore_df,
    CAST(featureValues[ORDINAL(96)] AS FLOAT64) AS profileSessionTransactionFromCreditCardCountZScore_df,
    CAST(featureValues[ORDINAL(97)] AS FLOAT64) AS profileRecipientTxnCount_df,
    CAST(featureValues[ORDINAL(98)] AS FLOAT64) AS profileDistinctRecipientCount_df,
    CAST(featureValues[ORDINAL(99)] AS FLOAT64) AS age_df,
    CAST(featureValues[ORDINAL(100)] AS FLOAT64) AS genderMale_df,
    CAST(featureValues[ORDINAL(101)] AS FLOAT64) AS maritalStatusSingle_df,
    CAST(featureValues[ORDINAL(102)] AS FLOAT64) AS maritalStatusMarried_df,
    CAST(featureValues[ORDINAL(103)] AS FLOAT64) AS maritalStatusDivorced_df,
    CAST(featureValues[ORDINAL(104)] AS FLOAT64) AS homeLongitude_df,
    CAST(featureValues[ORDINAL(105)] AS FLOAT64) AS homeLatitude_df,
    CAST(featureValues[ORDINAL(106)] AS FLOAT64) AS distanceFromHome_df,
FROM dedupe
