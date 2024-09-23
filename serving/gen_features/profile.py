from config import n_distinct_actions
from datetime import datetime
import pymongo


def profile_features(txn, db):
    """
    Generate profile features
    
    Parameters
    ----------
    txn : dict
        Transaction event
    db : pymongo.database.Database
        MongoDB database used by the model

    Returns
    -------
    features : dict
        Profile features
    """

    features = {}

    # find customer profile
    profile_date = txn["timestamp"][:7] + "-01"  # truncate datetime down to the 1st of the month
    query_filter = {"profileDate": profile_date, "customerId": txn["customerId"]}
    profile = db.profiles.find_one(query_filter)
    if profile is None:
        profile = {}

    # find login event at the beginning of the current session
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "login",
                    "timestamp": {"$lt": txn["timestamp"]}}
    login = db.events.find_one(query_filter, sort={"timestamp": pymongo.ASCENDING})
    if login is None:
        login = {}

    # indicator of whether or not we have a profle for this customer
    features["profileRawInd"] = 1 if profile else 0

    # amount features
    features["profileRawAmountMin"] = profile.get("amountMin", -1.0)
    features["profileRawAmountMax"] = profile.get("amountMax", -1.0)
    features["profileRawAmountAvg"] = profile.get("amountAvg", -1.0)
    features["profileRawAmountStd"] = profile.get("amountStd", -1.0)
    features["profileRawAmountPercentile10"] = profile.get("amountPercentile10", -1.0)
    features["profileRawAmountPercentile25"] = profile.get("amountPercentile25", -1.0)
    features["profileRawAmountPercentile50"] = profile.get("amountPercentile50", -1.0)
    features["profileRawAmountPercentile75"] = profile.get("amountPercentile75", -1.0)
    features["profileRawAmountPercentile90"] = profile.get("amountPercentile90", -1.0)
    if profile.get("amountStd") is None or profile.get("amountStd") == 0 or \
        profile.get("amountAvg") is None or txn.get("amount") is None:
        features["profileAmountZScore"] = -1.0
    else:
        features["profileAmountZScore"] = (txn["amount"] - profile["amountAvg"]) / profile["amountStd"]

    # time between start of session and first transaction
    features["profileRawMeanSecondsToTransaction"] = profile.get("meanSecondsToTransaction", -1.0)
    features["profileRawStdSecondsToTransaction"] = profile.get("stdSecondsToTransaction", -1.0)
    if profile.get("meanSecondsToTransaction") is None or profile.get("stdSecondsToTransaction") is None or \
        profile.get("stdSecondsToTransaction") == 0 or login.get("timestamp") is None:
        features["profileSecondsToTransactionZScore"] = -1.0
    else:
        txn_time = datetime.strptime(txn["timestamp"][:19], "%Y-%m-%d %H:%M:%S")  # truncating to match BQ logic
        login_time = datetime.strptime(login["timestamp"][:19], "%Y-%m-%d %H:%M:%S")
        secondsToTransaction = (txn_time - login_time).seconds
        features["profileSecondsToTransactionZScore"] = (secondsToTransaction - profile["meanSecondsToTransaction"]) \
                                                        / profile["stdSecondsToTransaction"]

    # number of sessions with transactions
    features["profileRawSessionCount"] = int(profile.get("sessionCount", -1))

    # number of transactions
    features["profileRawTransactionCount"] = int(profile.get("transactionCount", -1))
    
    # session action count averages
    features["profileRawMeanSessionActionCount"] = profile.get("meanSessionActionCount", -1.0)
    features["profileRawMeanSessionAction0Count"] = profile.get("meanSessionAction0Count", -1.0)
    features["profileRawMeanSessionAction1Count"] = profile.get("meanSessionAction1Count", -1.0)
    features["profileRawMeanSessionAction2Count"] = profile.get("meanSessionAction2Count", -1.0)
    features["profileRawMeanSessionAction3Count"] = profile.get("meanSessionAction3Count", -1.0)
    features["profileRawMeanSessionAction4Count"] = profile.get("meanSessionAction4Count", -1.0)
    features["profileRawMeanSessionAction5Count"] = profile.get("meanSessionAction5Count", -1.0)
    features["profileRawMeanSessionAction6Count"] = profile.get("meanSessionAction6Count", -1.0)
    features["profileRawMeanSessionAction7Count"] = profile.get("meanSessionAction7Count", -1.0)
    features["profileRawMeanSessionAction8Count"] = profile.get("meanSessionAction8Count", -1.0)
    features["profileRawMeanSessionAction9Count"] = profile.get("meanSessionAction9Count", -1.0)

    # session action count standard deviations
    features["profileRawStdSessionActionCount"] = profile.get("stdSessionActionCount", -1.0)
    features["profileRawStdSessionAction0Count"] = profile.get("stdSessionAction0Count", -1.0)
    features["profileRawStdSessionAction1Count"] = profile.get("stdSessionAction1Count", -1.0)
    features["profileRawStdSessionAction2Count"] = profile.get("stdSessionAction2Count", -1.0)
    features["profileRawStdSessionAction3Count"] = profile.get("stdSessionAction3Count", -1.0)
    features["profileRawStdSessionAction4Count"] = profile.get("stdSessionAction4Count", -1.0)
    features["profileRawStdSessionAction5Count"] = profile.get("stdSessionAction5Count", -1.0)
    features["profileRawStdSessionAction6Count"] = profile.get("stdSessionAction6Count", -1.0)
    features["profileRawStdSessionAction7Count"] = profile.get("stdSessionAction7Count", -1.0)
    features["profileRawStdSessionAction8Count"] = profile.get("stdSessionAction8Count", -1.0)
    features["profileRawStdSessionAction9Count"] = profile.get("stdSessionAction9Count", -1.0)

    # session action count z-scores
    if profile.get("meanSessionActionCount") is None or profile.get("stdSessionActionCount") is None or \
        profile.get("stdSessionActionCount") == 0:
        features["profileSessionActionCountZScore"] = -1.0
    else:
        query_filter = {"customerId": txn["customerId"],
                        "sessionId": txn["sessionId"],
                        "timestamp": {"$lt": txn["timestamp"]}}
        action_count = db.events.count_documents(query_filter) + 1
        features["profileSessionActionCountZScore"] = \
            (action_count - profile["meanSessionActionCount"]) / profile["stdSessionActionCount"]
    for i in range(n_distinct_actions):
        if profile.get(f"meanSessionAction{i}Count") is None or profile.get(f"stdSessionAction{i}Count") is None or \
            profile.get(f"stdSessionAction{i}Count") == 0:
            features[f"profileSessionAction{i}CountZScore"] = -1.0
        else:
            query_filter = {"customerId": txn["customerId"],
                            "sessionId": txn["sessionId"],
                            "action": f"action_{i}",
                            "timestamp": {"$lt": txn["timestamp"]}}
            action_i_count = db.events.count_documents(query_filter)
            features[f"profileSessionAction{i}CountZScore"] = \
                (action_i_count - profile[f"meanSessionAction{i}Count"]) / profile[f"stdSessionAction{i}Count"]

    # session transaction count averages
    features["profileRawMeanSessionTransactionCount"] = profile.get("meanSessionTransactionCount", -1.0)
    features["profileRawMeanSessionTransactionFromCheckingCount"] = profile.get("meanSessionTransactionFromCheckingCount", -1.0)
    features["profileRawMeanSessionTransactionFromSavingsCount"] = profile.get("meanSessionTransactionFromSavingsCount", -1.0)
    features["profileRawMeanSessionTransactionFromCreditCardCount"] = profile.get("meanSessionTransactionFromCreditCardCount", -1.0)

    # session transaction count standard deviations
    features["profileRawStdSessionTransactionCount"] = profile.get("stdSessionTransactionCount", -1.0)
    features["profileRawStdSessionTransactionFromCheckingCount"] = profile.get("stdSessionTransactionFromCheckingCount", -1.0)
    features["profileRawStdSessionTransactionFromSavingsCount"] = profile.get("stdSessionTransactionFromSavingsCount", -1.0)
    features["profileRawStdSessionTransactionFromCreditCardCount"] = profile.get("stdSessionTransactionFromCreditCardCount", -1.0)

    # session transaction count z-scores
    if profile.get("meanSessionTransactionCount") is None or profile.get("stdSessionTransactionCount") is None or \
        profile.get("stdSessionTransactionCount") == 0:
        features["profileSessionTransactionCountZScore"] = -1.0
    else:
        query_filter = {"customerId": txn["customerId"],
                        "sessionId": txn["sessionId"],
                        "action": "transaction",
                        "timestamp": {"$lt": txn["timestamp"]}}
        transaction_count = db.events.count_documents(query_filter) + 1
        features["profileSessionTransactionCountZScore"] = \
            (transaction_count - profile["meanSessionTransactionCount"]) / profile["stdSessionTransactionCount"]

    account_types = [('checking', 'Checking'),
                    ('savings', 'Savings'),
                    ('credit_card', 'CreditCard')]
    for value, name in account_types:
        if profile.get(f"meanSessionTransactionFrom{name}Count") is None or \
            profile.get(f"stdSessionTransactionFrom{name}Count") is None or \
            profile.get(f"stdSessionTransactionFrom{name}Count") == 0:
            features[f"profileSessionTransactionFrom{name}CountZScore"] = -1.0
        else:
            current_account_type_ind = 1 if txn.get("accountType") == value else 0
            query_filter = {"customerId": txn["customerId"],
                            "sessionId": txn["sessionId"],
                            "action": "transaction",
                            "accountType": value,
                            "timestamp": {"$lt": txn["timestamp"]}}
            features[f"profileSessionTransactionFrom{name}CountZScore"] = \
                (db.events.count_documents(query_filter) + current_account_type_ind - profile[f"meanSessionTransactionFrom{name}Count"]) \
                    / profile[f"stdSessionTransactionFrom{name}Count"]

    # number of times they previously sent money to this recipient
    features["profileRecipientTxnCount"] = 0
    if profile.get('recipients') is not None:
        for d in profile.get('recipients'):
            if d is not None and d.get('recipient') == txn['recipient']:
                features["profileRecipientTxnCount"] = int(d['txnCnt'])
                break
    
    # number of distinct recipients they previously sent money to
    features["profileDistinctRecipientCount"] = 0
    if profile.get('recipients') is not None:
        features["profileDistinctRecipientCount"] = len(profile['recipients'])

    return features
