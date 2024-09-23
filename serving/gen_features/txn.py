import pymongo
from datetime import datetime
from config import n_distinct_actions


def txn_features(txn, db):
    """
    Generate txn features
    
    Parameters
    ----------
    txn : dict
        Transaction event
    db : pymongo.database.Database
        MongoDB database used by the model

    Returns
    -------
    features : dict
        Transaction features
    """

    features = {}

    # find login event at the beginning of the current session
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "login",
                    "timestamp": {"$lt": txn["timestamp"]}}
    login = db.events.find_one(query_filter, sort={"timestamp": pymongo.ASCENDING})
    if login is None:
        login = {}

    # location
    features["longitude"] = login.get("longitude", 0.0)
    features["latitude"] = login.get("latitude", 0.0)

    # amount
    features["amount"] = txn.get("amount", -1.0)
    features["amountMod1"] = -1.0 if txn.get("amount") is None else txn["amount"] % 1
    features["amountMod100"] = -1.0 if txn.get("amount") is None else txn["amount"] % 100
    features["amountMod250"] = -1.0 if txn.get("amount") is None else txn["amount"] % 250
    features["amountMod500"] = -1.0 if txn.get("amount") is None else txn["amount"] % 500
    features["amountMod1000"] = -1.0 if txn.get("amount") is None else txn["amount"] % 1000

    # transaction time
    if txn.get("timestamp") is None:
        features["hour"] = features["dayOfWeek"] = features["dayOfMonth"] = -1
    else:
        features["hour"] = int(txn.get("timestamp")[11:13])
        features["dayOfWeek"] = (datetime.strptime(txn.get("timestamp")[0:10], "%Y-%m-%d").weekday() + 1) % 7 + 1  # convert to BQ day of week
        features["dayOfMonth"] = int(txn.get("timestamp")[8:10])

    # account type
    features["accountTypeChecking"] = 1 if txn.get("accountType") == 'checking' else 0
    features["accountTypeSavings"] = 1 if txn.get("accountType") == 'savings' else 0
    features["accountTypeCreditCard"] = 1 if txn.get("accountType") == 'credit_card' else 0

    # count of each type of action
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "transaction",
                    "timestamp": {"$lt": txn["timestamp"]}}
    features["transactionCount"] = db.events.count_documents(query_filter) + 1
    for i in range(n_distinct_actions):
        query_filter = {"customerId": txn["customerId"],
                        "sessionId": txn["sessionId"],
                        "action": f"action_{i}",
                        "timestamp": {"$lt": txn["timestamp"]}}
        features[f"action{i}Count"] = db.events.count_documents(query_filter)
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "timestamp": {"$lt": txn["timestamp"]}}
    features["actionCount"] = db.events.count_documents(query_filter) + 1

    # total duration and average duration per action
    if login.get("timestamp") is None:
        features["secondsToTransaction"] = -1
        features["avgActionDuration"] = -1.0
    else:
        txn_time = datetime.strptime(txn["timestamp"][:19], "%Y-%m-%d %H:%M:%S")  # truncating to match BQ logic
        login_time = datetime.strptime(login["timestamp"][:19], "%Y-%m-%d %H:%M:%S")
        features["secondsToTransaction"] = (txn_time - login_time).seconds
        features["avgActionDuration"] = features["secondsToTransaction"] / features["actionCount"] 

    # sum/avg/min/max amounts for transactions in session
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "transaction",
                    "timestamp": {"$lte": txn["timestamp"]}}
    feats = ["amountSum", "amountAvg", "amountMin", "amountMax"]
    for f in feats:
        function = f"${f[-3:].lower()}"  # e.g. "amountSum" -> "$sum"
        pipeline = [{"$match": query_filter},
                    {"$group": {"_id": None, "agg_value": {function: "$amount"}}}]
        features[f] = next(db.events.aggregate(pipeline))["agg_value"]

    # count transactions to the current recipient in session
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "transaction",
                    "recipient": txn["recipient"],
                    "timestamp": {"$lt": txn["timestamp"]}}
    features["recipientTransactionCount"] = db.events.count_documents(query_filter) + 1

    # number of distinct recipients
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "transaction",
                    "timestamp": {"$lte": txn["timestamp"]}}
    features["distinctRecipientCount"] = len(db.events.distinct('recipient', query_filter))

    # number of repeated recipients (# txns - # distinct recipients)
    features["repeatedRecipientCount"] = db.events.count_documents(query_filter) - features["distinctRecipientCount"]

    return features
