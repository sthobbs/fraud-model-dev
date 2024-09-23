import pymongo


def customer_info_features(txn, db):
    """
    Generate customer info features
    
    Parameters
    ----------
    txn : dict
        Transaction event
    db : pymongo.database.Database
        MongoDB database used by the model

    Returns
    -------
    features : dict
        Customer info features
    """

    features = {}

    # find customer info
    cust_info = db.customer_info.find_one({"customerId": txn["customerId"]})
    if cust_info is None:
        cust_info = {}

    # find login event at the beginning of the current session
    query_filter = {"customerId": txn["customerId"],
                    "sessionId": txn["sessionId"],
                    "action": "login",
                    "timestamp": {"$lt": txn["timestamp"]}}
    login = db.events.find_one(query_filter, sort={"timestamp": pymongo.ASCENDING})
    if login is None:
        login = {}

    # compute features
    features["age"] = cust_info.get("age", -1)
    features["genderMale"] = 1 if cust_info.get("gender") == 'M' else 0
    features["maritalStatusSingle"] = 1 if cust_info.get("maritalStatus") == 'single' else 0
    features["maritalStatusMarried"] = 1 if cust_info.get("maritalStatus") == 'married' else 0
    features["maritalStatusDivorced"] = 1 if cust_info.get("maritalStatus") == 'divorced' else 0
    features["homeLongitude"] = cust_info.get("homeLongitude", 0.0)
    features["homeLatitude"] = cust_info.get("homeLatitude", 0.0)
    if cust_info.get("homeLongitude") is None or cust_info.get("homeLatitude") is None or \
        login.get("longitude") is None or login.get("latitude") is None:
        features["distanceFromHome"] = -1
    else:
        features["distanceFromHome"] = ((cust_info["homeLongitude"] - login["longitude"]) ** 2 + \
                                        (cust_info["homeLatitude"]  - login["latitude"])  ** 2) ** 0.5

    return features
