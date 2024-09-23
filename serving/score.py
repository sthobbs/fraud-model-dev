from gcp_helpers.pubsub import PubSub
from config import n_threads, project_id, input_subscription, output_topic, \
    mongo_client_uri, db_name, event_collection, model_path, model_id, pause_before_feature_gen
from serving.gen_features.txn import txn_features
from serving.gen_features.profile import profile_features
from serving.gen_features.customer_info import customer_info_features
from threading import Thread, current_thread
import multiprocessing
import concurrent.futures
import json
import pymongo
import pickle
import datetime
import time

def write_to_mongo(db, collection, event):
    """Write event to mongo"""
    db[collection].insert_one(event)

def get_scoreevent(txn, features_dict, model):

    # get all features in the correct order
    features_list = []
    feature_names = model.feature_names_in_.tolist()
    for f in feature_names:
        features_list.append(features_dict[f])

    # get score
    score = model.predict_proba([features_list])[0][1].item()

    # construct score event
    now = datetime.datetime.now()
    score_event = {
        "fraudLabel": txn['fraudLabel'],
        "uniqueId": txn['uniqueId'],
        'customerId': txn['customerId'],
        'sessionId': txn['sessionId'],
        'timestamp': txn['timestamp'],
        'scoreTimestamp': now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        'action': txn['action'],
        'amount': txn['amount'],
        'recipient': txn['recipient'],
        'modelId': model_id,
        'score': score,
        'featureNamesStr': ", ".join(feature_names),
        'featureValuesStr': ", ".join(str(i) for i in features_list),
    }

    return score_event

def score_next_event(s, db):
    """
    Score the next event in the queue with within-transactions multithreading,
    then publish it to the output topic.

    Parameters
    ----------
    s : PubSub
        PubSub object containing the PubSub client
    db : pymongo.database.Database
        MongoDB database used by the model
    """

    # load the model
    # self.logger.info(f"Loading model object from {path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
            # self.logger.info(f"model loaded from path: {path}")
    except Exception:
        # self.logger.error(f"error loading model from path: {path}")
        raise

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            # get the next event from the queue
            event = s.received_messages.get()
            start_time = datetime.datetime.now()
            event = json.loads(event)
            
            # write event to mongo
            executor.submit(write_to_mongo, db, event_collection, event)

            # only transaction events should be scored
            if event.get('action') != 'transaction':
                continue

            # pause before feature gen
            time.sleep(pause_before_feature_gen)

            # score the event (concurrently)
            future1 = executor.submit(txn_features, event, db)
            future2 = executor.submit(profile_features, event, db)
            future3 = executor.submit(customer_info_features, event, db)
            features1 = future1.result()
            features2 = future2.result()
            features3 = future3.result()
            features = features1 | features2 | features3
            scoreevent = get_scoreevent(event, features, model)
            print(f"scored event on {multiprocessing.current_process().name} - {current_thread().name}", s.received_messages.qsize())

            # calculate latency
            end_time = datetime.datetime.now()
            latency = (end_time - start_time).total_seconds()
            scoreevent['latency'] = latency

            # publish the scored event to the output topic
            s.publish(str(scoreevent), block=False)

            # mark the event as processed
            s.received_messages.task_done()



def multithreaded_scoring():
    """Multithreaded scoring function for unbounded data from Pub/Sub."""

    print(f"starting scoring on {multiprocessing.current_process().name}")

    # Create a subscriber client and subscribe
    s = PubSub(project_id=project_id,
               topic_id=output_topic,
               subscription_id=input_subscription)
    s.subscribe(timeout=None, blocking=False)

    # Create a mongo client
    mongo_client = pymongo.MongoClient(mongo_client_uri)
    db = mongo_client[db_name]

    # Create threads for scoring
    threads = []
    for _ in range(n_threads):
        t = Thread(target=score_next_event, args=[s, db])
        threads.append(t)

    # Start threads
    for t in threads:
        t.start()

    # Wait for threads to finish (this is required so that the process doesn't exit, even with threads still running)
    for t in threads:
        t.join()
