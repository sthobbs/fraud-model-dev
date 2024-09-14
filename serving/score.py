from gcp_helpers.pubsub import PubSub
from config import n_threads, project_id, input_subscription, output_topic
from threading import Thread, current_thread
import multiprocessing
import concurrent.futures


def get_features1(event):
    return {'features_1': 1}

def get_features2(event):
    return {'features_2': 2}

def get_features3(event):
    return {'features_3': 3, 'features_4': 4}

def get_scoreevent(features1, features2, features3):
    return features1 | features2 | features3

def score_next_event(s):
    """
    Score the next event in the queue with within-transactions multithreading,
    then publish it to the output topic.

    Parameters
    ----------
    s : PubSub
        PubSub object containing the PubSub client
    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            # get the next event from the queue
            event = s.received_messages.get()

            # skip non-transaction events since they should not be scored
            if event.get('action') != 'transaction':
                continue
            
            # score the event (concurrently)
            future1 = executor.submit(get_features1, event)  # TODO: write get_features1(event), etc.
            future2 = executor.submit(get_features2, event)
            future3 = executor.submit(get_features3, event)
            features1 = future1.result()
            features2 = future2.result()
            features3 = future3.result()
            scoreevent = get_scoreevent(features1, features2, features3)
            print(f"scored event on {multiprocessing.current_process().name} - {current_thread().name}")

            # publish the scored event to the output topic
            s.publish(str(scoreevent))

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

    # Create threads for scoring
    threads = []
    for _ in range(n_threads):
        t = Thread(target=score_next_event, args=[s])
        threads.append(t)

    # Start threads
    for t in threads:
        t.start()

    # Wait for threads to finish (this is required so that the process doesn't exit, even with threads still running)
    for t in threads:
        t.join()
