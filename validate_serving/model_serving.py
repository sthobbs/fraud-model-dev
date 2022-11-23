
from gcp_helpers.pubsub import PubSub
from config import project_id, input_topic, output_topic, output_subscription
from time import sleep

raw_data_source_path = "data/raw/data/events.json"
score_event_dest_path = "data/scores/df_scores_raw.json"

def publish_events(delay=1):
    """
    publish raw event data to the Pub/Sub topic for model-serving dataflow to consume.

    Parameters
    ----------
    delay : int
        delay between publishing messages (in seconds)
    """

    # create a publisher
    p = PubSub(project_id, input_topic)
    
    # publish messages
    with open(raw_data_source_path, 'r') as f:
        count = 0
        while True:
            event = f.readline() # get JSON event string
            if not event:
                break
            p.publish_with_callback(event) # publish event to pubsub
            count += 1
            if count % 500 == 0:
                sleep(delay)
                print(f"{count} events published")
    # wait for all messages to be published
    p.wait_for_publish_to_finish()
    print("Finished publishing")
    print(f"Total events published: {count}")


def listen_for_predictions():
    """
    listen for predictions from the model-serving dataflow pipeline
    and save to file.
    """

    # create a subscriber
    s = PubSub(project_id, output_topic, output_subscription)

    # listen for messages
    score_events = s.subscribe(timeout=120)

    # save messages to file
    count = 0
    with open(score_event_dest_path, 'w') as f:
        for score_event in score_events:
            f.write(score_event + '\n')
            count += 1
    print(f"Total events received: {count}")


def run():
    
    # publish raw events
    publish_events(delay=1)

    # listen for score event predictions and save to file
    listen_for_predictions()


if __name__ == "__main__":
    run()
