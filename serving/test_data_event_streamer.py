
from gcp_helpers.streaming_wrapper import StreamingWrapper
from config import raw_data_dir, scored_data_dir, project_id, input_topic, output_subscription


def stream_test_data():
    """
    Stream test raw input data from disk to PubSub and write scored output to disk.
    """
    # start streaming wrapper
    event_streamer = StreamingWrapper(project_id=project_id)

    # startly slowly streaming raw data from disk to pubsub (in a separate thread)
    event_streamer.slow_stream(input_prefix=f"{raw_data_dir}/events_sample.json",
                               pubsub_topic=input_topic,
                               delay=0.1)

    # start listening for output scored events from pubsub and write to disk (in a separate thread)
    event_streamer.read_from_pubsub(pubsub_subscription=output_subscription,
                                    output_file=f"{scored_data_dir}/serving_scores_raw.json")
