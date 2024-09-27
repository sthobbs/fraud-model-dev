
from gcp_helpers.streaming_wrapper import StreamingWrapper
from config import project_id, input_topic, output_subscription, \
    test_data_input_path, test_data_output_path, event_streamer_delay


def stream_test_data():
    """
    Stream test raw input data from disk to PubSub and write scored output to disk.
    """
    # start streaming wrapper
    event_streamer = StreamingWrapper(project_id=project_id)

    # startly slowly streaming raw data from disk to pubsub (in a separate thread)
    event_streamer.slow_stream(input_prefix=test_data_input_path,
                               pubsub_topic=input_topic,
                               sorted_by='timestamp',
                               delay=event_streamer_delay)

    # start listening for output scored events from pubsub and write to disk (in a separate thread)
    event_streamer.read_from_pubsub(pubsub_subscription=output_subscription,
                                    output_file=test_data_output_path)
