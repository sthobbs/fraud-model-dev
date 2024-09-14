from gcp_helpers.pubsub import PubSub
from glob import glob
import time
from threading import Thread
import os

class StreamingWrapper():
    """
    Wrapper class for streaming events from local files to Google Cloud Pub/Sub and vice versa.
    """

    def __init__(self, project_id):
        """
        Initialize the StreamingWrapper class.

        Parameters
        ----------
        project_id : str
            Google Cloud project ID
        """
        
        self.project_id = project_id

    def non_blocking(func):
        """
        Decorator function to run a function in a separate thread without blocking the main thread.

        Parameters
        ----------
        func : function
            Function to be run in a separate thread
        """
        def wrapper(*args, **kwargs):
            if kwargs.get('blocking'):
                func(*args, **kwargs)
            else:
                t = Thread(target=func, args=args, kwargs=kwargs)
                t.start()
        return wrapper

    @non_blocking
    def slow_stream(self, input_prefix, pubsub_topic, delay=0.1, blocking=False):
        """
        Stream events from local files with 'input_prefix' file prefix to 'pubsub_topic',
        with 'delay' seconds between each event.

        Parameters
        ----------
        input_prefix : str
            Prefix of local folder containing input files
        pubsub_topic : str
            Google Cloud Pub/Sub topic ID
        delay : float
            Delay in seconds between each event
        blocking : bool
            If True, wait for all events to be published before returning
        """

        # create a publisher client
        s = PubSub(project_id=self.project_id, topic_id=pubsub_topic)

        # load data from local files and publish to PubSub
        files = glob(f'{input_prefix}*')
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    s.publish(line)
                    time.sleep(delay)

    @non_blocking
    def read_from_pubsub(self, pubsub_subscription, output_file, timeout=None, blocking=False):
        """
        Read events from 'pubsub_subscription' and write to 'output_file'.

        Parameters
        ----------
        pubsub_subscription : str
            Google Cloud Pub/Sub subscription ID
        output_file : str
            Output file
        timeout : float
            Timeout in seconds that the subscriber should listen for messages.
        blocking : bool
            If True, wait for all events to be published before returning
        """

        # create a subscriber client and subscribe
        s = PubSub(project_id=self.project_id, subscription_id=pubsub_subscription)
        s.subscribe(timeout=timeout, blocking=blocking)

        # remove file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # read events from PubSub and write to file
        with open(output_file, 'a') as f:
            while True:
                event = s.received_messages.get()
                # json.dump(event, f)
                # print(output_file, event)
                f.write(event)
                f.write('\n')
                f.flush()
                s.received_messages.task_done()

