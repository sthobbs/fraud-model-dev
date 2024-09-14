from serving.score import multithreaded_scoring
from serving.test_data_event_streamer import stream_test_data
from config import n_processes, test
import multiprocessing


def run():
    """Run scoring job."""

    # Start processes for concurrent scoring
    processes = []
    for _ in range(n_processes):
        p = multiprocessing.Process(target=multithreaded_scoring)
        p.start()
        processes.append(p)

    # wait for all processes to finish (required so that the process doesn't exit, and stop event_streamer)
    for p in processes:
        p.join()


if __name__ == '__main__':

    # if test is True, stream test raw input data from disk to PubSub and write scored output to disk.
    if test:
        stream_test_data()

    run()
