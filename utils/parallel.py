import threading
from time import time, sleep
from tqdm import tqdm
from hashlib import blake2b


def parallelize_threads(func, param_list):
    """
    Run a function in parallel using multithreading.

    Parameters
    ----------
    func : function
        The function to run in parallel.
    param_list : list
        A list of dictionaries of parameters to pass to the function.

    Returns
    -------
    list
        A list of the results of the function.
    """

    results = [None] * len(param_list)

    # wrap func to pass return values outside of thread through a list
    def wrapper(index, **kwargs):
        results[index] = func(**kwargs)

    # get time hash as unique prefix for thread names
    k = str(time()).encode('utf-8')
    h = blake2b(key=k, digest_size=16)
    prefix = h.hexdigest()

    # start all threads
    for i, params in enumerate(param_list):
        params['index'] = i
        threading.Thread(target=wrapper,
                         kwargs=params,
                         name=f"{prefix}_thread_{i}").start()

    # wait for all threads to finish with progress bar
    threads = {t for t in threading.enumerate() if t.name.startswith(prefix)}
    with tqdm(total=len(param_list)) as pbar:
        while len(threads) > 0:
            remove_threads = []
            # check for finished threads
            for t in threads:
                if not t.is_alive():
                    pbar.update(1)
                    remove_threads.append(t)
            # remove finished threads
            for t in remove_threads:
                threads.remove(t)
            sleep(0.1)

    # return results
    return results


# ### Example

# from time import ctime

# def print_time(threadName):
#     count = 0
#     delay = 1
#     while count < 5:
#         sleep(delay)
#         count += 1
#         print(f"{threadName}: {ctime(time())}")
#     return threadName


# thread_names = [
#     {'threadName': "Thread-1"},
#     {'threadName': "Thread-2"},
#     {"threadName": "Thread-3"}
# ]

# r = parallelize_threads(print_time, thread_names)
# print(r)

# print('done')
