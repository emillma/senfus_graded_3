from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
import time


def get_output(func, q, *args):
    retval = func(*args)
    q.put(retval)
    return 0


def timeout(func, args, timeout):
    q = Queue()
    p = Process(target=get_output, args=[func, q, *args])
    time.sleep(0.1)
    p.start()
    start_time = time.time()
    while q.empty() and time.time() < start_time + timeout:
        time.sleep(0.01)
    data = None if q.empty() else q.get()
    p.terminate()
    return data


def f(a):
    return [1, 'emil', a]


if __name__ == '__main__':
    data = timeout(f, [123], 1.5)
    print(data)
