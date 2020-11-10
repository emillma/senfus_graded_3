from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
import time


def get_output(func, q, *args):
    time.sleep(1)
    q.put(func(*args))


def timeout(func, args, timeout):
    q = Queue()
    p = Process(target=get_output, args=[func, q, *args])
    p.start()
    p.join(timeout=timeout)
    p.terminate()
    return None if q.empty() else q.get()


def f():
    return [1, 'emil']


if __name__ == '__main__':
    data = timeout(f, [], 1.5)
    print(data)
