from multiprocessing import Process, Pipe, shared_memory, Lock
import numpy as np
import time

lock = Lock()

def sharedMem(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    #arr = np.ndarray(shape=(max_size,), dtype=np.uint8, buffer=existing_shm.buf)
    #print('existing_shm name >>>', existing_shm.name)
    return existing_shm


def simpleProcess(name):
    print('simple process!')
    my_shm = sharedMem(name)
    arr = np.ndarray(shape=(5,), dtype=np.uint8, buffer=my_shm.buf)

    while True:
        lock.acquire()
        print('process 1 print >')
        print(arr[:])
        lock.release()
        time.sleep(1)

def simpleProcess_2(name):
    print('simple process 2!')
    my_shm = sharedMem(name)
    arr = np.ndarray(shape=(5,), dtype=np.uint8, buffer=my_shm.buf)

    while True:
        lock.acquire()
        data = arr[:]
        data += 1
        arr[:] = data
        #print(arr[:])
        lock.release()
        time.sleep(1)

if __name__ == "__main__":
    max_size = 1024
    shm = shared_memory.SharedMemory(create=True, size=max_size)
    arr = np.ndarray(shape=(5,), dtype=np.uint8, buffer=shm.buf)
    memName = shm.name
    print('shared memory name?>>', memName)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    arr[3] = 4
    arr[4] = 5    
    
    p = Process(target=simpleProcess, args=(memName, ))
    p.start()

    p2 = Process(target=simpleProcess_2, args=(memName,))
    p2.start()

    p.join()
    p2.join()

    print('end line of code')
    
