import scipy.io as sio                     
import numpy as np 
import time
from model import Model, Memory
from optimization import optimi
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import os


def process_task(task, K, result_queue, shared_memory):
    DLOO = Model(net=[K, 120, 80, K],
                 learning_rate=0.01,
                 training_interval=10,
                 batch_size=128,
                 memory=shared_memory)
    histQ = []
    for d in task:
        dt = d.reshape(1, K)
        x = DLOO.decode(dt[0], K)

        Q = []
        for xi in x:
            xi = xi.reshape(N, M)
            Q.append(optimi(d, xi))
        
        idx = np.argmin(Q)
        histQ.append(Q[idx])

        DLOO.encode(dt[0], x[idx])

    result_queue.put((DLOO.loss, histQ))
    
if __name__ == "__main__":
    N = 3  # Number of WDs
    M = 3  # Number of tasks per user
    K = N*M
    n = 10000 # time frames

    # Randomly generating the input data size for each task between 10MB and 30MB
    task = []
    for i in range (50):
        data = np.random.uniform(10, 30, (N, M)) 
        task.append (data)
    #print (task [0])
    shared_memory = Memory(size = 1024, input_dim=N*M, output_dim=N*M)
    
    
    start_time=time.time()
    
    result_queue = Queue()
    processes = []
    
    num_processes = 3
    tasks_per_process = len(task) // num_processes
    for i in range(num_processes):
        start_index = i * tasks_per_process
        end_index = (i + 1) * tasks_per_process if i < num_processes - 1 else len(task)
        p = Process(target=process_task, args=(task[start_index:end_index], K, result_queue, shared_memory))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    all_losses = []
    all_histQ = []
    while not result_queue.empty():
        losses, histQ = result_queue.get()
        all_losses.append(losses)
        all_histQ.append(histQ)
    
    total_time=time.time() - start_time    
    print('Total time consumed:%s'%total_time)
    
    #DLOO.plot_loss()
    #print (histQ)
    
    plt.figure(figsize=(12, 8))
    for i, losses in enumerate(all_losses):
        plt.plot(losses, label=f'Process {i+1} Loss')

    avg_loss = np.mean([np.mean(losses) for losses in all_losses])
    plt.axhline(y=avg_loss, color='r', linestyle='--', label='Average Loss')

    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('Loss for Each Process and Average Loss')
    plt.legend()
    plt.savefig("loss.png", dpi=300)
    plt.show()
    
    # Plotting histQ for each process
    plt.figure(figsize=(12, 8))
    for i, histQ in enumerate(all_histQ):
        plt.plot(histQ, label=f'Process {i+1} histQ')

    avg_histQ = np.mean([np.mean(histQ) for histQ in all_histQ])
    plt.axhline(y=avg_histQ, color='r', linestyle='--', label='Average histQ')

    plt.xlabel('Iterations')
    plt.ylabel('Q Value')
    plt.title('Q Value for Each Process and Average Q Value')
    plt.legend()
    plt.savefig("Q.png", dpi=300)
    plt.show()