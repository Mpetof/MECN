import scipy.io as sio                     
import numpy as np 
import tensorflow as tf
from model import Model
from optimization import optimi
import time

def channel ():
    Ad = 4.11
    fc = 915e6 # HZ
    d = 20 # meters
    PL = 3 # pass loss exponent
    
    h = Ad * (3e8 / (4 * np.pi * fc * d))**PL
    return h

def rician (h):
    K_factor = 0.6
    correlation_coefficient = 0.7  
    los_link_power = K_factor * h
    
    hu = np.sqrt(los_link_power) + np.sqrt(0.5 * (1 - 0.6)) * (np.random.normal(0, 1))
    hd = correlation_coefficient * hu + np.sqrt(1 - correlation_coefficient ** 2) * (np.random.normal(0, 1))
    return hu, hd

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:  # Avoid loops
            new_paths = find_all_paths(graph, node, end, path)
            for p in new_paths:
                paths.append(p)
    return paths

def check_single_transition(offloading_list, path_list):
    
    for path in path_list:
        
        path_offloading_list = [offloading_list[task] for task in path]

        transitions = []
        for i in range(1, len(path_offloading_list)):
            if path_offloading_list[i] != path_offloading_list[i - 1]:
                transitions.append(i)

        if len(transitions) == 2:
            if path_offloading_list[transitions[0] - 1] == 0 and path_offloading_list[transitions[0]] == 1 \
            and path_offloading_list[transitions[1] - 1] == 1 and path_offloading_list[transitions[1]] == 0:
                continue
        else:
            if len(transitions) == 0:
                transition_list.append ([-1])
                continue
            else:
                return False
    
    return True

def create_task_index_list(task_set, path_list):
    task_index_list = {task: [] for task in task_set}
    
    for i, path in enumerate(path_list):
        for task in path:
            if task in task_index_list:
                task_index_list[task].append(i)
    
    return task_index_list

def tree():
    weighted_graph = {
        0: {1: 1200}, #KByte
        1: {2: 1500},
        2: {3: 1600, 4: 1200},
        3: {5: 1400, 6: 1800},
        4: {7: 1300, 8: 1500},
        5: {9: 1000},
        6: {9: 2000},
        7: {9: 1000},
        8: {9: 1800}
    }
        
    unweighted_graph = {key: list(value.keys()) for key, value in weighted_graph.items()}
        
    all_paths = find_all_paths(unweighted_graph, 0, 9)
    return weighted_graph, all_paths

if __name__ == "__main__":
    N = 10                     # number of tasks
    n = 30000                  # number of time frames
    K = N                   # initialize K = N
    Memory = 1024          # capacity of memory structure
    
    ave_h = channel()
    graph, path = tree()
    task_list = np.arange(N)
    path_index = create_task_index_list (task_list, path)
    
    network = Model(net = [3, 256, 128, N - 2],
                learning_rate = 0.01,
                training_interval = 20,
                batch_size = 128,
                memory_size = Memory
                )
    
    for i in range(n):        
        hu,hd = rician(ave_h) # Hz
        print (hu)
        fc = np.random.uniform(2, 50) #GHz
        h = [hu, hd, fc]
        print (h)
        x_list = network.decode (h, K)
        eta = []
        for x in x_list:
            x = [0] + x + [0]
            if check_single_transition (x, path, graph, path_index):
                eta.append(optimi (x, fc, hu, hd, path, graph, path_index))
            else:
                continue
        idx = argmax (eta)
        break
        