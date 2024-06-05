import scipy.io as sio                     
import numpy as np 
import tensorflow as tf
from model import Model
from optimization import optimi
import time
import matplotlib.pyplot as plt

def channel ():
    Ad = 4.11
    fc = 915e6 # HZ
    d = 20 # meters
    PL = 3 # pass loss exponent
    
    h = Ad * (3e8 / (4 * np.pi * fc * d))**PL
    return h

def rician (h, size):
    K_factor = 0.6
    correlation_coefficient = 0.7  
    
    los_component = np.sqrt(K_factor * h)
    sigma = np.sqrt(h*(1-K_factor)/2)
    
    x_u = np.random.normal(los_component, sigma, size)
    y_u = np.random.normal(0, sigma, size)
    hu = x_u**2 + y_u**2
    
    x_d = np.random.normal(los_component, sigma, size)
    y_d = np.random.normal(0, sigma, size)
    hd = x_d**2 + y_d**2
    
    hd = correlation_coefficient * hu + np.sqrt(1 - correlation_coefficient**2) * hd
    
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

def validate(x, path):
    for p in path:
        trans = []
        for i in range (1, len(p)):
            if x[p[i]] != x[p[i - 1]]:
                trans.append(i)
        if len(trans)==2 or len(trans)==0:
            return True
        else:
            return False

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

def mesh():
    weighted_graph = {
        0: {1: 1500}, #KByte
        1: {2: 1000, 3: 1600, 4: 1400},
        2: {5: 1600},
        3: {6: 1300},
        4: {7: 1800},
        5: {8: 2000},
        6: {8: 1500},
        7: {8: 2000},
        8: {9: 1000}
    }
        
    unweighted_graph = {key: list(value.keys()) for key, value in weighted_graph.items()}
        
    all_paths = find_all_paths(unweighted_graph, 0, 9)
    return weighted_graph, all_paths

def general():
    weighted_graph = {
        0: {1: 1800}, #KByte
        1: {2: 1500, 3: 1600, 4: 1500},
        2: {8: 1200},
        3: {5: 1400, 6:1600},
        4: {6: 2000, 7:1800},
        5: {8: 1200},
        6: {8: 1400},
        7: {8: 1300},
        8: {9: 1000}
    }
        
    unweighted_graph = {key: list(value.keys()) for key, value in weighted_graph.items()}
        
    all_paths = find_all_paths(unweighted_graph, 0, 9)
    return weighted_graph, all_paths

if __name__ == "__main__":
    
    N = 10 
    n = 10000
    mode = "tree"
    ave_h = channel()
    graph, path = mesh()
    task_list = np.arange(N)
    hu,hd = rician(ave_h, n)
    fc = np.random.uniform(2, 50, n)
    # path_index = create_task_index_list (task_list, path)
    # print (path_index)
    
    network = Model(net = [3, 256, 128, N - 2],
                learning_rate = 0.01,
                training_interval = 20,
                batch_size = 40,
                memory_size = 128
                )

    for i in range(n):        
        h = [hu[i]*1e8, hd[i]*1e8, fc[i]*0.1]
        h = np.array(h)
        x_list = network.decode (h, N - 2)
        eta = []
        for x in x_list:
            x = np.concatenate(([0], x, [0]))
            if validate (x, path):
                eta.append(optimi (x, fc[i], hu[i], hd[i], path, graph))
            else:
                continue
        idx = np.argmax (eta)
        network.encode (h, x_list[idx])
        if len(network.loss) >= 10:
            if (network.loss[-1]<=0.1) and (np.abs(np.mean(network.loss[-5:]) - np.mean(network.loss[-10:-5])) <= 0.05):
                break
                
    plt.figure(figsize=(10, 6))
    plt.plot(network.loss, 'r')
    plt.xlabel('Training Step')
    plt.savefig('loss.png', dpi=300)
    plt.ylabel('Loss')