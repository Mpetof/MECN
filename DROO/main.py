import scipy.io as sio                     
import numpy as np 
import tensorflow as tf
from model import Model
from optimization import optimi
import time

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)
            
if __name__ == "__main__":
    
    N = 10                     # number of users
    n = 30000                  # number of time frames
    K = N                   # initialize K = N
    Memory = 1024          # capacity of memory structure
    Delta = 32             # Update interval for adaptive K
    
    
    channel = sio.loadmat('./data/data_%d' %N)['input_h'] # input channel gain
    rate_opt = sio.loadmat('./data/data_%d' %N)['output_obj'] # pre-validated optimal computation rate
    channel = channel * 1000000 # increase h to close to 1 for better training

    split_idx = int(.8* len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8* n)) # training/testing data size
    
    DROO = Model (net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory)
    
    K_max = []
    K_list = []
    q_hist = []
    q_ratio = []
    
    start_time=time.time()
    
    for i in range(n):
        
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        
        # update K
        if i > 0 and i % Delta == 0 :
            K = max (K_max [-Delta:-1]) + 1
            K_list.append (min(K +1, N))
            # print (K)
             
        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx
        
        # training
        h = channel[i_idx,:]
        x_list = DROO.decode (h, K)
        
        # choose optimization action
        q_list = []
        for x in x_list:
            q_list.append (optimi (h/1000000, x))
        idx = np.argmax (q_list)
        
        # update computation rate
        q_hist.append (q_list [idx])
        q_ratio.append (q_hist [-1] / rate_opt [i_idx][0])
        
        # update k max
        K_max.append (idx)
        
        # write memory
        DROO.encode (h, x_list[idx])
        
    total_time=time.time() - start_time
    
    print('Total time consumed:%s'%total_time)
    
    save_to_txt(q_hist, "Q_hist.txt")
    save_to_txt(q_ratio,"Q_rate.txt")
    save_to_txt(DROO.loss, "loss.txt")
    save_to_txt(K_list, "K_list.txt")
    
    
    
    