import scipy.io as sio                     
import numpy as np 
import torch 
import time
import math
from model import Model, MemoryDNN
from optimization import optimi, Algo1_NUM

def plot_rate( rate_his, rolling_intv = 50, ylabel='Normalized Computation Rate', file = "--.png"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.savefig(file, dpi=300)
    plt.show()
    
def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

def gen_channel (N): 
    dist_v = np.linspace(start = 120, stop = 255, num = N)
    Ad = 3
    fc = 915*10**6
    loss_exponent = 3 # path loss exponent
    light = 3*10**8
    h0 = np.ones((N))
    
    for i in range(0,N):
        h0[i] = Ad*(light/4/math.pi/fc/dist_v[i])**(loss_exponent)
        
    return h0

def racian_mec(h,factor):
    n = len(h)
    beta = np.sqrt(h*factor) # LOS channel amplitude
    sigma = np.sqrt(h*(1-factor)/2) # scattering sdv
    
    x = np.multiply(sigma*np.ones((n)),np.random.randn(n)) + beta*np.ones((n))
    y = np.multiply(sigma*np.ones((n)),np.random.randn(n))
    g = np.power(x,2) +  np.power(y,2)
    
    return g


if __name__ == "__main__":
    N = 10                     # number of users
    n = 10000                  # number of time frames
    K = N                   # initialize K = N
    Memory = 1024          # capacity of memory structure
    Delta = 32             # Update interval for adaptive K
    CHFACT = 10**10       # The factor for scaling channel value
    w = [1.5 if i%2==0 else 1 for i in range(N)] # weights for each user
    energy_thresh = np.ones((N))*0.08 # energy comsumption threshold in J per time slot
    nu = 1000 # energy queue factor;
    
    # initialize channel and data arrival
    channel = np.zeros((n,N)) # chanel gains
    
    arrival_lambda = 3*np.ones((N)) # average data arrival, 3 Mbps per user
    dataA = np.zeros((n,N))  # arrival data size
    
    Q = np.zeros((n,N)) # data queue in MbitsW
    Y = np.zeros((n,N)) # virtual energy queue in mJ
    Obj = np.zeros(n) # objective values after solving problem (26)
    energy = np.zeros((n,N)) # energy consumption
    rate = np.zeros((n,N)) # achieved computation rate
    
    # initialize model
    LyDROO = MemoryDNN(net = [N*3, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval = 20,
                    batch_size = 128,
                    memory_size = Memory
                    )
    
    K_max = []
    K_list = []
    
    # generate channel
    h0 = gen_channel (N)
    
    # iteration starts
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        
        # update K
        if i > 0 and i % Delta == 0 :
            K = max (K_max [-Delta:-1]) + 1
            K_list.append (min(K, N))
            # print (K)
             
        #real-time channel generation
        h_tmp = racian_mec(h0,0.3)
        # print (h_tmp)
        
        # increase h to close to 1 for better training;
        h = h_tmp*CHFACT
        channel[i,:] = h
        
        # real-time arrival generation
        dataA[i,:] = np.random.exponential(arrival_lambda)
        
        # 4) ‘Queueing module’ of LyDROO
        if i > 0:
            Q[i,:] = Q[i - 1,:] + dataA [i - 1,:] - rate[i - 1,:]
            Y[i,:] = np.maximum(Y[i-1,:] + (energy[i-1,:]- energy_thresh)*nu,0)
        
        # scale Q and Y to close to 1
        nn_input =np.concatenate( (h, Q[i,:]/10000,Y[i,:]/10000))
        # print (nn_input)
        
        # 1) 'Actor module' of LyDROO
        # generate a batch of actions
        x_list = LyDROO.decode(nn_input, K)
        #print (x_list)
        
        # 2) 'Critic module' of LyDROO
        # allocate resource for all generated offloading modes saved in x_list
        r_list = []
        v_list = []
        for x in x_list:
            r_list.append (Algo1_NUM (x, h, w, Q[i,:], Y[i,:]))
            v_list.append(r_list[-1][0])
            # print (r_list[-1])
        idx = np.argmax (v_list)
        
        K_max.append(idx)
        Obj[i],rate[i,:],energy[i,:]  = r_list[idx]
        
        # 3) 'Policy update module' of LyDROO
        # encode the mode with largest reward
        LyDROO.encode (nn_input, x_list[idx])
        
    plot_rate(Q.sum(axis=1)/N, 100, 'Average Data Queue', "Data Queue.png")
    plot_rate(energy.sum(axis=1)/N, 100, 'Average Energy Consumption', "Average Energy Consumption.png")
    LyDROO.plot_cost()
    
    save_to_txt(Q, "Data Queue.txt")
    save_to_txt(LyDROO.cost_his, "loss.txt")
    save_to_txt(Y, "energy queue.txt")
    save_to_txt(channel/CHFACT, "channel gains.txt")
    save_to_txt(energy, "energy consumption.txt")
    
    sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'energy_queue':Y,'rate':rate,'energy_consumption':energy,'data_rate':rate,'objective':Obj})