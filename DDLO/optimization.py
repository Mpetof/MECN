import numpy as np
from scipy.optimize import minimize

def optimi (d, x):
    
    d = d* 1048576 * 8
    C = 150e6  # Total bandwidth bits/s
    local_computation_time = 4.75e-7  # s/bit
    local_processing_energy_consumption = 3.25e-7  # J/bit
    transmission_energy_consumption = 1.42e-7  # J/bit
    cpu_rate_edge_server = 10e9  # cycles/s
    cpu_cycles_per_bit = 1000 # cycles/bit
    alpha = 1.5e-7  # J/bit
    beta = 1  # Weight between energy consumption and processing delay
    N = d.shape [0]
    
    # bandwidth optimization
    d1 = np.zeros_like(d)
    d0 = np.zeros_like(d)
    
    d1[x == 1] = d[x == 1]
    d0[x == 0] = d[x == 0]
    
    a = np.sum(d1, axis=1)
    
    denominator = np.sum(np.sqrt(a))
    
    if denominator == 0:
        k = 0
    else:
        k = C / denominator
        
    c = k * np.sqrt(a)
    c = c.reshape (-1, 1)
    #print (d1)
    #print (c)
    
    # calculate Q
    El = d0 * local_processing_energy_consumption
    Ec = d1 * transmission_energy_consumption + alpha * d1
    Tl = d0 * local_computation_time
    with np.errstate(divide='ignore', invalid='ignore'):
        Tc = np.where(c != 0, d1 / c + d1 * cpu_cycles_per_bit / cpu_rate_edge_server, 0)
    #Tc = np.nan_to_num(Tc)
    #print (Tc)
    #print (Tl)
    
    Q = 0
    for i in range (N):
        temp1 = np.sum (El[i]) + np.sum (Ec[i])
        temp2 = max (np.sum(Tl[i]), np.sum(Tc[i]))
        Q += temp1 + beta * temp2
    return Q
    
    
    