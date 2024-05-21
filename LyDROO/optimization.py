import numpy as np
from scipy.special import lambertw
from scipy.optimize import linprog

def Algo1_NUM(mode,h,w,Q,Y, V=20):
   
    ch_fact = 10**10   # scaling factors to avoid numerical precision problems
    d_fact = 10**6
    Y_factor = 10
    Y = Y*Y_factor
    phi = 100  # number of cpu cycles for processing 1 bit data
    W = 2  # bandwidth MHz
    k_factor = (10**(-26))*(d_fact**3)
    vu =1.1
    
    N0 = W*d_fact*(10**(-17.4))*(10**(-3))*ch_fact # noise power in watt
    P_max = 0.1 # maximum transmit power 100mW
    f_max = 300 # maximum local computing frequency 100MHz
    
    N = len(Q)
    
    if len(w) == 0:
        w = np.ones((N));
    
    
    a =  np.ones((N)) # control parameter in (26) of paper
    q = Q
   
    for i in range(len(a)):
        a[i] = Q[i]  + V*w[i]
       
    energy = np.zeros((N));
    rate = np.zeros((N));
    f0_val =0;
    
            
    idx0=np.where(mode==0)[0]
    M0 = len(idx0)
    
    if M0==0:
        f0_val =0 # objective value of local computing user
    else:
        Y0 = np.zeros((M0)); # virtual engery queue
        a0 = np.zeros((M0));  
        q0 = np.zeros((M0));
        f0 = np.zeros((M0)); # optimal local computing frequency
        for i in range(M0): 
            tmp_id = idx0[i]
            Y0[i] = Y[tmp_id];
            a0[i] = a[tmp_id];
            q0[i] = q[tmp_id]; 
            if Y0[i] == 0:
                f0[i] = np.minimum(phi*q0[i],f_max);
            else:
                tmp1 = np.sqrt(a0[i]/3/phi/k_factor/Y0[i])
                tmp2 = np.minimum(phi*q0[i],f_max)
                f0[i] = np.minimum(tmp1,tmp2)
            energy[tmp_id] = k_factor*(f0[i]**3);
            rate[tmp_id] = f0[i]/phi;
            f0_val =  f0_val + a0[i]*rate[tmp_id] - Y0[i]*energy[tmp_id];                         
            
    idx1=np.where(mode==1)[0]
    M1 = len(idx1) 
    
    
    
    if M1==0:
        f1_val =0 # objective value of local computing users
    else:
        Y1 = np.zeros((M1)); # virtual engery queue
        a1 = np.zeros((M1));  
        q1 = np.zeros((M1));
        h1 = np.zeros((M1)); 
        R_max = np.zeros((M1));
        tau1 = np.zeros((M1));
        
        delta0 = 1; # precision parameter
        lb = 0; # upper and lower bound of dual variable
        ub =10**4;
        
        for i in range(M1):
            tmp_id = idx1[i];
            Y1[i] = Y[tmp_id];
            a1[i] = a[tmp_id];
            q1[i] = q[tmp_id];
            h1[i] = h[tmp_id];
            SNR = h1/N0;
            R_max[i] = W/vu*np.log2(1+ SNR[i]*P_max);
            
        rat = np.zeros((M1)) # c/tau
        e_ratio = np.zeros((M1)) #e/tau
        parac = np.zeros((M1))
        c = np.zeros((M1))
                   
         
        while np.abs(ub - lb) > delta0:
            mu = (lb+ub)/2;
            for i in range(M1):
                if Y1[i] == 0:
                    rat[i] = R_max[i];
                else:
                    A = 1 + mu/Y1[i]/P_max;
                    A = np.minimum(A,20);
                    tmpA = np.real(lambertw(-A*np.exp(-A)))
                    tmp1 = np.minimum(-A/tmpA,10**20); 
                    snr0 = 1/P_max * (tmp1-1);
                    if SNR[i]<=snr0:
                        rat[i] = R_max[i];
                    else:
                        z1 = np.exp(-1)*(mu*SNR[i]/Y1[i]-1);  
                        rat[i] = (np.real(lambertw(z1))+1)*W/np.log(2)/vu; 
                e_ratio[i] = 1/SNR[i]*(2**(rat[i]*vu/W)-1);    
                parac[i] = a1[i] - mu/rat[i] -Y1[i]/rat[i]*e_ratio[i]; 
                if parac[i]>0:
                    c[i] = q1[i]
                else:
                    c[i] = 0                
                tau1[i] = c[i]/rat[i];
      
            if np.sum(tau1)>1:
                lb=mu
            else: 
                ub=mu

        para_e =  np.zeros((M1));
        para   = np.zeros((M1));
        d = np.zeros((M1));
        tau_fact = np.zeros((M1));
        A_matrix = np.zeros((2*M1+1,M1));
        b = np.zeros((2*M1+1));
    
        for i in range(M1):
            para_e[i] = Y1[i]*e_ratio[i]/rat[i];
            para[i] = a1[i] - para_e[i];
            d[i] = q1[i];
            tau_fact[i] = 1/rat[i];
     
        A_matrix[0:M1,:] = np.eye(M1,dtype=int); 
        A_matrix[M1:2*M1,:] = -np.eye(M1,dtype=int);
        A_matrix[2*M1,:] = tau_fact;
    
        b[0:M1] = d
        b[M1:2*M1] = np.zeros((M1))
        b[2*M1] =1

        res = linprog(-para, A_ub=A_matrix, b_ub=b)
        r1 = np.maximum(res.x,0)
        r1 = np.around(r1, decimals=6)  
    
        tau1 = np.zeros((M1));  
        f1_val =0;
    
        for i in range(M1):
            tmp_id = idx1[i]
            tau1[i] = r1[i]/rat[i]
            rate[tmp_id] = r1[i]
            energy[tmp_id] = e_ratio[i]*tau1[i];
            f1_val = f1_val + a1[i]*rate[tmp_id]- Y1[i]*energy[tmp_id];
        
    f_val = f1_val + f0_val 

    
    f_val = np.around(f_val, decimals=6)     
    rate =  np.around(rate, decimals=6)
    energy =  np.around(energy, decimals=6) 
    

    return f_val,rate,energy    

def optimi (h, x, w, Q, Y):
    
    # experiment parameter
    ch_fact = 10**10   # scaling factors to avoid numerical precision problems
    d_fact = 10**6
    phi = 100
    ki = 10**-26   
    Vu = 1.1   
    N = len (Q)
    V = 20
    W = 2  # bandwidth MHz
    
    f_max = 300 # maximum local computing frequency 100MHz
    N0 = W*d_fact*(10**(-17.4))*(10**(-3))*ch_fact # noise power in watt
    P_max = 0.1 # maximum transmit power 100mW
    
    # local & edge index
    x0=np.where(x==0)[0]
    x1=np.where(x==1)[0]
      
    # define parameter a 
    a = np.ones ((N))
    for i in range (N):
        print (Q[i])
        a[i] = Q[i] + V*w[i]
        print (a[i])
        
    def local_fre (a, Y, Q):
        temp1 = np.sqrt (a/3/phi/ki/Y)
        temp2 = min (phi*Q, f_max)
        
        f = min (temp1, temp2)
        return f
    
    def phi_mu (mu, Y):
        A = 1 + mu/Y/P_max
        tempA = np.real(lambertw(-A*np.exp(-A)))
        return 1/P_max*(-A/tempA - 1)
    
    def Lu (mu, temp1, Y):
        tempA = mu*temp1/Y - 1
        tempB = np.real(lambertw(np.exp(-1)*tempA)) + 1
        return W*tempB/np.log(2)/Vu
    
            
    # local optimal
    local_sum = 0
    for i in x0:
        f = 0
        f = local_fre (a[i], Y[i], Q[i])
        local_sum += a[i]*f/phi - Y[i]*ki*f**3
    
    # offloading optimal
    delta = 0.1
    UB = 999999999
    LB = 0
    l_u = np.zeros ((N))
    g_lu = np.zeros ((N))
    
    while UB - LB > delta:        
        mu = (float(UB) + LB)/2
        tau = []
        
        for i in x1:
            r = 0
            temp1 = h[i] / N0
            R_max = W/Vu*np.log2(1 + P_max*temp1)

            # calculate l(mu)
            if temp1 <= phi_mu (mu, Y[i]):
                l_u[i] = R_max
            else:
                l_u[i] = Lu (mu, temp1, Y[i])

            # calculate optimal r_local
            g_lu[i] = 1/temp1*(2**(l_u[i]*Vu/W) - 1)
            temp2 = a[i] - mu/l_u[i] - Y[i]*g_lu[i]/l_u[i]

            if temp2 < 0:
                r = 0
            else:
                r = Q [i]

            # calculate tau
            tau.append (r / l_u[i])
            
        if np.sum (tau) > 1:
            LB = mu
        else:
            UB = mu
            
    # linear programming
    N1 = len(x1)
    c = np.zeros((N1))
    d = []
    lu_trans = []
    A_matrix = np.zeros((2*N1+1,N1))
    b = np.zeros((2*N1+1))
    
    idx = 0
    for i in x1:
        print (a[i])
        temp1 = Y[i] * g_lu[i]/l_u[i]
        print (temp1)
        c[idx] = a[i] - temp1
        d.append (Q[i])
        lu_trans.append (1/l_u[i])
        idx += 1
    
    A_matrix[0:N1,:] = np.eye(N1,dtype=int) 
    A_matrix[N1:2*N1,:] = -np.eye(N1,dtype=int)
    A_matrix[2*N1,:] = lu_trans
    
    b[0:N1] = d
    b[N1:2*N1] = np.zeros((N1))
    b[2*N1] = 1 

    res = linprog(-1*c, A_ub=A_matrix, b_ub=b)
    r_op = np.maximum(res.x,0)
    r_op = np.around(r_op, decimals=6)
    
    tau_op = []
    idx = 0
    online_sum = 0
    
    for i in x1: 
        tau_op.append (r_op[idx]/l_u[i])
        energy = g_lu[i]*tau_op[idx]
        online_sum = online_fre + a1[i]*r_op[idx]- Y1[i]*energy
        idx += 1
    
    frequency = local_sum + online_sum
    
    frequency = np.around(frequency, decimals=6) 
    
    return frequency
    