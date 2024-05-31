import numpy as np

def optimi (x, fc, hu, hd, path, graph, path_index):
    
    L = np.array([60.5, 80.3, 152.6, 105.8, 195.3, 86.4, 166.8, 100.3]) # Mcycles
    P_MD = 0.1  # W
    P_AP = 1  # W
    O = len(path)
    betat = 0.5
    betae = 0.5
    k = 1e-26 # J/Hz^2
    fmax = 1e7 # Hz
    W = 2e6 # Hz
    sigma = 1e-10 # W
    Ru = W*np.log2(1 + P_MD*hu/sigma) # bps
    Rd = W*np.log2(1 + P_AP*hd/sigma) # bps
    
    def get_edge_weight(graph, start, end):
        if start in graph and end in graph[start]:
            return graph[start][end]
        else:
            return None
    
    def local_fre (Lambda, path_index):
        completed = {}
        frequency = []
        for p in path_index:
            if p in completed:
                frequency.append (completed[p])
            else:
                temp1 = sum (Lambda[i] for i in p)
                temp2 = (temp1/2/k/betae)**(1/3)
                f = min (temp2, fmax)
                frequency.append (f)
                completed[p] = f
        return frequency
    
    convergence_threshold = 0.01
    max_iterations = 1000
    Lambda = np.full(O, betat/O)
    Lambda_new = np.full (O, 99)
    psi = 0
    
    while np.all(np.abs(Lambda_new - Lambda) > convergence_threshold) and psi < max_iterations:
        fl = local_fre (Lambda, path_index)
        T = np.zeros(O)
        for i in range(O):
            time = 0 # s
            energy = 0 # J
            pre = 0
            for p in path[i]:
                if x[p] == 1:
                    time += L[p]*1e6 / fc*1e9
                if x[p] == 0:
                    time += L[p]*1e6 / fl[p]
                    energy += k*L[p]*1e6*(fl[p]**2)
                if x[pre] == 0 and x[p] == 1:
                    data = get_edge_weight (graph, pre, p)
                    time += data*8 / Ru
                    energy += data / Ru * P_MD
                if x[pre] == 1 and x[p] == 0:
                    data = get_edge_weight (graph, pre, p)
                    time += data*8 / Rd
                pre = p
            T[i] = time
        Tmax = max(T)
        Lambda_new = Lambda - 0.1 * (T - Tmax)
        
        # optimization
        temp1 = (betat - np.sum(Lambda_new)) / O
        ans = Lambda_new + temp1

        while np.any(ans < 0):
            ans[ans < 0] = 0
            index = ans > 0
            count = np.sum(index)
            if count == 0:
                break
            remaining_lambda = Lambda_new[index]
            lambda_sum_active = np.sum(remaining_lambda)
            temp2 = (betat - lambda_sum_active) / count
            ans[active_indices] = remaining_lambda + temp2
        
        psi += 1
        if psi > 1:
            Lambda = Lambda_new
        Lambda_new = ans
        
    fl = local_fre (Lambda, path_index)
    T = np.zeros(O)
    energy = 0
    eta = 0
    for i in range(O):
        time = 0
        pre = 0
        for p in path[i]:
            if x[p] == 1:
                time += L[p]*1e6 / fc*1e9
            if x[p] == 0:
                time += L[p]*1e6 / fl[p]
                energy += k*L[p]*1e6*(fl[p]**2)
            if x[pre] == 0 and x[p] == 1:
                data = get_edge_weight (graph, pre, p)
                time += data*8 / Ru
                energy += data*8 / Ru * P_MD
            if x[pre] == 1 and x[p] == 0:
                data = get_edge_weight (graph, pre, p)
                time += data*8 / Rd
            pre = p
        T[i] = time
    Tmax = max(T)
    eta = betae*energy + betat*Tmax
    return eta