import numpy as np
from scipy import optimize
from scipy.optimize import minimize
from scipy.special import lambertw

def optimi (h, x, weights = []):
    
    if len(weights) == 0: # default weight
        weights = [1.5 if i%2==1 else 1 for i in range(len(x))]
    
    # experiment parameter
    phi = 100
    p = 3
    u = 0.7
    eta1 = ((u*p)**(1.0/3))/phi
    ki = 10**-26   
    eta2 = u*p/10**-10
    B = 2*10**6
    Vu = 1.1   
    epsilon=B/(Vu*np.log(2))
    
    # local & edge index
    x0=np.where(x==0)[0]
    x1=np.where(x==1)[0]
    
    # local & edge channel gain
    h0=np.array([h[i] for i in x0])
    h1=np.array([h[i] for i in x1])

    w0=np.array([weights[x0[i]] for i in range(len(x0))])
    w1=np.array([weights[x1[i]] for i in range(len(x1))])
    
        
    def sum_rate(s):
        sum1=sum(w0*eta1*(h0/ki)**(1.0/3)*s[0]**(1.0/3))
        sum2=0
        for i in range(len(x1)):
            sum2+=w1[i]*epsilon*s[i+1]*np.log(1+eta2*h1[i]**2*s[0]/s[i+1])
        return sum1+sum2

    def phi(v, j):
        return 1/(-1-1/(lambertw(-1/(np.exp( 1 + v/w1[j]/epsilon))).real))

    def p1(v):
        p1 = 0
        for j in range(len(x1)):
            p1 += h1[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):
        sum1 = sum(w0*eta1*(h0/ki)**(1.0/3))*p1(v)**(-2/3)/3 # lagarange derivative #1
        sum2 = 0
        for j in range(len(x1)):
            sum2 += w1[j]*h1[j]**2/(1 + 1/phi(v,j))# lagarange derivative #2
        return sum1 + sum2*epsilon*eta2 - v
    
    def tau(v, j):
        return eta2*h1[j]**2*p1(v)*phi(v,j)

    # bisection starts here
    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v
    s = []
    s.append(p1(v))
    for j in range(len(x1)):
        s.append(tau(v, j))

   # print('Optimal s:', s)
   # print('Maximum value of sum_rate(s):', sum_rate(s))
    
    return sum_rate(s)

"""    
    def negative_sum_rate(s):
        return -sum_rate(s)
    
    def constraint_all_s_positive(s):
        return s  

    def constraint_sum_s_less_than_one(s):
        return 1 - sum(s)

    constraints = (
        {'type': 'ineq', 'fun': constraint_all_s_positive},
        {'type': 'ineq', 'fun': constraint_sum_s_less_than_one}
    )

    s0 = np.ones(len(w1) + 1) * (1.0 / (len(w1) + 1))

    result = minimize(negative_sum_rate, s0, constraints=constraints)

    optimal_s = result.x
    max_value = -result.fun

    print('Optimal s:', optimal_s)
    print('Maximum value of sum_rate(s):', max_value)
    
        def sum_rate(s):
        sum1=sum(w0*eta1*(h0/ki)**(1.0/3)*s[0]**(1.0/3))
        sum2=0
        for i in range(len(x1)):
            sum2+=w1[i]*epsilon*s[i+1]*np.log(1+eta2*h1[i]**2*s[0]/s[i+1])
        return sum1+sum2

    def phi(v, j):
        return 1/(-1-1/(lambertw(-1/(np.exp( 1 + v/w1[j]/epsilon))).real))

    def p1(v):
        p1 = 0
        for j in range(len(x1)):
            p1 += h1[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):
        sum1 = sum(w0*eta1*(h0/ki)**(1.0/3))*p1(v)**(-2/3)/3 # lagarange derivative #1
        sum2 = 0
        for j in range(len(x1)):
            sum2 += w1[j]*h1[j]**2/(1 + 1/phi(v,j))# lagarange derivative #2
        return sum1 + sum2*epsilon*eta2 - v
    
    def tau(v, j):
        return eta2*h1[j]**2*p1(v)*phi(v,j)
"""
