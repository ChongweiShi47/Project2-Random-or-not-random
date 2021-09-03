#generate a vector of random numbers which obeys the given distribution.
#
# n: length of the vector
# mu: mean value
# sigma: standard deviation.
# dist: choices for the distribution, you need to implement at least normal 
#       distribution and uniform distribution.
#
# For normal distribution, you can use ``numpy.random.normal`` to generate.
# For uniform distribution, the interval to sample will be [mu - sigma/sqrt(3), mu + sigma/sqrt(3)].

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

def generate_random_numbers(n, mu, sigma, dist="normal"):
    # write your code here.
    if dist == "normal":
        return np.random.normal(mu, sigma, n)
    elif dist == "uniform":
        # write your code here.
        low=mu-sigma/sqrt(3)
        high=mu+sigma/sqrt(3)
        return np.random.uniform(low,high,n)
    else:
        raise Exception("The distribution {unknown_dist} is not implemented".format(unknown_dist=dist))
        
        
# test your code:
y_test = generate_random_numbers(1000, 0, 0.1, "normal")
y_test_uniform=generate_random_numbers(2000,0,0.1,"uniform")
sns.distplot(y_test,hist = False, kde = True,kde_kws = {'linewidth': 3})
sns.distplot(y_test_uniform,hist=True,kde=True,kde_kws={'Linewidth': 3})

#Settings of minimization problems:

y1=generate_random_numbers(105, 0.5, 1.0,'normal')
y2=generate_random_numbers(105, 0.5, 1.0,'uniform')

# IGD, the ordering is permitted to have replacement. 

def IGD_wr_task1(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    obj_function=[]
    x=[x0]
    for k in range(n):
        gamma_k=1/(k+1)
        x1=x0-gamma_k*(x0-y[ordering[k]])
        x0=x1
        obj_function.append(1/2*sum((x1-y)**2))
        x.append(x1)
    return obj_function,x1,x



# IGD, the ordering is not permitted to have replacement.

def IGD_wo_task1(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    obj_function=[]
    x=[x0]
    for k in range(n):
        gamma_k=1/(k+1)
        x1=x0-gamma_k*(x0-y[ordering[k]])
        x0=x1
        obj_function.append(1/2*sum((x1-y)**2))
        x.append(x1)
    return obj_function,x1,x


'''Get the result of normal distribution'''

obj_function_norm_1,xk_norm_1,x_norm_1=IGD_wr_task1(y1)

obj_function_norm_2,xk_norm_2,x_norm_2=IGD_wo_task1(y1)

##plot the history

for history in [obj_function_norm_1,obj_function_norm_2]:
    plt.plot(history)

'''Get the result of uniform distribution'''

obj_function_uniform_1,xk_uniform_1,x_uniform_1=IGD_wr_task1(y2)

obj_function_uniform_2,xk_uniform_2,x_uniform_2=IGD_wo_task1(y2)

##plot the history

for history in [obj_function_uniform_1,obj_function_uniform_2]:
    plt.plot(history)


'''Conclusion and breifly prove:
  
    According to the plot, we found the 'without_replacement' method can generate better result of the task1. 
    Because of multiple times of randomization, IGD_wo_task1 can always converge to the mean of y and 
    the value of the objective result is more steady
    
    Proof of convergence:
        y_mean-x_{k+1}=y_mean-x_k+gamman_k(x_k-y_{ik})=y_mean-k/(k+1)*x_k-1/(k+1)*y_{ik}=
        y_mean-k/(k+1)*((k-1)/k*x_{k-1}+1/k*y_{ik-1})-1/(k+1)*y_{ik}=
        y_mean-(k-1)/(k+1)*x_{k-1}-1/(k+1)(y_{ik}+y_{ik-1})=.......
        =y_mean-0-1/(k+1)*(k)*y_mean...0 when k is sufficiently large
'''



'''plot the xk'''
for x in [x_norm_1,x_norm_2]:
    plt.plot(x)


for x in [x_uniform_1,x_uniform_2]:
    plt.plot(x)
