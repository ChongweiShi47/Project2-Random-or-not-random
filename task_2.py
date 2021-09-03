import numpy as np
import matplotlib.pyplot as plt

beta=np.random.uniform(1,2,50)
y=50
gamma=0.95*min(1/beta)
# IGD, the ordering is permitted to have replacement. 

def IGD_wr_task2(beta,y):
    n = len(beta)
    ordering = np.random.choice(n, n, replace=True)    
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    obj_function=[]
    x=[x0]
    for k in range(n):
        beta_k=beta[ordering[k]]
        x1=x0-gamma*beta_k*(x0-y)
        x0=x1
        obj_function.append(1/2*sum(beta*(x1-y)**2))
        x.append(x1)
    return obj_function,x1,x


# IGD, the ordering is not permitted to have replacement.

def IGD_wo_task2(beta,y):
    n = len(beta)
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    obj_function=[]
    x=[x0]
    for k in range(n):
        beta_k=beta[ordering[k]]
        x1=x0-gamma*beta_k*(x0-y)
        x0=x1
        obj_function.append(1/2*sum(beta*(x1-y)**2))
        x.append(x1)
    return obj_function,x1,x

obj_function_1,xk_1,x_1=IGD_wr_task2(beta,y)

obj_function_2,xk_2,x_2=IGD_wo_task2(beta,y)

##plot the history##
for history in [obj_function_1,obj_function_2]:
    plt.plot(history)

for x in [x_1,x_2]:
    plt.plot(x)

'''Conclusion:
    The "without replacement" method is more robust, so that it is better
'''