import numpy as np
import matplotlib.pyplot as plt

# generation of exact solution and data y and matrix A.

def generate_problem_task3(m, n, rho):
    A = np.random.normal(0., 1.0, (m, n))
    x = np.random.random(n) # uniform in (0,1)
    w = np.random.normal(0., rho, m)
    y = A@x + w
    return A, x, y

# We generate the problem with 200x100 matrix. rho as 0.01.

A, xstar, y = generate_problem_task3(200, 100, 0.01)

def objective_function(A,x,y):
    sum=0
    for i in range(A.shape[0]):
        sum+=(A[i,:]@x-y[i])**2
    return sum
# In these two functions, we could only focus on the first n steps and try to make comparisons on these data only.
# In practice, it requires more iterations to converge, due to the matrix might not be easy to deal with.
# You can put the ordering loop into a naive loop: namely, we simply perform the IGD code several rounds.

# IGD, the ordering is permitted to have replacement. 

def IGD_wr_task3(y, A):
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    gamma=1e-3
    m,n=A.shape[0],A.shape[1]
    ordering = np.random.choice(n, n, replace=True)
    x0=np.zeros(n)
    obj_function=[]
    x=[]
    for k in range(n):
        alpha_k=A[ordering[k],:]
        x1=x0-gamma*alpha_k*(alpha_k*x0-y[ordering[k]])
        x0=x1
        x.append(np.linalg.norm(x1-xstar))
        obj_function.append(objective_function(A,x1,y))
    return obj_function,x1,x


# IGD, the ordering is not permitted to have replacement.

def IGD_wo_task3(y, A):
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    gamma=1e-3
    m,n=A.shape[0],A.shape[1]
    ordering = np.random.choice(n, n, replace=False)
    x0=np.zeros(n)
    obj_function=[]
    x=[]
    for k in range(n):
        alpha_k=A[ordering[k],:]
        x1=x0-gamma*alpha_k*(alpha_k*x0-y[ordering[k]])
        x0=x1
        x.append(np.linalg.norm(x1-xstar))
        obj_function.append(objective_function(A,x1,y))
    return obj_function,x1,x

obj_function_1,xk_1,x_1=IGD_wr_task3(y,A)

obj_function_2,xk_2,x_2=IGD_wo_task3(y,A)

##plot the history##
for history in [obj_function_1,obj_function_2]:
    plt.plot(history)


##Show the norm of xk-xstar
for x in [x_1,x_2]:
    plt.plot(x)

'''Conclusion:
    Actually in this case,the two methods are quite similar, but if we try more time, we find the method without
    replacement is still a little bit better
'''