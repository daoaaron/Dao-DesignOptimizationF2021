# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:32:02 2021

@author: Knight
"""
import numpy as np

#region Our Functions! ------------------
#objfun= lambda x: (x[0]+1)**2 + (x[1])**2 + (x[2]-1)**2
#grad = lambda x: np.array([2*(x[0]+1),2*x[1],2*(x[2]-1)])
objfun= lambda x: ((1-2*x[0]-3*x[1])+1)**2 + (x[0])**2 + (x[1]-1)**2
grad = lambda x: np.array([10*x[0]+12*x[1]-8, 12*x[0]+20*x[1]-14]) # Note that x2 is x[0] and x3 is x[1]
# A test ------------------------

#endregion

#%% GRAD SETUP.
x0 = np.array([0, 0]).T # Initial guess.
x_solve_grad = [x0]
x= x_solve_grad[0]
a=.01
e=10 # Initialize!

# Let's use an AMIJO line search
def amijo(x):
    a=1 # initiate
    while objfun(x-a*grad(x)) > objfun(x)-a*(.5)*np.matmul(grad(x),grad(x)):
        #m1=objfun(x)
        #m2=objfun(x)-a*(0.8)*np.matmul(g,g)
        a=.5*a
    return a
    
    
while e > .001:
    a=amijo(x)
    x= x - a*grad(x)
    #x_solve=np.concatenate((x_solve, x), axis=1)
    x_solve_grad.append(x)
    e = np.linalg.norm(grad(x))
    print(e)



#%% NEWTON SETUP.
x0 = np.array([0, 0]).T # Initial guess.
x_solve_newton = [x0]
x= x_solve_newton[0]
a=.01
e=10 # Initialize!
H=np.array([[10, 12],[12,20]])

# Let's use an AMIJO line search
def amijo(x):
    a=1 # initiate
    while objfun(x-a*grad(x)) > objfun(x)-a*(.5)*np.matmul(grad(x),np.matmul(np.linalg.inv(H),grad(x))):
        a=.5*a
    return a
    
    
while e > .001:
    a=amijo(x)
    x= x - a*grad(x)
    #x_solve=np.concatenate((x_solve, x), axis=1)
    x_solve_newton.append(x)
    e = np.linalg.norm(grad(x))
    print(e)
    
    

