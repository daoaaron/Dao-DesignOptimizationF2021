# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:09:02 2021

@author: Knight
"""

#%% HOUSEKEEPIN'
import torch as t
from torch.autograd import Variable
import numpy as np

p_satw=28.824099527405245 #10**(8.071 - 1730.63/(20+233.426))
p_sat14=17.460784103526855  #17.4610**(7.43155 - 1554.679/(20+240.337))
x=[ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ]
p=[ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ]
#%% The FUNCTIONS.

def press(a,xi): # Calculating pressure!!
    x1=xi
    x2=1-x1
    A12= a[0]
    A21= a[1]
    return x1 * t.exp(A12* ( (A21*x2)/(A12*x1 + A21*x2) )**2 ) * p_satw + x2 * t.exp(A21* ( (A12*x1)/(A12*x1 + A21*x2) )**2 ) * p_sat14

def objf(a): # Total sum objective function...
    total=0
    for i in range(len(x)):
        xi=x[i]
        pres=press(a,xi)
        #print(press)
        total = total + (pres-p[i])**2  # Sum the squared difference.
    return total

def lines(a):  # because you just gotta have a line search!
    step=.1 # initiate
    while objf(a-step*a.grad) > objf(a)-step*(0)*np.matmul(a.grad,a.grad):
        step=.5*step
    return step
#%%
a = Variable(t.tensor([1.0, 6]), requires_grad=True) 
e = 100 # Also our error term! Initialize.


#%% RUNNING.
#while loss.data.numpy() > .1:
#for i in range(200):
while e > 0.1:
    objective=objf(a)
    objective.backward()
    step=lines(a)
    print('Objective func is now ' +str(objective.data.numpy()))
    e = t.linalg.norm(a.grad) # update the error value.
    with t.no_grad():
        print('a is ' + str(a))
        print('grad is ' + str(a.grad))
        a -= step * a.grad
        print('a is now ' + str(a) + ' and e is '+str(e))
        print('step')
        # need to clear the gradient at every step, or otherwise it will accumulate...
        a.grad.zero_()
        
print(a.data.numpy())
print(objective.data.numpy())
#%%