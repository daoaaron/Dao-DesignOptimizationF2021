# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:09:02 2021

@author: Knight
"""

#%% HOUSEKEEPIN'
import torch as t
from torch.autograd import Variable
import numpy as np


#%% The FUNCTIONS.

def objf(a): # Total sum objective function...
    p_satw=28.824099527405245 #10**(8.071 - 1730.63/(20+233.426))
    p_sat14=17.460784103526855  #17.4610**(7.43155 - 1554.679/(20+240.337))
    x=[ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ]
    p=[ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ]

    total=0
    for i in range(len(x)):
        x1=x[i]
        x2=1-x1
        A12= a[0]
        A21= a[1]
        press = x1 * t.exp(A12* ( (A21*x2)/(A12*x1 + A21*x2) )**2 ) * p_satw + x2 * t.exp(A21* ( (A12*x1)/(A12*x1 + A21*x2) )**2 ) * p_sat14
        #print(press)
        total = total + (press-p[i])**2  # Sum the squared difference.
    return total

# VARIABLE SETUP.
a = Variable(t.tensor([1.0, 6]), requires_grad=True) 
a = Variable(t.tensor([-16.5242,   -1.8135]), requires_grad=True)

step=.0001
#%%
def lines(a):
    step=.001 # initiate
    while objf(a-step*a.grad) > objf(a)-step*(.5)*np.matmul(a.grad,a.grad):
        step=.5*step
    return step

#%%
# =============================================================================
# 
# objective=objf(a)
# objective.backward()
# print(a.grad)    
# 
# objf(a)
# =============================================================================
#%% RUNNING.
#while loss.data.numpy() > .1:
for i in range(50):
    objective=objf(a)
    objective.backward()
    step=lines(a)
    print('Objective func is now ' +str(objective.data.numpy()))
    with t.no_grad():
        print('a is ' + str(a))
        print('grad is ' + str(a.grad))
        a -= step * a.grad
        print('a is now ' + str(a))
        print('step')
        # need to clear the gradient at every step, or otherwise it will accumulate...
        a.grad.zero_()
        
print(a.data.numpy())
print(objective.data.numpy())
#%%