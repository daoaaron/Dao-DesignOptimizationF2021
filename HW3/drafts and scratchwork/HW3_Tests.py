# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:51:09 2021

@author: Knight
"""

# HOUSEKEEPIN'
import torch as t
from torch.autograd import Variable
#from math import exp
import numpy as np

# SETUP and SAT PRESSURES.

p_satw=10**(8.071 - 1730.63/(20+233.426))
p_sat14=10**(7.43155 - 1554.679/(20+240.337))

x=[ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ]
p=[ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ]
# =============================================================================
# x=t.tensor([ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ])
# p=t.tensor([ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ])
# data=t.tensor(np.array([[ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ],[ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ]]))
# =============================================================================

#data=np.array([[ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ],[ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ]])

#%% The FUNCTIONS.

def press(x1, a): # pressure calculation per given x1 and parameters a.
    A12= a[0]
    A21= a[1]
    x2=1-x1
    add1 = x1 * t.exp(A12* ( (A21*x2)/(A12*x1 + A21*x2) )**2 ) * p_satw
    add2 = x2 * t.exp(A21* ( (A12*x1)/(A12*x1 + A21*x2) )**2 ) * p_sat14
    return add1+add2
    
def objf(x,p,a): # Total sum objective function.
    #data=zip(x,p) # now we can iterate thru both of them!!
    sum=0
    for i in range(10):
        xi=x[i]
        pi=p[i]
        sum += (press(xi,a)-pi)**2  # Sum the squared difference.
    return sum

# VARIABLE SETUP.
a = Variable(t.tensor([1.0, 6]), requires_grad=True) 
step=.01

#%% TAKING the GRADIENT!
loss=objf(x,p,a)
loss.backward()
print(objf(x,p,a))
#%% Let's run this thing
# Start gradient descent
#for i in range(1000):  # TODO: change the termination criterion
while loss.data.numpy() > .1:
    loss=objf(x,p,a)
    loss.backward()
    print(loss.data.numpy())
    with t.no_grad():
        print('a is' + str(a))
        print('grad is' + str(a.grad))
        a -= step * a.grad
        print('a is now' + str(a))
        print('step')
        # need to clear the gradient at every step, or otherwise it will accumulate...
        a.grad.zero_()
        
print(a.data.numpy())
print(loss.data.numpy())
#%%

#%% IN CLASS EXAMPLE
# Here is a code for gradient descent without line search

