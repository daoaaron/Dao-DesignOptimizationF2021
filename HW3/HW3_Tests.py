# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:51:09 2021

@author: Knight
"""

import torch as t
from torch.autograd import Variable
from math import exp
import numpy as np

# SETUP and SAT PRESSURES.

p_satw=10**(8.071 - 1730.63/(20+233.426))
p_sat14=10**(7.43155 - 1554.679/(20+240.337))

x=np.array([ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ])
p=np.array([ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ])


# The FUNCTION.

def press(x1, a): # pressure calculation per given x1 and parameters a.
    A12= a[0]
    A21= a[1]
    x2=1-x1
    add1 = x1 * exp(A12* ( (A21*x2)/(A12*x1 + A21*x2) )**2 ) * p_satw
    add2 = x2 * exp(A21* ( (A12*x1)/(A12*x1 + A21*x2) )**2 ) * p_sat14
    return add1+add2
    
def objf(x,a,p): # Total sum objective function.
    data=list(zip(x,p)) # now we can iterate thru both of them!!
    sum=0
    for xi,pi in data:
        sum += (press(xi,a)-pi)**2  # Sum the squared difference.
    return sum

# VARIABLE SETUP.
a = Variable(t.tensor([1.0, 2.0]), requires_grad=True) 

#%%

print(objf(x,a,p))
























    