# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:21:30 2021

@author: Knight
"""

import torch as t
from torch.autograd import Variable


x = Variable(t.tensor([1.0, 0.0]), requires_grad=True)

# Fix the step size
a = 0.01

b=[-1, -2]

# Here's a loss
def loos(x):
    sum1= (x[0] + b[0])**2
    sum2=(x[1] - b[1])**2
    return sum1 + sum2
#%%
# Start gradient descent
for i in range(1000):  # TODO: change the termination criterion
    loss = loos(x)
    loss.backward()
    print(loss.data.numpy())
    with t.no_grad():
        print(x)
        print(x.grad)
        x -= a * x.grad
        print(x)
        print('step')
        # need to clear the gradient at every step, or otherwise it will accumulate...
        x.grad.zero_()
        
print(x.data.numpy())
print(loss.data.numpy())


#%% And now we modify it

x = Variable(t.tensor([1.0, 2.0]), requires_grad=True)

p_satw=10**(8.071 - 1730.63/(20+233.426))
p_sat14=10**(7.43155 - 1554.679/(20+240.337))

q=[ 0.0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0 ]
p=[ 28.1 , 34.4 , 36.7 , 36.9 , 36.8 , 36.7 , 36.5 , 35.4 , 32.9 , 27.7 , 17.5 ]

def loos(x):
    sum=0
    for i in range(len(x)):
        q1=q[i]
        q2=1-q1
        A12= x[0]
        A21= x[1]
        press = q1 * t.exp(A12* ( (A21*q2)/(A12*q1 + A21*q2) )**2 ) * p_satw + q2 * t.exp(A21* ( (A12*q1)/(A12*q1 + A21*q2) )**2 ) * p_sat14
        sum += (press-p[i])**2  # Sum the squared difference.
    return sum
#%%

# Start gradient descent
for i in range(20):  # TODO: change the termination criterion
    loss = loos(x)
    loss.backward()
    print(loss.data.numpy())
    with t.no_grad():
        print(x)
        print(x.grad)
        x -= a * x.grad
        print(x)
        print('step')
        # need to clear the gradient at every step, or otherwise it will accumulate...
        x.grad.zero_()
        
print(x.data.numpy())
print(loss.data.numpy())