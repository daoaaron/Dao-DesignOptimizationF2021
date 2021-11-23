# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:50:10 2021

@author: Knight
"""

import numpy as np

#%% Functions

def objf(x):
    x1=x[0]
    x2=x[1]
    return x1**2+(x2-3)**2
    
def g1(x):
    return x[1]**2-2*x[0]

def g2(x):
    return (x[1]-1)**2 + 5*x[0] - 15

def dLdx(x, mu):
    mu1=mu[0]
    mu2=mu[1]
    x1=x[0]
    x2=x[1]
    top=2*x1-2*mu1+5*mu2
    bottom=2*(mu1+mu2+1)*x2 - 2*(mu2+3)
    return np.array([[top],[bottom]])

#%%
W0=np.identity(2)
W_store=[W0]
mu=np.array([0,0])

e=1e-3

x0=np.array([1,1])
x_store=[x0]
#%%

while dLdx(x_store[-1],mu)>e:
    lol=3
    
