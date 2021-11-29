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

def Dldx(x, mu):
    mu1=mu[0]
    mu2=mu[1]
    x1=x[0]
    x2=x[1]
    top=2*x1-2*mu1+5*mu2
    bottom=2*(mu1+mu2+1)*x2 - 2*(mu2+3)
    return np.array([[top],[bottom]])

#%%

def QP(x,W,A):
    # Need to output [s lam_bar]'
    
    # 1.  Active constraints
    
    
  #  if  # regarding mu > 0 
    
    
   # if # regarding dgdx*s + g <= 0 
    
    return

def merit(x,mu):
    w1=
    objf(x) + w1*max(0,g1(x)) + w2*max(0,g2(x))
    return

def linesearch():
    a=1
    t=0.5
    
    return

def BFGS():
    return


#%% Get ready to run 
W0=np.identity(2)
W_store=[W0]
mu=np.array([0,0])

e=1e-3

x0=np.array([1,1])
x_store=[x0]

grads=np.array([])  # Storing g. Is this the 
norm=np.linalg.norm(Dldx(x0,mu))  # Initialize.
norms=[norm]
#%%

while norm >e:
    x=x_store[-1]
   
    
    
    # 3.1. Solve QP for sk, mu_k+1
    
    # 3.2. Perform a line search. Take in mu_k+1, sk. 
        # Update weights in merit function w based on mu_k+1.
    
    # 3.3 calcualte x_k+1.
    
    
    
    # 3.4 Calculate W_k+1 with BFGS.
        # 3.4.1 Evaluate dLdx for x_k, mu_k+1
        # 3.4.2 Evaluate dLdx for x_k+1, mu_k+1
        # 3.4.3 Evaluate theta
        # 3.4.4 Evaluate y
        # 3.4.5 Perform BFGS with y, sk, Wk
    
    # Calculate the gradient. 
    
    # find the norm of the gradient.

#%%

    
    
    
    
