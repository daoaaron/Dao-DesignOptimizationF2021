# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:23:13 2021

@author: Knight
"""

import math
import numpy as np

def objfun(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    return x1**2+x2**2+x3**2

#%% Derivatives.

def Pfpd(x):
    return 2*x[0] 

def Pfps(x):
    return np.array([2*x[1], 2*x[2]])

def Phps(x):
    return np.array([[2/5*x[1], 2/25*x[2]],[1, -1]])

def Phpd(x):
    return np.array([[x[0]/2],[1]])

def Dfdd(x):  # REDUCED GRAD!
    # This is with x1 = d; x2,x3=s
    return Pfpd(x) - np.matmul( np.matmul(Pfps(x), np.linalg.inv(Phps(x))), Phpd(x) )

def xeval(x,a,dfdd):
    d_eval= (x[0]-a*dfdd)[0]
    s_eval= x[1:3] + a* np.transpose( np.matmul(  np.matmul(np.linalg.inv(Phps(x)) , Phpd(x) ), np.transpose([Dfdd(x)]) ) )[0]
    return [d_eval, s_eval[0], s_eval[1] ]
#%%

def linesearch(dfdd, x):
    a=1
    b=.5
    t=.3
    while objfun(xeval(x,a,dfdd)) > (objfun(x) - a*t* dfdd**2):
        a=b*a
    return a
        
#%% THE LOOP

x0=[1, 2, 3]  # Hard coded: x2 and x3 are state variables

e=10**(-3)
k=0

x_store=[x0]

dfdd=100

#%%
while np.linalg.norm(dfdd) > e:
    x=x_store[-1]
    dfdd=Dfdd(x)
    # 4.1
    a= linesearch(dfdd, x)
    # 4.2
    dk= x[0]- a*dfdd
    # 4.3
    sk0= x[1:3] + a* np.transpose(  np.matmul(np.matmul(np.linalg.inv(Phps(x)), Phpd(x)),  np.transpose(dfdd)) )
    # 4.4
    # 4.5
