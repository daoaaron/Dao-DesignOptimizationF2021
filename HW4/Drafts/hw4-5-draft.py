# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:23:13 2021

@author: Knight
"""
import math
import numpy as np
from matplotlib import pyplot as plt

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

def xeval(x,a,dfdd): # For the linesearch.
    d_eval= (x[0]-a*dfdd)[0]
    s_eval= x[1:3] + a* np.transpose( np.matmul(  np.matmul(np.linalg.inv(Phps(x)) , Phpd(x) ), np.transpose([Dfdd(x)]) ) )[0]
    return np.append(d_eval,s_eval)

def linesearch(dfdd, x):
    a=1
    b=.5
    t=.3
    while objfun(xeval(x,a,dfdd)) > (objfun(x) - a*t* dfdd**2):
        a=b*a
    return a

def solve(x):  # Takes in intermediate x value [dk, sk0], gives final x value [dk, sk]
    while np.linalg.norm(np.array([ [ x[0]**2/4 + x[1]**2/5 + x[2]**2/25 -1 ], [x[0]+x[1]-x[2] ] ]))  > e: # While |h| > e....
        phps=Phps(x)
        skj1= np.transpose( np.transpose([x[1:3]]) - np.matmul( np.linalg.inv(phps), np.array([ [ x[0]**2/4 + x[1]**2/5 + x[2]**2/25 -1 ], [x[0]+x[1]-x[2] ] ])   ))  # Step 2 of the solve algorithm, but transposing the output.
        x=np.append(x[0:1], np.transpose(skj1[0]))
    return x
        
    
        

        
#%% THE LOOP
x1=0
x3= 1/12 * ( (600-170*(x1**2))**(1/2) +10*x1)
x2= x3-x1

x0=np.array([x1, x2, x3])  # NEEDS TO SATISFY h=0! Hard coded: x2 and x3 are state variables



e=10**(-3)

x_store=[x0]
err=[]


while np.linalg.norm(Dfdd(x_store[-1])) > e:
    x=x_store[-1]
    dfdd=Dfdd(x)
    print('x is ' + str(x))
    print('dfdd is ' + str(np.linalg.norm(dfdd)))
    err.append( math.log( np.linalg.norm(dfdd)))  # At the beginning of the iteration, what's the error?
    # 4.1
    a= linesearch(dfdd, x)
    # 4.2
    dk= x[0]- a*dfdd
    # 4.3
    sk0= x[1:3] + a* np.transpose(  np.matmul(np.matmul(np.linalg.inv(Phps(x)), Phpd(x)),  np.transpose(dfdd)) )
    xk0=np.append(dk,sk0)  # Intermediate x value.
    print('xk0 is ' +str(xk0))
    
    # 4.4
    x = solve(xk0)
    x_store.append(x)
    
print(x_store[-1])
plt.plot(err)
plt.title('Error')
