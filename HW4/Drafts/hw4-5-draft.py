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

def dfdd(x):
    # This is with x1 = d; x2,x3=s
    return pfpd(x) - pfps(x) * np.linalg.inv()
