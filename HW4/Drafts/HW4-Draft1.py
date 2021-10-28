# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:46:45 2021

@author: Knight
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

def sample_loss(x): # takes in a vector [x1,x2]
    x1=x[0]
    x2=x[1]
    x3=3-x1-x2
    return -1 *( x1*x2 + x2*x3 +x1*x3)

#%% Start runnin'
x_1 = np.linspace(-10,10)
x_2 = np.linspace(-10,10)

# We need the cartesian combination of these two vectors
param_grid = np.array([[x1i, x2i] for x1i in x_1 for x2i in x_2])

real_loss = [sample_loss(params) for params in param_grid]


low=param_grid[np.array(real_loss).argmin(),:]
x3=3-low[0]-low[1]
# The minimum is at:
print('The min value of ' + str(np.amin(real_loss)) +' is at '+ str(low) + 'with x3 equal to '+ str(x3))
