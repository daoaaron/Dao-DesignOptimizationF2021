# -*- coding: utf-8 -*-
"""

MODIFIED WTIH DRAG and TWO-DIMSENSIONAL STATE.
 
"""

#%% Setup.

import logging
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#%% ENVIRONMENT PARAMETERS.

FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL = 0.18  # thrust constant
DRAG_CONST = 0.002 # All Cv, rho, area, and 0.5 multiplied together, divided by mass.


#%% SYSTEM DYNAMICS.


class Dynamics(nn.Module):  # An object to keep all system dynamics.

    def __init__(self):
        super(Dynamics, self).__init__()  # this SUPER allows you to pull other classes. 

    @staticmethod
    def forward(state, action):

        """
        action: thrust or no thrust ( a 2-element vector)
        state[0] = x (one-dimensional side-to-side movement, rightward)
        state[1] = y (height)
        state[2] = x_dot (side-to-side velocity, rightward)
        state[3] = y_dot (vertical velocity, downward)
        """
        # What if we used a 2D state [x1,x2] and 2D velocity [x1_dot, x2_dot]
        
        # Apply gravity
        delta_state_gravity = t.tensor([0., 0., 0., -GRAVITY_ACCEL * FRAME_TIME])  # GRAVITY ACTS DOWN!
        
        # We gotta add DRAG HERE TOO!!!!
        delta_state_DRAG = t.tensor([0., 0., -DRAG_CONST * state[2]**2 * FRAME_TIME,  -DRAG_CONST * state[3]**2 * FRAME_TIME])  # Drag is a function of y_dot^2. RESISTS velocity. 

        
        # Thrust
        delta_state_thrust = BOOST_ACCEL * FRAME_TIME * t.tensor([0.,0., 1, 1.]) * t.cat((t.zeros(2),action)) 
        # Need to output a 4x1. But 'action' is a 2x1. So let's use TORCH CONCAT to make it a 4x1 with zeros. ALSO, action can be [-1, 1]

        # Update VELOCITY. ( Not position. )
        state = state + delta_state_thrust + delta_state_gravity + delta_state_DRAG
        #print(state)
        # Update state
        # This won't update velocity (since we already did) BUT will update position with timestep*velocity. 
        step_mat = t.tensor([[1., 0., FRAME_TIME, 0.],
                            [0., 1., 0., 1*FRAME_TIME],  # because here, gravity acts UP (when added, should decrease y) and thrust acts DOWN (when added, should increase y)
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1]])
        state = t.matmul(step_mat, state)
        #print(state)

        return state

#%% a deterministic controller

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            # You can add more layers here. DO IT
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # yeah we added another.
            nn.Tanh() # hehehe trying pos or neg
            #nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        #print(action)
        return action

#%% the SIMULATOR! that rolls out x(1), x(2), ..., x(T)


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        state = [-1., 1.,0., 0.]  # TODO: need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2  # You want everything to be zero. 
    
    
#%% set up the optimizer
# Note:
# 0. LBFGS is a good choice if you don't have a large batch size (i.e., a lot of initial states to consider simultaneously)
# 1. You can also try SGD and other momentum-based methods implemented in PyTorch
# 2. You will need to customize "visualize"
# 3. loss.backward is where the gradient is calculated (d_loss/d_variables)
# 4. self.optimizer.step(closure) is where gradient descent is done

class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.1) #0.01

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return closure()
    
    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize(epoch+1)

    def visualize(self, ep):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0] # First column. 
        y = data[:, 1] # Second column. 
        dxdt= data[:,2]
        dydt=data[:,3]
        fig, (ax1,ax2) = plt.subplots(1, 2)
        ax1.plot(x, y)
        ax1.set_title('[%d] Trajectory' % (ep))
        #plt.plot(x, y)
        ax2.plot(dxdt)
        ax2.plot(dydt)
        ax2.set_title('Velocity')

        plt.show()
        
#%% Now it's time to run the code!

T = 100  # number of time steps
dim_input = 4  # STATE SPACE dimensions
dim_hidden = 8  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(20)  # solve the optimization problem