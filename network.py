import os
import random
import numpy as np

import torch
import constants
import torchvision 
import torch.nn as nn
import torch.optim as optim
 
from torch.distributions import Normal, Categorical

class HDL_Net(nn.Module):

    def __init__(self,                                                         
                 N_input_channels = 1,                      #encoder input channel
                 input_dims       = [2048],                 #FLC input dimension
                 FCL_dims         = [1024, 512, 256],       #FLC network dimension
                 N_output         = 2,                      #output dimension
                 lr               = 1e-4,                   #learning rate
                 epsilon          = 0.1,                    #epsilon for action explorations
                 name             = 'hld_net',              #define the network name
                 checkpt_dir      = 'logs/models'):
        
        super(HDL_Net, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        #initialise input and output dimention
        self.N_output       = N_output

        #initialise vision encoder
        #[(W-K+2P)/S] + 1
        self.encoder = nn.Sequential(
            #W: 128 => 64
            nn.Conv2d(N_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                nn.BatchNorm1d(FCL_dims[0]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[0], FCL_dims[1]),
                                nn.BatchNorm1d(FCL_dims[1]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[1], FCL_dims[2]),
                                nn.BatchNorm1d(FCL_dims[2]),
                                nn.ReLU())

        #initialise outputs
        self.actions_q_values = nn.Linear(FCL_dims[2], self.N_output)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir

        #check if dir exists 
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        #used for preventing 0 value
        self.sm_c = 1e-6                 

        #define epsilon
        self.epsilon = epsilon

        self.to(self.device)

    def forward(self, state):

        #move to correct device
        state = state.to(self.device)

        x = self.encoder(state)
        
        q_values = self.actions_q_values(x)

        return q_values
    
    def make_decisions(self, state):

        if random.random() <= self.epsilon:
            return random.randrange(self.N_output)
        else:
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1).item()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))

class Critic(nn.Module):

    def __init__(self,                                                         
                 max_action       = [0.05, 0.05, 0.05, np.deg2rad(30.)], #action range 
                 N_input_channels = 2,                      #encoder input channel
                 input_dims       = [512],                 #FLC input dimension
                 FCL_dims         = [256, 128],             #FLC network dimension
                 N_output         = 1,                      #output dimension
                 N_action         = 4,                      #action dimension
                 N_action_type    = 3,                      #action type dimension
                 lr               = 1e-4,                   #learning rate
                 name             = 'critic',               #define the network name
                 checkpt_dir      = 'logs/models'):
        
        super(Critic, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        #initialise vision encoder
        #[(W-K+2P)/S] + 1
        self.encoder = nn.Sequential(
            #W: 128 => 64
            nn.Conv2d(N_input_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
        )

        #initialise network layers
        self.fc = nn.Sequential(
            nn.Linear(input_dims[0], FCL_dims[0]),
            nn.BatchNorm1d(FCL_dims[0]),
            nn.ReLU(),
            nn.Linear(FCL_dims[0], FCL_dims[1]),
            nn.BatchNorm1d(FCL_dims[1]),
            nn.ReLU()
        )

        self.Q_value      = nn.Linear(FCL_dims[1], N_output)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir

        #check if dir exists 
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        self.to(self.device)

    def forward(self, state_action):
    
        #move to correct device
        state_action = state_action.to(self.device)

        x = self.encoder(state_action)
        x = self.fc(x)

        #compute q value
        x = self.Q_value(x)
        
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))

class Actor(nn.Module):
    
    def __init__(self,                                                         
                 max_action = [0.05, 0.05, 0.05, np.deg2rad(30.)], #action range 
                 N_input_channels = 2,                    #encoder input channel
                 input_dims       = [2048],                #FLC input dimension
                 FCL_dims         = [1024, 512, 256],       #FLC network dimension
                 N_action         = 4,                    #action dimension
                 N_gripper_action = 2,                    #action type dimension
                 lr               = 1e-4,                 #learning rate
                 action_type      = 'grasp',              #define if this actor is used for grasping or pushing
                 name             = 'actor',              #define the network name
                 checkpt_dir      = 'logs/models'):       #define checkpoint directory
        
        super(Actor, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        #initialise input and output dimention
        self.N_action       = N_action
        self.N_action_type  = N_gripper_action

        #initialise vision encoder
        #[(W-K+2P)/S] + 1
        self.encoder = nn.Sequential(
            #W: 128 => 64
            nn.Conv2d(N_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Flatten(),
        )

        #initialise fully connected layers
        # self.fc = nn.Sequential(
        #     nn.Linear(input_dims[0], FCL_dims[0]),
        #     # nn.BatchNorm1d(FCL_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(FCL_dims[0], FCL_dims[1]),
        #     # nn.BatchNorm1d(FCL_dims[1]),
        #     nn.ReLU()
        # )

        #initialise outputs
        self.mean                = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                                 nn.BatchNorm1d(FCL_dims[0]),
                                                 nn.ReLU(),
                                                 nn.Linear(FCL_dims[0], FCL_dims[1]),
                                                 nn.BatchNorm1d(FCL_dims[1]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[1], FCL_dims[2]),
                                                 nn.BatchNorm1d(FCL_dims[2]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[2], N_action))
        
        self.std                 = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                                 nn.BatchNorm1d(FCL_dims[0]),
                                                 nn.ReLU(),
                                                 nn.Linear(FCL_dims[0], FCL_dims[1]),
                                                 nn.BatchNorm1d(FCL_dims[1]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[1], FCL_dims[2]),
                                                 nn.BatchNorm1d(FCL_dims[2]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[2], N_action))
        if action_type == "grasp":
            self.gripper_actions =  nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                                  nn.BatchNorm1d(FCL_dims[0]),
                                                 nn.ReLU(),
                                                 nn.Linear(FCL_dims[0], FCL_dims[1]),
                                                 nn.BatchNorm1d(FCL_dims[1]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[1], FCL_dims[2]),
                                                 nn.BatchNorm1d(FCL_dims[2]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[2], N_gripper_action))
        else:
            self.gripper_actions = None

        #initialise max action range
        self.max_action = torch.tensor(max_action).to(self.device)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir

        #check if dir exists 
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        #used for preventing 0 value
        self.sm_c = 1e-6                 

        self.to(self.device)

    def forward(self, state):

        #move to correct device
        state = state.to(self.device)

        x = self.encoder(state)

        # x = self.fc(x)

        #compute normal distribution mean
        mean = self.mean(x)

        #compute normal distribution std
        std = self.std(x)

        #clamp the std within range
        std = torch.clamp(std, 
                          min = self.sm_c, 
                          max = 1.)
        
        #compute gripper action type
        if self.gripper_actions is not None:
            gripper_action_probs = self.gripper_actions(x)
        else:
            gripper_action_probs = None

        return mean, std, gripper_action_probs
        # return mean, gripper_action_probs

    def get_actions(self, 
                    state, 
                    is_reparametrerise = True):
        #TODO [FINISH 22 AUG 2024]: add torch.softmax to gripper_action_probs
        mean, std, gripper_action_probs = self.forward(state)

        normal = Normal(mean, std)

        if is_reparametrerise:
            z = normal.rsample()
        else:
            z = normal.sample()

        actions     = torch.tanh(z)*self.max_action
        # actions     = torch.tanh(mean)*self.max_action
        if gripper_action_probs is not None:
            gripper_action = Categorical(torch.softmax(gripper_action_probs, axis = 1)).sample()
        else:
            gripper_action = constants.CLOSE_GRIPPER

        return actions, gripper_action, z, normal, gripper_action_probs

    def compute_log_prob(self, normal, a, z):

        log_probs  = normal.log_prob(z) - torch.log(1-a.pow(2) + self.sm_c)
        log_probs  = log_probs.sum(-1, keepdim = True)

        return log_probs.float()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))