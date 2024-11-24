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
                 N_input_channels = 1, #encoder input channel
                 input_dims = [1024], #FLC input dimension
                 FCL_dims = [512, 256], #FLC network dimension
                 N_output = 2, #output dimension
                 lr = 1e-4, #learning rate
                 epsilon = 0.25, #epsilon for action explorations
                 name = 'hld_net', #define the network name
                 checkpt_dir = 'logs/models'):
        
        super(HDL_Net, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        #initialise input and output dimention
        self.N_output       = N_output

        #initialise vision encoder
        #[(W-K+2P)/S] + 1
        self.encoder = nn.Sequential(
            #W: 128 => 64
            nn.Conv2d(N_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                nn.BatchNorm1d(FCL_dims[0]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[0], FCL_dims[1]),
                                nn.BatchNorm1d(FCL_dims[1]),
                                nn.ReLU())

        #initialise outputs
        self.actions_q_values = nn.Linear(FCL_dims[1], self.N_output)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir

        #check if dir exists 
        self.name = name
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, self.name))

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
        
        x = self.fc(x)

        q_values = self.actions_q_values(x)

        return q_values
    
    def make_decisions(self, state, take_max = False):
        
        q_values = self.forward(state)

        prob = random.random()

        if take_max or prob > self.epsilon:
            index = torch.argmax(q_values, dim=1).item()
            print(f"[HLD net] max output: {index}")
        else:
            index = random.randrange(self.N_output)
            print(f"[HLD net] random output: {index}")
            
        return q_values, index

    def save_checkpoint(self, is_best = False, name = None):
        if is_best:
            torch.save(self.state_dict(), self.checkpt_file)
        elif not is_best and name is not None:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_' + name + '_checkpt'))
            torch.save(self.state_dict(), checkpt_file)
        else:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_checkpt'))
            torch.save(self.state_dict(), checkpt_file)

    def load_checkpoint(self, is_best = False):
        if is_best:
            self.load_state_dict(torch.load(self.checkpt_file))
        else:
            file = os.path.abspath(os.path.join(self.checkpt_dir, self.name + '_checkpt'))
            self.load_state_dict(torch.load(file))

class Critic(nn.Module):

    def __init__(self,                                                         
                 max_action       = [0.05, 0.05, 0.05, np.deg2rad(30.)], #action range 
                 N_input_channels = 2, #encoder input channel
                 input_dims       = [1024], #FLC input dimension
                 FCL_dims         = [512, 256], #FLC network dimension
                 N_output         = 1, #output dimension
                 N_action         = 4, #action dimension
                 N_action_type    = 3, #action type dimension
                 lr               = 1e-4, #learning rate
                 name             = 'critic', #define the network name
                 checkpt_dir      = 'logs/models'):
        
        super(Critic, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        #initialise vision encoder
        #[(W-K+2P)/S] + 1
        self.encoder = nn.Sequential(
            #W: 128 => 64
            nn.Conv2d(N_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(1024),
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
        self.name = name
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

    def save_checkpoint(self, is_best = False, name = None):

        if is_best:
            torch.save(self.state_dict(), self.checkpt_file)
        elif not is_best and name is not None:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_' + name + '_sac_checkpt'))
            torch.save(self.state_dict(), checkpt_file)
        else:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_sac_checkpt'))
            torch.save(self.state_dict(), checkpt_file)

    def load_checkpoint(self, is_best = False):
        if is_best:
            self.load_state_dict(torch.load(self.checkpt_file))
        else:
            file = os.path.abspath(os.path.join(self.checkpt_dir, self.name + '_sac_checkpt'))
            self.load_state_dict(torch.load(file))

class Actor(nn.Module):
    
    def __init__(self,                                                         
                 max_action = [0.05, 0.05, 0.05, np.deg2rad(30.)], #action range 
                 N_input_channels = 2, #encoder input channel
                 input_dims = [1024], #FLC input dimension
                 FCL_dims = [512, 256], #FLC network dimension
                 N_action = 4, #action dimension
                 N_gripper_action = 2, #action type dimension
                 lr = 1e-4, #learning rate
                 action_type = 'grasp', #define if this actor is used for grasping or pushing
                 name = 'actor', #define the network name
                 checkpt_dir = 'logs/models'): #define checkpoint directory
        
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
            nn.Conv2d(N_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
        )

        #initialise outputs
        self.mean                = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                                 nn.BatchNorm1d(FCL_dims[0]),
                                                 nn.ReLU(),
                                                 nn.Linear(FCL_dims[0], FCL_dims[1]),
                                                 nn.BatchNorm1d(FCL_dims[1]),
                                                 nn.ReLU(), 
                                                 nn.Linear(FCL_dims[1], N_action))
        
        self.std                 = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                                 nn.BatchNorm1d(FCL_dims[0]),
                                                 nn.ReLU(),
                                                 nn.Linear(FCL_dims[0], FCL_dims[1]),
                                                 nn.BatchNorm1d(FCL_dims[1]),
                                                 nn.ReLU(),
                                                 nn.Linear(FCL_dims[1], N_action))
        
        # if action_type == "grasp":
        #     self.gripper_actions =  nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
        #                                           nn.BatchNorm1d(FCL_dims[0]),
        #                                           nn.ReLU(),
        #                                           nn.Linear(FCL_dims[0], FCL_dims[1]),
        #                                           nn.BatchNorm1d(FCL_dims[1]),
        #                                           nn.ReLU(), 
        #                                           nn.Linear(FCL_dims[1], N_gripper_action))
        # else:
        #     self.gripper_actions = None

        self.gripper_actions = None

        #initialise max action range
        self.max_action = torch.tensor(max_action).to(self.device)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir

        #check if dir exists 
        self.name = name
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.optimiser    = optim.Adam(self.parameters(), lr = lr)
        self.bc_optimiser = optim.Adam(self.parameters(), lr = lr)

        #used for preventing 0 value
        self.sm_c = 1e-6                 

        self.to(self.device)

    def forward(self, state):

        #move to correct device
        state = state.to(self.device)

        #encode state
        x = self.encoder(state)

        #compute normal distribution mean
        mean = self.mean(x)

        #compute normal distribution std
        std = self.std(x)

        #clamp the std within range
        std = torch.clamp(std, 
                          min = self.sm_c, 
                          max = 1.)
        
        # #compute gripper action type
        # if self.gripper_actions is not None:
        #     gripper_action_probs = self.gripper_actions(x)
        # else:
        #     gripper_action_probs = None

        # return mean, std, gripper_action_probs

        return mean, std

    def get_actions(self, 
                    state, 
                    is_reparametrerise = True):

        # mean, std, gripper_action_prob = self.forward(state)

        mean, std = self.forward(state)

        normal = Normal(mean, std)

        if is_reparametrerise:
            z = normal.rsample()
        else:
            z = normal.sample()

        normalised_action = torch.tanh(z)
        action = normalised_action*self.max_action

        # if gripper_action_prob is not None:
        #     gripper_action = Categorical(torch.softmax(gripper_action_prob, axis = 1)).sample()
        # else:
        #     gripper_action = torch.FloatTensor([constants.CLOSE_GRIPPER]).unsqueeze(0)

        # return action, normalised_action, gripper_action, z, normal, gripper_action_prob

        return action, normalised_action, z, normal

    def compute_log_prob(self, normal, a, z):


        log_probs  = normal.log_prob(z) - torch.log(1-a.pow(2) + self.sm_c)
        log_probs  = log_probs.sum(-1, keepdim = True)/a.shape[-1]

        return log_probs.float()

    def save_checkpoint(self, is_best = False, name = None):
        if is_best:
            torch.save(self.state_dict(), self.checkpt_file)
        elif not is_best and name is not None:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_' + name + '_sac_checkpt'))
            torch.save(self.state_dict(), checkpt_file)
        else:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_sac_checkpt'))
            torch.save(self.state_dict(), checkpt_file)

    def load_checkpoint(self, is_best = False):
        if is_best:
            self.load_state_dict(torch.load(self.checkpt_file))
        else:
            file = os.path.abspath(os.path.join(self.checkpt_dir, self.name + '_sac_checkpt'))
            self.load_state_dict(torch.load(file))

class QNet(nn.Module):
    def __init__(self,                                                         
                 N_input_channels = 2, #encoder input channel
                 input_dims = [1024], #FLC input dimension
                 FCL_dims = [512, 256], #FLC network dimension
                 N_action = 3, #action dimension
                 lr = 1e-4, #learning rate
                 epsilon = 0.1, #epsilon for action explorations
                 name = 'QNet', #define the network name
                 checkpt_dir = 'logs/models'): #define checkpoint directory
    
        super(QNet, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        #initialise input and output dimention
        self.N_action = N_action

        #initialise vision encoder
        #[(W-K+2P)/S] + 1
        self.encoder = nn.Sequential(
            #W: 128 => 64
            nn.Conv2d(N_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #W: 64 => 32 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #W: 32 => 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #W: 16 => 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #W: 8 => 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #W: 4 => 2
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #W: 2 => 1
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
        )

        #initialise outputs
        self.Qx = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                nn.BatchNorm1d(FCL_dims[0]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[0], FCL_dims[1]),
                                nn.BatchNorm1d(FCL_dims[1]),
                                nn.ReLU(), 
                                nn.Linear(FCL_dims[1], N_action))
        
        self.Qy = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                nn.BatchNorm1d(FCL_dims[0]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[0], FCL_dims[1]),
                                nn.BatchNorm1d(FCL_dims[1]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[1], N_action))
        
        self.Qz = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                nn.BatchNorm1d(FCL_dims[0]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[0], FCL_dims[1]),
                                nn.BatchNorm1d(FCL_dims[1]),
                                nn.ReLU(),
                                nn.Linear(FCL_dims[1], N_action))

        self.Qyaw = nn.Sequential(nn.Linear(input_dims[0], FCL_dims[0]),
                                  nn.BatchNorm1d(FCL_dims[0]),
                                  nn.ReLU(),
                                  nn.Linear(FCL_dims[0], FCL_dims[1]),
                                  nn.BatchNorm1d(FCL_dims[1]),
                                  nn.ReLU(),
                                  nn.Linear(FCL_dims[1], N_action))

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir

        #check if dir exists 
        self.name = name
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name))

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        #used for preventing 0 value
        self.sm_c = 1e-6                 

        self.epsilon = epsilon

        self.to(self.device)

    def forward(self, state):

        #move to correct device
        state = state.to(self.device)

        #encode state
        x = self.encoder(state)

        #compute Qx
        Qx = self.Qx(x)

        #compute Qy
        Qy = self.Qy(x)

        #compute Qz
        Qz = self.Qz(x)

        #compute Qyaw
        Qyaw = self.Qyaw(x)

        return Qx, Qy, Qz, Qyaw
    
    def get_actions(self, state, take_max = False):
        
        Qx, Qy, Qz, Qyaw = self.forward(state)
        print(f"Qx: {Qx}")
        print(f"Qy: {Qy}")
        print(f"Qz: {Qz}")
        print(f"Qyaw: {Qyaw}")

        prob = random.random()

        if take_max or prob > self.epsilon:
            x_ind = torch.argmax(Qx, dim=1).item()
            y_ind = torch.argmax(Qy, dim=1).item()
            z_ind = torch.argmax(Qz, dim=1).item()
            yaw_ind = torch.argmax(Qyaw, dim=1).item()

            print("[MAX OUTPUT]")
            print(f"[Q net] x: {x_ind}, y: {y_ind}, z: {z_ind}, yaw_ind: {yaw_ind}")
        else:
            x_ind = random.randrange(self.N_action)
            y_ind = random.randrange(self.N_action)
            z_ind = random.randrange(self.N_action)
            yaw_ind = random.randrange(self.N_action)

            print("[RANDOM OUTPUT]")
            print(f"[Q net] x: {x_ind}, y: {y_ind}, z: {z_ind}, yaw_ind: {yaw_ind}")
            
        return Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind
    
    def save_checkpoint(self, is_best = False, name = None):
        if is_best:
            torch.save(self.state_dict(), self.checkpt_file)
        elif not is_best and name is not None:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_' + name + '_checkpt'))
            torch.save(self.state_dict(), checkpt_file)
        else:
            checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir,  self.name + '_checkpt'))
            torch.save(self.state_dict(), checkpt_file)

    def load_checkpoint(self, is_best = False):
        if is_best:
            self.load_state_dict(torch.load(self.checkpt_file))
        else:
            file = os.path.abspath(os.path.join(self.checkpt_dir, self.name + '_checkpt'))
            self.load_state_dict(torch.load(file))