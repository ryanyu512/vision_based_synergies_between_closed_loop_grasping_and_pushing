import time

#data processing
import copy
import numpy as np

#for visualisation
from PIL import Image
import matplotlib.pyplot as plt

#self-customed library
import utils
import constants

from env import Env
from buffer import BufferReplay
from network import Actor, Critic, HDL_Net

#DL related
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

class DemoDataset(Dataset):
    def __init__(self, states, actions, q_values):
        self.states   = states
        self.actions  = actions
        self.q_values = q_values

    def __len__(self):
        return len(self.grasp_states)

    def __getitem__(self, index):
        state   = self.states[index]
        action  = self.actions[index]
        q_value = self.q_values[index]

        return state, action, q_value

class DemoDatasetHldNet(Dataset):
    def __init__(self, states, action_types):
        self.states       = states
        self.action_types = action_types

    def __len__(self):
        return len(self.grasp_states)

    def __getitem__(self, index):
        state        = self.states[index]
        action_type  = self.action_types[index]

        return state, action_type
    

#TODO [NOTE 21 AUG 2024]: set learning rate through agent    
#TODO [NOTE 21 AUG 2024]: add evaluation for behaviour cloning
class Agent():

    def __init__(self, 
                 env,
                 N_action = 4,
                 N_gripper_action = 2,
                 N_batch  = 64, 
                 alpha    = 0.2,
                 tau      = 0.05,
                 gamma    = 0.95,
                 max_memory_size    = 50000,
                 save_step_interval = 10,
                 is_debug = False):

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"device: {self.device}")

        #initialise env
        self.env      = env
        print("[SUCCESS] initialise environment")

        #initialise action and action_type dimension
        self.N_action = N_action
        self.N_gripper_action = N_gripper_action

        #initialise high level network
        self.hld_net       = HDL_Net(N_input_channels = 1)

        #initialise grasp actor
        self.grasp_actor   = Actor(N_input_channels = 2) #depth image + gripper status
        #initialise grasp critic network 1
        self.grasp_critic1 = Critic(N_input_channels = 8) #depth image + gripper status + dx + dy + dz + dyaw + gripper action
        #initialise grasp ciritc network 2
        self.grasp_critic2 = Critic(N_input_channels = 8) #depth image + gripper status + dx + dy + dz + dyaw + gripper action
        #initialise grasp critic network target 1
        self.grasp_critic1_target = Critic(N_input_channels = 8) #depth image + gripper status + dx + dy + dz + dyaw + gripper action
        #initialise grasp critic network target 2
        self.grasp_critic2_target = Critic(N_input_channels = 8) #depth image + gripper status + dx + dy + dz + dyaw + gripper action

        #initialise grasp actor
        self.push_actor   = Actor(N_input_channels = 2, action_type="push") #depth image + yaw angle 
        #initialise grasp critic network 1
        self.push_critic1 = Critic(N_input_channels = 6) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp ciritc network 2
        self.push_critic2 = Critic(N_input_channels = 6) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp critic network target 1
        self.push_critic1_target = Critic(N_input_channels = 6) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp critic network target 2
        self.push_critic2_target = Critic(N_input_channels = 6) #depth image + yaw angle + dx + dy + dz + dyaw 

        #soft update to make critic target align with critic
        self.soft_update()
        print("[SUCCESS] initialise networks")

        #initialise buffer replay
        self.buffer_replay = BufferReplay(max_memory_size = max_memory_size)
        print("[SUCCESS] initialise memory buffer")
        #initialise batch size
        self.N_batch = N_batch
        #initalise small constant to prevent zero value
        self.sm_c    = 1e-6
        #initialise temperature factor
        self.alpha   = alpha
        #initialise discount factor
        self.gamma   = gamma
        #initialise soft update factor
        self.tau     = tau
        #initialise history 
        self.r_hist    = []
        self.step_hist = []

        #initialise if debug
        self.is_debug = is_debug

        #initialise save interval
        self.save_step_interval = save_step_interval

    def get_raw_data(self, action_type):

        _, depth_img         = self.env.get_rgbd_data()
        _, gripper_tip_ori   = self.env.get_obj_pose(self.env.gripper_tip_handle, 
                                                     self.env.sim.handle_world)
        if action_type == constants.PUSH:
            yaw_ang        = gripper_tip_ori[2]
            gripper_status = None
        else:
            yaw_ang        = None
            gripper_status = self.env.gripper_status


        return depth_img, gripper_status, yaw_ang

    def preprocess_state(self, depth_img, gripper_state, yaw_ang, is_grasp = True):
        
        #copy image
        in_depth_img = copy.copy(depth_img)

        #check nan
        in_depth_img[np.isnan(in_depth_img)] = 0
        #check negative value
        in_depth_img[in_depth_img < 0] = 0

        #scale depth image into range 0 - 1
        in_depth_img = (in_depth_img.astype(np.float32) - self.env.near_clip_plane)/(self.env.far_clip_plane - self.env.near_clip_plane)
        in_depth_img = np.expand_dims(in_depth_img, axis = 2)

        state = copy.copy(in_depth_img)

        #turn gripper state into image
        if gripper_state is not None and not np.isnan(gripper_state):
            gripper_state_img  = np.ones_like(in_depth_img)*gripper_state
            #scale gripper_state_img into range 0 - 1
            gripper_state_img = gripper_state_img.astype(np.float32)/constants.GRIPPER_NON_CLOSE_NON_OPEN #GRIPPER_NON_CLOSE_NON_OPEN = 2, largest value
            #compute states
            state  = np.concatenate((state, gripper_state_img), axis = -1)

        #turn yaw ang into image
        if yaw_ang is not None and not np.isnan(yaw_ang):
            #wrap the yaw ang into range of -180 to + 180
            yaw_img = np.ones_like(in_depth_img)*utils.wrap_ang(yaw_ang)
            #scale yaw image into range -1 to +1
            yaw_img = yaw_img.astype(np.float32)/np.math.pi
            #compute states
            state  = np.concatenate((state, yaw_img), axis = -1)

        state = torch.from_numpy(state.astype(np.float32)).permute(2,0,1)

        return state
    
    def turn_action2image(self, move_action, gripper_action):
        
        resol = self.env.depth_resol
        dx_img     = np.ones((1, resol[0], resol[1]))*move_action[0]/5.
        dy_img     = np.ones((1, resol[0], resol[1]))*move_action[1]/5.
        dz_img     = np.ones((1, resol[0], resol[1]))*move_action[2]/5.
        dyaw_image = np.ones((1, resol[0], resol[1]))*move_action[3]/np.deg2rad(30.)
        gc_img     = np.ones((1, resol[0], resol[1]))*np.argmax(gripper_action)

        action_img = np.concatenate((dx_img, dy_img, dz_img, dyaw_image, gc_img), axis = 0)
        action_img = torch.FloatTensor(action_img).unsqueeze(0).to(self.device)            

        return action_img

    def soft_update(self, tau = None):
        
        if tau is None:
            tau = 1.
        else:
            tau = self.tau

        # Grasp Critic networks
        for (target_param, param) in zip(self.grasp_critic1_target.parameters(), self.grasp_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for (target_param, param) in zip(self.grasp_critic2_target.parameters(), self.grasp_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Push Critic networks
        for (target_param, param) in zip(self.push_critic1_target.parameters(), self.push_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for (target_param, param) in zip(self.push_critic2_target.parameters(), self.push_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update BatchNorm running statistics
        for target_layer, source_layer in zip(self.grasp_critic1_target.encoder, self.grasp_critic1.encoder):
            if isinstance(target_layer, torch.nn.BatchNorm2d):
                # print("grasp_critic1 2d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        for target_layer, source_layer in zip(self.grasp_critic2_target.encoder, self.grasp_critic2.encoder):
            if isinstance(target_layer, torch.nn.BatchNorm2d):
                # print("grasp_critic2 2d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        for target_layer, source_layer in zip(self.push_critic1_target.encoder, self.push_critic1.encoder):
            if isinstance(target_layer, torch.nn.BatchNorm2d):
                # print("push_critic1 2d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        for target_layer, source_layer in zip(self.push_critic2_target.encoder, self.push_critic2.encoder):
            if isinstance(target_layer, torch.nn.BatchNorm2d):
                # print("push_critic2 2d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        # Update BatchNorm1d running statistics
        for target_layer, source_layer in zip(self.grasp_critic1_target.fc, self.grasp_critic1.fc):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                # print("grasp_critic1 1d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        for target_layer, source_layer in zip(self.grasp_critic2_target.fc, self.grasp_critic2.fc):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                # print("grasp_critic2 1d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        for target_layer, source_layer in zip(self.push_critic1_target.fc, self.push_critic1.fc):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                # print("push_critic1 1d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

        for target_layer, source_layer in zip(self.push_critic2_target.fc, self.push_critic2.fc):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                # print("push_critic2 1d, running_mean, running_var")
                target_layer.running_mean = tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * source_layer.running_var + (1 - tau) * target_layer.running_var

    def get_train_test_dataloader_hld_net(self, exp, train_ratio = 0.8):

        #            0,            1,           2
        # action_index, depth_states, action_type

        #split experience into training (80%) and valuation (20%)
        indices       = np.arange(len(exp[0]))
        train_size    = int(train_ratio*len(indices))
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        #turn experience into compatible inputs to grasp network
        states        = []
        action_types  = []

        for i in range(len(indices)):
            #get state
            state = self.preprocess_state(depth_img     = exp[1][i], 
                                          gripper_state = None, 
                                          yaw_ang       = None)
            
            states.append(state)

            action_type = copy.copy(exp[2][i])
            action_types.append(action_type)

        states       = np.array(states)
        action_types = np.array(action_types)

        states       = torch.FloatTensor(states)
        action_types = torch.FloatTensor(action_types)

        dataset  = DemoDatasetHldNet(states, action_types)

        #create subset
        train_subset = Subset(dataset, train_indices)
        test_subset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size = self.N_batch, shuffle = True, 
                                  drop_last=True if train_size > self.N_batch else False)
        test_loader  = DataLoader( test_subset, batch_size = self.N_batch, shuffle = False)

        return train_loader, test_loader



    def get_train_test_dataloader(self,
                                  exp, 
                                  train_ratio = 0.8, 
                                  is_grasp = True):

        #            0,          1,            2,              3,          4,       5,               6,       7,                 8,     9,         10,         11
        # action_index, priorities, depth_states, gripper_states, yaw_states, actions, gripper_actions, rewards, next_depth_states, dones, predict_qs, labeled_qs

        #split experience into training (80%) and valuation (20%)
        indices       = np.arange(len(exp[0]))
        train_size    = int(train_ratio*len(indices))
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        #turn experience into compatible inputs to grasp network
        states   = []
        actions  = []
        q_values = []
        for i in range(len(indices)):
            #get state
            state = self.preprocess_state(depth_img     = exp[2][i], 
                                          gripper_state = exp[3][i], 
                                          yaw_ang       = exp[4][i])
            
            states.append(state)

            #get action
            if is_grasp:
                action = np.hstack((exp[5][i], exp[6][i]))
            else:
                action = exp[5][i]

            actions.append(action)

            #get q values
            q_value = copy.copy(exp[11][i])
            q_values.append(q_value)

        states   = np.array(states)
        actions  = np.array(actions)
        q_values = np.array(q_values)

        states   = torch.FloatTensor(states)
        actions  = torch.FloatTensor(actions)
        q_values = torch.FloatTensor(q_values)

        dataset  = DemoDataset(states, actions, q_values)

        #create subset
        train_subset = Subset(dataset, train_indices)
        test_subset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size = self.N_batch, shuffle = True, 
                                  drop_last=True if train_size > self.N_batch else False)
        test_loader  = DataLoader( test_subset, batch_size = self.N_batch, shuffle = False)

        return train_loader, test_loader

    def behaviour_cloning_hld(self, train_loader, hld_net, num_epochs=500, lr=1e-4):

        #set network in training mode
        hld_net.train()

        # Initialize optimizers for both critics and actor
        hld_optimizer = optim.Adam(hld_net.parameters(), lr=lr)
        
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            mean_hld_loss   = 0.

            data_cnt = 0
            for state, target_action_type in train_loader:

                state              = state.to(self.device)
                target_action_type = target_action_type.to(self.device)
                target_action_type = torch.nn.functional.one_hot(target_action_type.long(), 
                                                                 num_classes = self.N_gripper_action).float()

                # Forward pass through both critics
                action_type = hld_net(state)

                # Compute loss for both critics
                hld_net_loss = ce_loss(action_type, target_action_type)

                # update hld-net
                hld_optimizer.zero_grad()
                hld_net_loss.backward()
                hld_optimizer.step()

                mean_hld_loss   += hld_net_loss.item()
            
                data_cnt += state.shape[0]
        
            mean_hld_loss   /= data_cnt

            print(f"Epoch {epoch+1}/{num_epochs}, hld Loss: {mean_hld_loss :.6f}")

    def behaviour_cloning_eval_hld(self, test_loader, hld_net):
        
        #set network in evaluation mode
        hld_net.eval()
        
        ce_loss = nn.CrossEntropyLoss()


        mean_hld_loss   = 0.
        data_cnt = 0

        with torch.no_grad():
            for state, target_action_type in test_loader:

                state              = state.to(self.device)
                target_action_type = target_action_type.to(self.device)
                target_action_type = torch.nn.functional.one_hot(target_action_type.long(), 
                                                                 num_classes = self.N_gripper_action).float()

                # Forward pass through both critics
                action_type = hld_net(state)

                # Compute loss for both critics
                hld_net_loss = ce_loss(action_type, target_action_type)

                mean_hld_loss   += hld_net_loss.item()
            
                data_cnt += state.shape[0]
        
            mean_hld_loss   /= data_cnt

        print(f"hld Loss: {mean_hld_loss :.6f}")


    def behaviour_cloning(self, train_loader, critic1, critic2, actor, num_epochs=500, is_grasp = True, lr=1e-4):

        #set networks to training mode
        actor.train()
        critic1.train()
        critic2.train()

        # Initialize optimizers for both critics and actor
        critic1_optimizer = optim.Adam(critic1.parameters(), lr=lr)
        critic2_optimizer = optim.Adam(critic2.parameters(), lr=lr)
        actor_optimizer   = optim.Adam(actor.parameters(), lr=lr)
        
        mse_loss = nn.MSELoss()
        ce_loss  = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            mean_actor_loss   = 0.
            mean_critic1_loss = 0.
            mean_critic2_loss = 0.

            data_cnt = 0
            for state, target_action, target_q_value in train_loader:

                state          = state.to(self.device)
                target_action  = target_action.to(self.device)
                target_q_value = target_q_value.to(self.device)

                #compute normalisation factor
                normalise_factor    = torch.tensor([0.05, 0.05, 0.05, np.deg2rad(30.), 1., 1.]).to(self.device) if is_grasp else torch.tensor([0.05, 0.05, 0.05, np.deg2rad(30.)]).to(self.device)

                #compute action state
                normalise_target_action = target_action/normalise_factor.view(1, normalise_factor.shape[0])
            
                #transform vector into image format
                target_action_img   = normalise_target_action.view(target_action.shape[0], target_action.shape[1], 1, 1)*torch.ones((target_action.shape[0], target_action.shape[1], 128, 128)).to(self.device)

                #compute action state
                action_state = torch.concatenate((state, target_action_img), axis = 1).float()

                # Forward pass through both critics
                q1_pred = critic1(action_state)
                q2_pred = critic2(action_state)

                target_q_value = target_q_value.unsqueeze(1)

                # Use the actor to get the predicted actions
                actions, gripper_action, z, normal, gripper_action_probs = actor.get_actions(state)

                # Compute loss for both critics
                critic1_loss = mse_loss(q1_pred, target_q_value)
                critic2_loss = mse_loss(q2_pred, target_q_value)

                # compute actor loss 
                actor_loss = mse_loss(actions.float(), target_action[:,0:4])
                if is_grasp:
                    actor_loss += ce_loss(gripper_action_probs, target_action[:,4:])

                # update critic 1
                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                # update critic 2
                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # update actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                mean_actor_loss   += actor_loss.item()
                mean_critic1_loss += critic1_loss.item()
                mean_critic2_loss += critic2_loss.item()
            
                data_cnt += state.shape[0]
        
            mean_actor_loss   /= data_cnt
            mean_critic1_loss /= data_cnt
            mean_critic2_loss /= data_cnt

            print(f"Epoch {epoch+1}/{num_epochs}, Critic1 Loss: {mean_critic1_loss :.6f}, Critic2 Loss: {mean_critic2_loss :.6f}, Actor Loss: {mean_actor_loss :.6f}")

    def behaviour_cloning_eval(self, test_loader,  critic1, critic2, actor, is_grasp = True):

        #set to evaluation mode
        actor.eval()
        critic1.eval()
        critic2.eval()

        total_loss = 0.0

        mse_loss = nn.MSELoss()
        ce_loss  = nn.CrossEntropyLoss()

        mean_actor_loss   = 0.
        mean_critic1_loss = 0.
        mean_critic2_loss = 0.

        data_cnt = 0

        with torch.no_grad():
            for state, target_action, target_q_value  in test_loader:

                state          = state.to(self.device)
                target_action  = target_action.to(self.device)
                target_q_value = target_q_value.to(self.device)

                #compute normalisation factor
                normalise_factor    = torch.tensor([0.05, 0.05, 0.05, np.deg2rad(30.), 1., 1.]).to(self.device) if is_grasp else torch.tensor([0.05, 0.05, 0.05, np.deg2rad(30.)]).to(self.device)

                #compute action state
                normalise_target_action = target_action/normalise_factor.view(1, normalise_factor.shape[0])
            
                #transform vector into image format
                target_action_img   = normalise_target_action.view(target_action.shape[0], target_action.shape[1], 1, 1)*torch.ones((target_action.shape[0], target_action.shape[1], 128, 128)).to(self.device)

                #compute action state
                action_state = torch.concatenate((state, target_action_img), axis = 1).float()

                # Forward pass through both critics
                q1_pred = critic1(action_state)
                q2_pred = critic2(action_state)

                target_q_value = target_q_value.unsqueeze(1)

                # Use the actor to get the predicted actions
                actions, gripper_action, z, normal, gripper_action_probs = actor.get_actions(state)

                # Compute loss for both critics
                critic1_loss = mse_loss(q1_pred, target_q_value)
                critic2_loss = mse_loss(q2_pred, target_q_value)

                # compute actor loss 
                actor_loss = mse_loss(actions.float(), target_action[:,0:4])
                if is_grasp:
                    actor_loss += ce_loss(gripper_action_probs, target_action[:,4:])

                data_cnt += state.shape[0]
                mean_actor_loss   += actor_loss.item()
                mean_critic1_loss += critic1_loss.item()
                mean_critic2_loss += critic2_loss.item()
            

            mean_actor_loss   /= data_cnt
            mean_critic1_loss /= data_cnt
            mean_critic2_loss /= data_cnt

            print(f"Critic1 Loss: {mean_critic1_loss :.6f}, Critic2 Loss: {mean_critic2_loss :.6f}, Actor Loss: {mean_actor_loss :.6f}")


    def gather_guidance_experience(self,
                                   max_episode = 1,
                                   is_debug = True):

        #start trainiing/evaluation loop
        for episode in range(max_episode):

            #initialise episode data
            step = 0
            ep_r = 0.
            done = False
            action_type = None
            is_success_grasp = False
            while not done and not self.buffer_replay.is_full:
                
                print(f"==== episode: {episode} step: {step} ====")

                #action selection
                delta_moves = self.env.grasp_guidance_generation()
                
                #the 2nd condition prevent stucking in impossible grasping action
                if len(delta_moves) == 0:
                    delta_moves = self.env.push_guidance_generation()
                    action_type = constants.PUSH
                else:
                    action_type = constants.GRASP

                if len(delta_moves) == 0:
                    done     = True
            
                depth_states      = np.zeros((len(delta_moves), self.env.depth_resol[0], self.env.depth_resol[1]))
                gripper_states    = np.zeros(len(delta_moves))
                yaw_states        = np.zeros(len(delta_moves))
                actions           = np.zeros((len(delta_moves), self.N_action))
                gripper_actions   = np.zeros((len(delta_moves), self.N_gripper_action))
                action_types      = np.zeros(len(delta_moves))
                rewards           = np.zeros(len(delta_moves))
                next_depth_states = np.zeros((len(delta_moves), self.env.depth_resol[0], self.env.depth_resol[1]))
                dones             = np.zeros(len(delta_moves)).astype(bool)
                labeled_qs        = np.zeros(len(delta_moves))

                for i in range(len(delta_moves)):

                    #get raw data
                    depth_img, gripper_state, yaw_state = self.get_raw_data(action_type)

                    #get state
                    state = self.preprocess_state(depth_img     = depth_img, 
                                                  gripper_state = gripper_state, 
                                                  yaw_ang       = yaw_state)
                    
                    #action selection
                    move   = np.array(delta_moves[i])
                    n_move = np.array(delta_moves[i+1] if i+1 < len(delta_moves) else [0,0,0,0,move[-2],move[-1]])

                    #select action
                    action,   action_type,  gripper_action             = np.array(move[0:self.N_action]), action_type,   np.array(move[-2:])
                    next_action, next_action_type, next_gripper_action = np.array(n_move[0:self.N_action]), action_type, np.array(n_move[-2:])

                    gripper_action = torch.FloatTensor(gripper_action).unsqueeze(0).to(self.device)
                    action  = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    
                    next_gripper_action = torch.FloatTensor(next_gripper_action).unsqueeze(0).to(self.device)
                    next_action         = torch.FloatTensor(next_action).unsqueeze(0).to(self.device)

                    #step
                    reward, is_success_grasp = self.env.step(action_type, 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][0:3], 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][3], 
                                                             is_open_gripper = True if move[4] == 1 else False)

                    #get raw data
                    next_depth_img, next_gripper_state, next_yaw_state = self.get_raw_data(action_type)

                    print(f"[STEP]: {step} [ACTION TYPE]: {action_type} [REWARD]: {reward}") if is_debug else None
                    print(f"[MOVE]: {action.to(torch.device('cpu')).detach().numpy()[0]}") if is_debug else None

                    #get state
                    next_state = self.preprocess_state(depth_img     = next_depth_img, 
                                                       gripper_state = next_gripper_state, 
                                                       yaw_ang       = next_yaw_state)

                    done = False if self.env.N_pickable_obj > 0 else True
                    
                    # store experience in one action
                    s_time = time.time()
                    depth_states[i]      = depth_img
                    gripper_states[i]    = gripper_state
                    yaw_states[i]        = yaw_state
                    actions[i]           = action.to(torch.device('cpu')).detach().numpy()[0]
                    gripper_actions[i]   = gripper_action.to(torch.device('cpu')).detach().numpy()[0]
                    action_types[i]      = action_type
                    rewards[i]           = reward
                    next_depth_states[i] = next_depth_img
                    dones[i]             = True if (i == len(delta_moves) - 1 or done) else False

                    #update history
                    ep_r += reward
                    step += 1

                    #check if done
                    if done:
                        self.env.reset(reset_obj = True)
                        print("[SUCCESS] finish one episode")
                        self.r_hist.append(ep_r)
                        self.step_hist.append(step)         
                        break 
                    else:

                        #return home position if grasp successfully
                        if is_success_grasp:
                            print("[SUCCESS] grasp an object")
                            self.env.return_home()
                            print("[SUCCESS] return home position")

                        #check if out of working space
                        elif self.env.is_out_of_working_space:
                            print("[WARN] out of working space")
                            self.env.reset(reset_obj = False)

                        #check if action executable
                        elif not self.env.can_execute_action:
                            print("[WARN] action is not executable")
                            self.env.reset(reset_obj = False)

                        #check if collision to ground
                        elif self.env.is_collision_to_ground:
                            print("[WARN] collision to ground")
                            self.env.reset(reset_obj = False)

                #update estimated Q values
                if np.sum(rewards) > 0: #only save successful experience
                    labeled_qs[-1] = rewards[-1]
                    print(f'labeled_qs {len(labeled_qs) - 1}: {labeled_qs[-1]}')
                    for i in range(len(labeled_qs) - 2, -1, -1):
                        labeled_qs[i] = rewards[i] + self.gamma*labeled_qs[i+1]
                        print(f'labeled_qs {i}: {labeled_qs[i]}')

                    #store transition experience          
                    for i in range(len(delta_moves)): 
                        self.buffer_replay.store_transition(True if i == 0 else False, #store if at home pos
                                                            depth_states[i], 
                                                            gripper_states[i],
                                                            yaw_states[i],
                                                            actions[i], 
                                                            gripper_actions[i], 
                                                            action_types[i],
                                                            rewards[i], 
                                                            next_depth_states[i], 
                                                            dones[i], 
                                                            labeled_qs[i], 
                                                            labeled_qs[i]) 
                    print("[SUCCESS] store successful demonstration")
                else:
                    print("[SUCCESS] discard fail demonstration")

            if self.buffer_replay.is_full:
                self.buffer_replay.save_buffer()
                break
            
    def save_models(self):
        #save grasp network
        self.grasp_actor.save_checkpoint()
        self.grasp_critic1.save_checkpoint()
        self.grasp_critic2.save_checkpoint()
        self.grasp_critic1_target.save_checkpoint()
        self.grasp_critic2_target.save_checkpoint()

        #save push network
        self.push_actor.save_checkpoint()
        self.push_critic1.save_checkpoint()
        self.push_critic2.save_checkpoint()
        self.push_critic1_target.save_checkpoint()
        self.push_critic2_target.save_checkpoint()

        print('[SUCCESS] save_models')
         
    def online(self):

        if self.buffer_replay.memory_cntr < self.N_batch:
            return

        # batch, s, a, a_type, r, ns, done = self.buffer_replay.sample_buffer(self.N_batch)
        
        # r.shape    = (r.shape[0], 1)
        # done.shape = (done.shape[0], 1)

        # s      = torch.FloatTensor(s).to(self.device)
        # a      = torch.FloatTensor(a).to(self.device)
        # a_type_onehot = torch.FloatTensor(a_type).to(self.device)
        # ns     = torch.FloatTensor(ns).to(self.device)
        # r      = torch.FloatTensor(r).to(self.device)
        # done   = torch.FloatTensor(done).to(self.device)

        # #update critic
        # with torch.no_grad():
        #     #compute next action, next action type, next action normals, next action type probability
        #     na, na_type, nz, n_normal, na_type_probs = self.actor.get_actions(ns)
            
        #     #compute one hot vector
        #     na_type_onehot = torch.nn.functional.one_hot(na_type.long(), 
        #                                                  num_classes = self.Na_type).float()
        #     #compute log probability
        #     nlog_probs = self.actor.compute_log_prob(normal = n_normal, a = na, z = nz)

        #     #compute next q value
        #     nq1 = self.critic1_target(ns, na, na_type_onehot)
        #     nq2 = self.critic2_target(ns, na, na_type_onehot)
        #     nq  = (torch.min(nq1, nq2) - self.alpha*nlog_probs)
        #     q_target = (r + (1-done)*self.gamma*nq)

        # q1 = self.critic1(s, a, a_type_onehot)
        # q2 = self.critic2(s, a, a_type_onehot)

        # # print(f'[online] q_target.shape: {q_target.shape}')
        # c1_loss = (0.5*torch.nn.MSELoss()(q1, q_target))
        # c2_loss = (0.5*torch.nn.MSELoss()(q2, q_target))
        # c_loss  = (c1_loss + c2_loss)

        # self.critic1.optimiser.zero_grad()
        # self.critic2.optimiser.zero_grad()
        # c_loss.backward()
        # self.critic1.optimiser.step()
        # self.critic2.optimiser.step()

        # #update actor
        # a, a_type, z, normal, atype_probs = self.actor.get_actions(s)
        # log_probs = self.actor.compute_log_prob(normal = normal, a = a, z = z)
        # min_q = torch.min(
        #     self.critic1(s, a, torch.nn.functional.one_hot(a_type.long(), 
        #                                                    self.Na_type).float()),
        #     self.critic2(s, a, torch.nn.functional.one_hot(a_type.long(), 
        #                                                    self.Na_type).float())
        # )
        # a_loss = (self.alpha*log_probs - min_q).mean()

        # self.actor.zero_grad()
        # a_loss.backward()
        # self.actor.optimiser.step()

        # # Soft update target critic networks
        # self.soft_update(self.tau)

        # #TODO update the buffer priority
        # with torch.no_grad():
        #     #compute next action, next action type, next action normals, next action type probability
        #     a, a_type, z, normal, a_type_probs = self.actor.get_actions(ns)
            
        #     #compute one hot vector
        #     a_type_onehot = torch.nn.functional.one_hot(a_type.long(), 
        #                                                  num_classes = self.Na_type).float()

        #     #compute next q value
        #     q1 = self.critic1_target(ns, na, na_type_onehot)
        #     q2 = self.critic2_target(ns, na, na_type_onehot)
        #     q  = (q1 + q2)*0.5

        #     q = q.to(torch.device('cpu')).detach().numpy()
        #     q.shape = (q.shape[0],);
        #     self.buffer_replay.update_buffer(batch, q)

    # def interact(self, 
    #              is_train = True, 
    #              max_episode = 1,
    #              is_debug = True):
        
    #     #start trainiing/evaluation loop
    #     for episode in range(max_episode) if is_train else 1:

    #         #initialise episode data
    #         step = 0
    #         ep_r = 0.

    #         while True:

    #             #get raw data
    #             color_img, depth_img = self.env.get_rgbd_data()

    #             #preprocess raw data
    #             in_color_img, in_depth_img = self.preprocess_input(color_img = color_img, 
    #                                                                depth_img = depth_img)

    #             #add the extra dimension in the 1st dimension
    #             in_color_img = in_color_img.unsqueeze(0)
    #             in_depth_img = in_depth_img.unsqueeze(0)

    #             #get state
    #             s = self.encoder.get_latent_vectors(inputs = in_depth_img)

    #             #action selection
    #             a, a_type, z, normal, a_type_probs = self.actor.get_actions(s)

    #             #compute one hot vector
    #             a_type_onehot = torch.nn.functional.one_hot(a_type.long(), 
    #                                                         num_classes = self.Na_type).float()

    #             #step
    #             #TODO:test the step function
    #             next_color_img, next_depth_img, r, is_success_grasp = self.env.step(a_type.to(torch.device('cpu')).detach().numpy()[0], 
    #                                                                                 a.to(torch.device('cpu')).detach().numpy()[0][0:3], 
    #                                                                                 a.to(torch.device('cpu')).detach().numpy()[0][3])

    #             print(f"[STEP]: {step} [ACTION TYPE]: {a_type} [REWARD]: {r}") if is_debug else None
    #             print(f"[MOVE]: {a.to(torch.device('cpu')).detach().numpy()[0]}") if is_debug else None

    #             #preprocess raw data
    #             next_in_color_img, next_in_depth_img = self.preprocess_input(color_img = next_color_img, 
    #                                                                          depth_img = next_depth_img)

    #             next_in_color_img = next_in_color_img.unsqueeze(0)
    #             next_in_depth_img = next_in_depth_img.unsqueeze(0)

    #             #convert next color img and next depth img into next state
    #             ns = self.encoder.get_latent_vectors(inputs = next_depth_img)

    #             #check if terminate this episode
    #             #TODO: test the function
    #             done = False if self.env.N_pickable_obj > 0 else True

            # #compute predict q value and labeled q value
            # with torch.no_grad(): 

            #     action_img = self.turn_action2image(action.to(torch.device('cpu')).detach().numpy()[0], 
            #                                         gripper_action.to(torch.device('cpu')).detach().numpy()[0])
                

            #     state_action = torch.cat([state, action_img], dim = 1).float()

            #     if action_type == GRASP:
            #         q1 = self.grasp_critic1(state_action)
            #         q2 = self.grasp_critic2(state_action)
            #     else:
            #         q1 = self.push_critic1(state_action)
            #         q2 = self.push_critic2(state_action)

            #     q  = torch.min(q1, q2)

            #     # #compute next action based on next action type
            #     # if action_type == GRASP:
            #     #     na, nga_type, nz, n_normal, na_type_probs = self.grasp_actor.get_actions(next_state)
            #     # else:
            #     #     na, nga_type, nz, n_normal, na_type_probs = self.push_actor.get_actions(next_state)

            #     next_action_img = self.turn_action2image(next_action.to(torch.device('cpu')).detach().numpy()[0], 
            #                                              next_gripper_action.to(torch.device('cpu')).detach().numpy()[0])

            #     if action_type == GRASP:
            #         nq1 = self.grasp_critic1(state_action)
            #         nq2 = self.grasp_critic2(state_action)
            #     else:
            #         nq1 = self.push_critic1(state_action)
            #         nq2 = self.push_critic2(state_action)
            
            #     nq  = torch.min(nq1, nq2)

    #             #store experience 
    #             self.buffer_replay.store_transition(s.to(torch.device('cpu')).detach().numpy(), 
    #                                                 a.to(torch.device('cpu')).detach().numpy(), 
    #                                                 a_type_onehot.to(torch.device('cpu')).detach().numpy(), 
    #                                                 r, 
    #                                                 ns.to(torch.device('cpu')).detach().numpy(), 
    #                                                 done, 
    #                                                 q.to(torch.device('cpu')).detach().numpy(), 
    #                                                 nq.to(torch.device('cpu')).detach().numpy())    

    #             #update parameter
    #             self.online()

    #             #update history
    #             ep_r += r
    #             step += 1
                
    #             #check if done
    #             if done:
    #                 self.env.reset(reset_obj = True)
    #                 print("[SUCCESS] finish one episode")
    #                 self.r_hist.append(ep_r)
    #                 self.step_hist.append(step)         
    #                 break 
    #             else:

    #                 #return home position if grasp successfully
    #                 if is_success_grasp:
    #                     print("[SUCCESS] grasp an object")
    #                     self.env.return_home()

    #                 #check if out of working space
    #                 elif self.env.is_out_of_working_space:
    #                     print("[WARN] out of working space")
    #                     self.env.reset(reset_obj = False)

    #                 #check if action executable
    #                 elif not self.env.can_execute_action:
    #                     print("[WARN] action is not executable")
    #                     self.env.reset(reset_obj = False)

    #                 #check if collision to ground
    #                 elif self.env.is_collision_to_ground:
    #                     print("[WARN] collision to ground")
    #                     self.env.reset(reset_obj = False)


