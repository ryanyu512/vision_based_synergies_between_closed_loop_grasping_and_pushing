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
from buffer_hld import BufferReplay_HLD
from network import Actor, Critic, HDL_Net

#DL related
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

class DemoDataset(Dataset):
    def __init__(self, states, next_states, actions, next_actions, rewards, dones, success_masks):
        self.states        = states
        self.next_states   = next_states
        self.actions       = actions
        self.next_actions  = next_actions
        self.rewards       = rewards
        self.dones         = dones
        self.success_masks = success_masks

    def __len__(self):
        return len(self.grasp_states)

    def __getitem__(self, index):
        state         = self.states[index]
        next_state    = self.next_states[index]
        action        = self.actions[index]
        next_action   = self.next_actions[index]
        reward        = self.rewards[index]
        done          = self.dones[index]
        success_masks = self.success_masks[index]

        return state, next_state, action, next_action, reward, done, success_masks

class DemoDatasetHldNet(Dataset):

    def __init__(self, states, next_states, action_types, rewards, dones):
        self.states        = states
        self.next_states   = next_states
        self.action_types  = action_types
        self.rewards       = rewards
        self.dones         = dones

    def __len__(self):
        return len(self.grasp_states)

    def __getitem__(self, index):
        state        = self.states[index]
        next_state   = self.next_states[index]
        action_type  = self.action_types[index]
        reward       = self.rewards[index]
        done         = self.dones[index]

        return state, next_state, action_type, reward, done
    
class Agent():

    def __init__(self, 
                 env,
                 N_action = 4,
                 N_gripper_action = 2,
                 N_push_step  = 10,
                 N_grasp_step = 10,
                 N_batch_hld  = 32,
                 N_batch      = 64, 
                 hld_lr       = 1e-5,
                 lr           = 1e-4,
                 alpha        = 0.2,
                 tau          = 0.05,
                 gamma        = 0.95,
                 max_memory_size    = 50000,
                 max_step           = 200,
                 save_step_interval = 10,
                 is_debug           = False):

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"device: {self.device}")

        #initialise env
        self.env = env
        print("[SUCCESS] initialise environment")

        #initialise action and action_type dimension
        self.N_action = N_action
        self.N_push_step  = N_push_step
        self.N_grasp_step = N_grasp_step 
        self.N_gripper_action = N_gripper_action

        #initialise high level network
        self.hld_net        = HDL_Net(N_input_channels = 1, lr = lr)
        self.hld_net_target = HDL_Net(N_input_channels = 1, lr = lr)

        #initialise grasp actor
        self.grasp_actor   = Actor(name = "grasp_actor", N_input_channels = 3, lr = lr) #depth image + gripper status
        #initialise grasp critic network 1
        self.grasp_critic1 = Critic(name = "grasp_critic1", N_input_channels = 9, lr = lr) #depth image + gripper status + dx + dy + dz + dyaw + gripper action
        #initialise grasp ciritc network 2
        self.grasp_critic2 = Critic(name = "grasp_critic2", N_input_channels = 9, lr = lr) #depth image + gripper status + dx + dy + dz + dyaw + gripper action
        #initialise grasp critic network target 1
        self.grasp_critic1_target = Critic(name = "grasp_critic1_target", N_input_channels = 9, lr = lr) #depth image + gripper status + dx + dy + dz + dyaw + gripper action
        #initialise grasp critic network target 2
        self.grasp_critic2_target = Critic(name = "grasp_critic2_target", N_input_channels = 9, lr = lr) #depth image + gripper status + dx + dy + dz + dyaw + gripper action

        #initialise grasp actor
        self.push_actor   = Actor(name = "push_actor", N_input_channels = 2, #depth image + yaw angle
                                  action_type="push", 
                                  lr = lr)  
        #initialise grasp critic network 1
        self.push_critic1 = Critic(name = "push_critic1", N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp ciritc network 2
        self.push_critic2 = Critic(name = "push_critic2", N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp critic network target 1
        self.push_critic1_target = Critic(name = "push_critic1_target", N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp critic network target 2
        self.push_critic2_target = Critic(name = "push_critic2_target", N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 

        #soft update to make critic target align with critic
        self.soft_update(critic = self.hld_net, target_critic = self.hld_net_target)
        self.soft_update(critic = self.grasp_critic1, target_critic = self.grasp_critic1_target)
        self.soft_update(critic = self.grasp_critic2, target_critic = self.grasp_critic2_target)
        self.soft_update(critic = self.push_critic1, target_critic = self.push_critic1_target)
        self.soft_update(critic = self.push_critic2, target_critic = self.push_critic2_target)
        print("[SUCCESS] initialise networks")

        #initialise buffer replay
        self.buffer_replay     = BufferReplay(max_memory_size = max_memory_size)
        self.buffer_replay_hld = BufferReplay_HLD()
        print("[SUCCESS] initialise memory buffer")
        #initialise batch size
        self.N_batch = N_batch
        #initialise batch size for hld-net
        self.N_batch_hld = N_batch_hld
        #initialise small constant to prevent zero value
        self.sm_c    = 1e-6
        #initialise learning rate of low-level action learning rate
        self.lr      = lr
        #initialise learning rate of hld learning rate
        self.hld_lr  = hld_lr
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

        #initialise max step per episode
        self.max_step = max_step

    def get_raw_data(self, action_type):

        _, depth_img         = self.env.get_rgbd_data()
        _, gripper_tip_ori   = self.env.get_obj_pose(self.env.gripper_tip_handle, 
                                                     self.env.sim.handle_world)
        if action_type == constants.PUSH:
            yaw_ang        = gripper_tip_ori[2]
            gripper_status = None
        else:
            yaw_ang        = gripper_tip_ori[2]
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
        max_depth_diff = (self.env.far_clip_plane - self.env.near_clip_plane)
        offset_depth_img = in_depth_img.astype(np.float32) - self.env.near_clip_plane
        in_depth_img = offset_depth_img/max_depth_diff
        in_depth_img = np.expand_dims(in_depth_img, axis = 2)

        state = copy.copy(in_depth_img)

        #turn gripper state into image
        #GRIPPER_NON_CLOSE_NON_OPEN = 2, largest value
        max_gripper_action = constants.GRIPPER_NON_CLOSE_NON_OPEN 
        if gripper_state is not None and not np.isnan(gripper_state):
            gripper_state_img  = np.ones_like(in_depth_img)*gripper_state
            #scale gripper_state_img into range 0 - 1
            gripper_state_img = gripper_state_img.astype(np.float32)/max_gripper_action 
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
        max_action = constants.MAX_ACTION
        dx_img     = np.ones((1, resol[0], resol[1]))*move_action[0]/max_action[0]
        dy_img     = np.ones((1, resol[0], resol[1]))*move_action[1]/max_action[1]
        dz_img     = np.ones((1, resol[0], resol[1]))*move_action[2]/max_action[2]
        dyaw_image = np.ones((1, resol[0], resol[1]))*move_action[3]/max_action[3]
        gc_img     = np.ones((1, resol[0], resol[1]))*np.argmax(gripper_action)

        action_img = np.concatenate((dx_img, dy_img, dz_img, dyaw_image, gc_img), axis = 0)
        action_img = torch.FloatTensor(action_img).unsqueeze(0).to(self.device)            

        return action_img

    def soft_update(self, critic, target_critic, tau = None):
        
        if tau is None:
            tau = 1.

        # Weights
        for (target_param, param) in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # BatchNorm2d
        for target_layer, source_layer in zip(target_critic.encoder, critic.encoder):
            if isinstance(target_layer, torch.nn.BatchNorm2d):
                target_layer.running_mean.copy_(tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean)
                target_layer.running_var.copy_(tau * source_layer.running_var + (1 - tau) * target_layer.running_var)

        # BatchNorm1d
        for target_layer, source_layer in zip(target_critic.fc, critic.fc):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                target_layer.running_mean.copy_(tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean)
                target_layer.running_var.copy_(tau * source_layer.running_var + (1 - tau) * target_layer.running_var)

    def get_train_test_dataloader_hld_net(self, exp, train_ratio = 0.8):

        #          0,            1,            2,       3,                 4,     
        # priorities, depth_states, action_types, rewards, next_depth_states, 
        #     5,          6,          7,               8,               9
        # dones, predict_qs, labeled_qs, predict_next_qs, labeled_next_qs 

        #split experience into training (80%) and valuation (20%)
        indices       = np.arange(len(exp[0]))
        train_size    = int(train_ratio*len(indices))
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        #turn experience into compatible inputs to grasp network
        # state, next_state, action_type, reward, done, next_q_value
        states        = []
        next_states   = []
        action_types  = []
        rewards       = []
        dones         = []

        # states, target_action_types, rewards, next_q_values 
        for i in range(len(indices)):
            #get state
            state = self.preprocess_state(depth_img     = exp[1][i], 
                                          gripper_state = None, 
                                          yaw_ang       = None)
            
            states.append(state)

            next_state = self.preprocess_state(depth_img     = exp[4][i], 
                                               gripper_state = None, 
                                               yaw_ang       = None)
            
            next_states.append(next_state)

            #get action types
            action_type = copy.copy(exp[2][i])
            action_types.append(action_type)

            #get reward
            reward = copy.copy(exp[3][i])
            rewards.append(reward)

            #get dones
            done = copy.copy(exp[5][i])
            dones.append(done)

        states        = torch.FloatTensor(np.array(states))
        next_states   = torch.FloatTensor(np.array(next_states))
        action_types  = torch.FloatTensor(np.array(action_types))
        rewards       = torch.FloatTensor(np.array(rewards))
        dones         = torch.FloatTensor(np.array(dones))

        dataset  = DemoDatasetHldNet(states, next_states, action_types, rewards, dones)

        #create subset
        train_subset = Subset(dataset, train_indices)
        test_subset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size = self.N_batch_hld, shuffle = True, 
                                  drop_last=True if train_size > self.N_batch else False)
        test_loader  = DataLoader( test_subset, batch_size = self.N_batch_hld, shuffle = False)

        return train_loader, test_loader

    def get_train_test_dataloader(self,
                                  exp, 
                                  train_ratio = 0.8, 
                                  is_grasp = True):

        #            0,          1,            2,              3,          4,             
        # action_index, priorities, depth_states, gripper_states, yaw_states, 
        #       5,               6,            7,                    8,
        # actions, gripper_actions, next_actions, next_gripper_actions,  
        #       9,                10,                  11,              12,         
        # rewards, next_depth_states, next_gripper_states, next_yaw_states, 
        #    13,         14,         15            16
        # dones, predict_qs, labeled_qs, success_mask

        #split experience into training (80%) and valuation (20%)
        indices       = np.arange(len(exp[0]))
        train_size    = int(train_ratio*len(indices))
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        #turn experience into compatible inputs to grasp network
        #state, next_state, action, next_action, reward, done, success_masks
        states        = []
        next_states   = []
        actions       = []
        next_actions  = []
        rewards       = []
        dones         = []
        success_masks = []
        
        for i in range(len(indices)):
            #get state and next state
            state = self.preprocess_state(depth_img     = exp[2][i], 
                                          gripper_state = exp[3][i], 
                                          yaw_ang       = exp[4][i])
            
            states.append(state)

            next_state = self.preprocess_state(depth_img     = exp[10][i], 
                                               gripper_state = exp[11][i], 
                                               yaw_ang       = exp[12][i])
            
            next_states.append(next_state)

            #get action and next action
            if is_grasp:
                action      = np.hstack((exp[5][i], exp[6][i]))
                next_action = np.hstack((exp[7][i], exp[8][i]))
            else:
                action      = exp[5][i]
                next_action = exp[7][i]

            actions.append(action)
            next_actions.append(next_action)

            #get reward
            rewards.append(exp[9][i])

            #get dones
            dones.append(exp[13][i])

            #get success_masks
            success_masks.append(exp[16][i])

        states        = torch.FloatTensor(np.array(states))
        next_state    = torch.FloatTensor(np.array(next_states))
        actions       = torch.FloatTensor(np.array(actions))
        next_actions  = torch.FloatTensor(np.array(next_actions))
        rewards       = torch.FloatTensor(np.array(rewards))
        dones         = torch.FloatTensor(np.array(dones))
        success_masks = torch.FloatTensor(np.array(success_masks))

        dataset  = DemoDataset(states, next_state, actions, next_actions, rewards, dones, success_masks)

        #create subset
        train_subset = Subset(dataset, train_indices)
        test_subset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size = self.N_batch, shuffle = True, 
                                  drop_last=True if train_size > self.N_batch else False)
        test_loader  = DataLoader( test_subset, batch_size = self.N_batch, shuffle = False)

        return train_loader, test_loader

    def behaviour_cloning_hld(self, train_loader, test_loader, hld_net, hld_net_target, num_epochs=500):

        #initialise optimizers for both critics and actor
        hld_optimizer = optim.Adam(hld_net.parameters(), lr=self.hld_lr)
        
        #initilialise loss function
        mse_loss  = nn.MSELoss()

        #initialise best evaluation loss
        best_hld_loss_eval = np.inf

        for epoch in range(num_epochs):
            mean_hld_loss   = 0.

            batch_cnt = 0

            #set network in training mode
            hld_net.train()

            for states, next_states, target_action_types, rewards, dones in train_loader:            

                states              = states.to(self.device)
                next_states         = next_states.to(self.device)
                target_action_types = target_action_types.to(self.device)
                rewards             = rewards.unsqueeze(1).to(self.device)
                dones               = dones.unsqueeze(1).to(self.device)

                # compute current q - values
                batch_indices = torch.arange(states.size(0)).long().to(self.device)
                qs = hld_net(states)[batch_indices, target_action_types.long()]

                # Compute target q - values
                with torch.no_grad():
                    next_qs   = torch.max(hld_net_target(next_states), dim=1)[0]
                    target_qs = rewards + self.gamma * (1 - dones) * next_qs

                hld_net_loss = mse_loss(qs, target_qs)

                # update hld-net
                hld_optimizer.zero_grad()
                hld_net_loss.backward()
                hld_optimizer.step()

                mean_hld_loss += hld_net_loss.item()
            
                batch_cnt += 1

                #soft update target network 
                self.soft_update(critic = hld_net, target_critic = hld_net_target, tau = self.tau)
        
            mean_hld_loss /= batch_cnt

            hld_loss_eval = self.behaviour_cloning_eval_hld(test_loader, hld_net, hld_net_target)

            if self.is_debug:
                loss_msg  = f"Epoch: {epoch+1}/{num_epochs} "
                loss_msg += f"hld loss: {mean_hld_loss :.6f} "
                loss_msg += f"hld loss eval: {hld_loss_eval :.6f}/{best_hld_loss_eval :.6f}"
                print(loss_msg)

            if (hld_loss_eval < best_hld_loss_eval):
               best_hld_loss_eval = hld_loss_eval

               hld_net.save_checkpoint()
               hld_net_target.save_checkpoint()

               print("[SUCCESS] save model!")

    def behaviour_cloning_eval_hld(self, test_loader, hld_net, hld_net_target):
        
        #set network in evaluation mode
        hld_net.eval()
        
        #initialise loss funciton
        mse_loss  = nn.MSELoss()

        #initialise mean loss value
        mean_hld_loss   = 0.

        #initialise batch counter
        batch_cnt = 0

        with torch.no_grad():

            for states, next_states, target_action_types, rewards, dones in test_loader:            

                states              = states.to(self.device)
                next_states         = next_states.to(self.device)
                target_action_types = target_action_types.to(self.device)
                rewards             = rewards.unsqueeze(1).to(self.device)
                dones               = dones.unsqueeze(1).to(self.device)

                # compute current q - values
                batch_indices = torch.arange(states.size(0)).long().to(self.device)
                qs = hld_net(states)[batch_indices, target_action_types.long()]

                # Compute target q - values
                with torch.no_grad():
                    next_qs   = torch.max(hld_net_target(next_states), dim=1)[0]
                    target_qs = rewards + self.gamma * (1 - dones) * next_qs

                hld_net_loss = mse_loss(qs, target_qs)

                mean_hld_loss += hld_net_loss.item()
            
                batch_cnt += 1

        mean_hld_loss /= batch_cnt

        return mean_hld_loss


    def behaviour_cloning(self, train_loader, test_loader, critic1, critic2, target_critic1, target_critic2, actor, num_epochs=500, is_grasp = True):
        #TODO [FINISH 25 AUG 2024]: handle failure experience 

        # Initialize optimizers for both critics and actor
        critic1_optimizer = optim.Adam(critic1.parameters(), lr=self.lr)
        critic2_optimizer = optim.Adam(critic2.parameters(), lr=self.lr)
        actor_optimizer   = optim.Adam(actor.parameters(), lr=self.lr)
        
        mse_loss = nn.MSELoss()
        mse_loss_no_reduce = nn.MSELoss(reduction = 'none')
        ce_loss_no_reduce  = nn.CrossEntropyLoss(reduction='none')

        best_actor_loss_eval   = np.inf
        best_critic1_loss_eval = np.inf
        best_critic2_loss_eval = np.inf

        #get action max range
        max_action = constants.MAX_ACTION
        for epoch in range(num_epochs):

            #set networks to training mode
            actor.train()
            critic1.train()
            critic2.train()

            #initialise mean loss for each episode
            mean_actor_loss   = 0.
            mean_critic1_loss = 0.
            mean_critic2_loss = 0.

            #initialise batch counter
            batch_cnt = 0

            for state, next_state, target_action, next_target_action, reward, done, success_mask in train_loader:

                state              = state.to(self.device)
                next_state         = next_state.to(self.device)
                target_action      = target_action.to(self.device)
                next_target_action = next_target_action.to(self.device)
                reward             = reward.unsqueeze(1).to(self.device)
                done               = done.unsqueeze(1).to(self.device)
                success_mask       = success_mask.to(self.device)

                #compute normalisation factor
                if is_grasp:
                    normalise_factor    = torch.tensor(max_action + [1., 1.]).to(self.device) 
                else:
                    normalise_factor    = torch.tensor(max_action).to(self.device)

                #compute action state
                normalise_target_action = target_action/normalise_factor.view(1, normalise_factor.shape[0])
                normalise_next_target_action = next_target_action/normalise_factor.view(1, normalise_factor.shape[0])

                #transform vector into image format
                ta_shape = target_action.shape
                torch_ones = torch.ones((ta_shape[0], ta_shape[1], 128, 128)).to(self.device)
                target_action_img   = normalise_target_action.view(ta_shape[0], ta_shape[1], 1, 1)*torch_ones

                next_ta_shape = next_target_action.shape
                torch_ones = torch.ones((next_ta_shape[0], next_ta_shape[1], 128, 128)).to(self.device)
                next_target_action_img   = normalise_next_target_action.view(next_ta_shape[0], next_ta_shape[1], 1, 1)*torch_ones

                #compute next action state
                next_action_state = torch.concatenate((next_state, next_target_action_img), axis = 1).float()

                #forward pass through both critics
                with torch.no_grad():
                    next_q1  = target_critic1(next_action_state)
                    next_q2  = target_critic2(next_action_state)
                    next_q   = torch.min(next_q1, next_q2)
                    target_q = reward + (1 - done)*self.gamma*next_q

                #compute action state
                action_state = torch.concatenate((state, target_action_img), axis = 1).float()
 
                q1 = critic1(action_state)
                q2 = critic2(action_state)
                
                # Use the actor to get the predicted actions
                actions, gripper_action, z, normal, gripper_action_probs = actor.get_actions(state)

                #normalise actions
                normalised_actions = actions/normalise_factor[0:4].view(1, normalise_factor[0:4].shape[0])

                # Compute loss for both critics
                critic1_loss = mse_loss(q1, target_q)
                critic2_loss = mse_loss(q2, target_q)

                # compute actor loss 
                move_loss  = mse_loss_no_reduce(normalised_actions.float(), normalise_target_action[:,0:4].float())
                move_loss  = torch.sum(move_loss, dim=1)* success_mask

                actor_loss = move_loss
                if is_grasp:
                    target_gripper_action_class = target_action[:, 4:].argmax(dim=1)
                    gripper_action_loss = ce_loss_no_reduce(gripper_action_probs, target_gripper_action_class)* success_mask 
                    actor_loss += gripper_action_loss
                
                actor_loss_denominator = success_mask.sum()
                if actor_loss_denominator.item() > 0:
                    actor_loss =  actor_loss.sum()/actor_loss_denominator
                else:
                    actor_loss = actor_loss.mean()

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
            
                batch_cnt += 1

                #TODO [FINISH 26 AUG 2024]: copy critic network to target critic network
                self.soft_update(critic = critic1, target_critic = target_critic1, tau = self.tau)
                self.soft_update(critic = critic2, target_critic = target_critic2, tau = self.tau)
        
            mean_actor_loss   /= batch_cnt
            mean_critic1_loss /= batch_cnt
            mean_critic2_loss /= batch_cnt

            critic1_loss_eval, critic2_loss_eval, actor_loss_eval = self.behaviour_cloning_eval(test_loader,  critic1, critic2, target_critic1, target_critic2, actor, is_grasp)

            if self.is_debug:
                print(f"Epoch: {epoch+1}/{num_epochs} ")

                train_loss_msg  = "[TRAIN] "
                train_loss_msg += f"critic1 loss: {mean_critic1_loss :.6f}, critic2 loss: {mean_critic2_loss :.6f}, actor Loss: {mean_actor_loss :.6f} "
                print(train_loss_msg)

                critic_loss_eval_msg  = "[EVAL] "
                critic_loss_eval_msg += f"critic1 loss eval: {critic1_loss_eval :.6f}/{best_critic1_loss_eval :.6f} "
                critic_loss_eval_msg += f"critic2 loss eval: {critic2_loss_eval :.6f}/{best_critic2_loss_eval :.6f} "
                print(critic_loss_eval_msg)

                actor_loss_eval_msg  = "[EVAL] "
                actor_loss_eval_msg += f"actor loss eval: {actor_loss_eval :.6f}/{best_actor_loss_eval :.6f} "
                print(actor_loss_eval_msg)

            if critic1_loss_eval < best_critic1_loss_eval:
               
                best_critic1_loss_eval = critic1_loss_eval
                critic1.save_checkpoint()
                target_critic1.save_checkpoint()
                print("[SUCCESS] save critic1 model!")

            if critic2_loss_eval < best_critic2_loss_eval:
               
                best_critic2_loss_eval = critic2_loss_eval
                critic2.save_checkpoint()
                target_critic2.save_checkpoint()
                print("[SUCCESS] save critic2 model!")

            if (actor_loss_eval < best_actor_loss_eval):
               
                best_actor_loss_eval  = actor_loss_eval
                actor.save_checkpoint()

                print("[SUCCESS] save actor model!")

    def behaviour_cloning_eval(self, test_loader,  critic1, critic2, target_critic1, target_critic2, actor, is_grasp = True):

        #set to evaluation mode
        actor.eval()
        critic1.eval()
        critic2.eval()

        #initialise loss function
        mse_loss = nn.MSELoss()
        mse_loss_no_reduce = nn.MSELoss(reduction = 'none')
        ce_loss_no_reduce  = nn.CrossEntropyLoss(reduction='none')

        #initialise mean loss
        mean_actor_loss   = 0.
        mean_critic1_loss = 0.
        mean_critic2_loss = 0.

        #initialise batch ounter
        batch_cnt = 0

        #define max action range
        max_action = constants.MAX_ACTION

        with torch.no_grad():

            for state, next_state, target_action, next_target_action, reward, done, success_mask in test_loader:

                state              = state.to(self.device)
                next_state         = next_state.to(self.device)
                target_action      = target_action.to(self.device)
                next_target_action = next_target_action.to(self.device)
                reward             = reward.to(self.device)
                done               = done.to(self.device)
                success_mask       = success_mask.to(self.device)

                #compute normalisation factor
                if is_grasp:
                    normalise_factor    = torch.tensor(max_action + [1., 1.]).to(self.device) 
                else:
                    normalise_factor    = torch.tensor(max_action).to(self.device)

                #compute action state
                normalise_target_action = target_action/normalise_factor.view(1, normalise_factor.shape[0])
                normalise_next_target_action = next_target_action/normalise_factor.view(1, normalise_factor.shape[0])

                #transform vector into image format
                ta_shape = target_action.shape
                torch_ones = torch.ones((ta_shape[0], ta_shape[1], 128, 128)).to(self.device)
                target_action_img   = normalise_target_action.view(ta_shape[0], ta_shape[1], 1, 1)*torch_ones

                next_ta_shape = next_target_action.shape
                torch_ones = torch.ones((next_ta_shape[0], next_ta_shape[1], 128, 128)).to(self.device)
                next_target_action_img   = normalise_next_target_action.view(next_ta_shape[0], next_ta_shape[1], 1, 1)*torch_ones

                #compute next action state
                next_action_state = torch.concatenate((next_state, next_target_action_img), axis = 1).float()

                #forward pass through both critics
                with torch.no_grad():
                    next_q1  = target_critic1(next_action_state)
                    next_q2  = target_critic2(next_action_state)
                    next_q   = torch.min(next_q1, next_q2)
                    target_q = reward + (1 - done)*self.gamma*next_q

                #compute action state
                action_state = torch.concatenate((state, target_action_img), axis = 1).float()
 
                q1 = critic1(action_state)
                q2 = critic2(action_state)

                # Use the actor to get the predicted actions
                actions, gripper_action, z, normal, gripper_action_probs = actor.get_actions(state)

                #normalise actions
                normalised_actions = actions/normalise_factor[0:4].view(1, normalise_factor[0:4].shape[0])

                # Compute loss for both critics
                critic1_loss = mse_loss(q1, target_q)
                critic2_loss = mse_loss(q2, target_q)

                # compute actor loss 
                move_loss  = mse_loss_no_reduce(normalised_actions.float(), normalise_target_action[:,0:4].float())
                move_loss  = torch.sum(move_loss, dim=1)* success_mask

                actor_loss = move_loss
                if is_grasp:
                    target_gripper_action_class = target_action[:, 4:].argmax(dim=1)
                    gripper_action_loss = ce_loss_no_reduce(gripper_action_probs, target_gripper_action_class)* success_mask 
                    actor_loss += gripper_action_loss
                
                actor_loss_denominator = success_mask.sum()
                if actor_loss_denominator.item() > 0:
                    actor_loss =  actor_loss.sum()/actor_loss_denominator
                else:
                    actor_loss = actor_loss.mean()

                mean_actor_loss   += actor_loss.item()
                mean_critic1_loss += critic1_loss.item()
                mean_critic2_loss += critic2_loss.item()

                batch_cnt += 1
            
            mean_actor_loss   /= batch_cnt
            mean_critic1_loss /= batch_cnt
            mean_critic2_loss /= batch_cnt
        
        return mean_critic1_loss, mean_critic2_loss, mean_actor_loss


    def gather_guidance_experience(self,
                                   is_debug = True):

        if self.buffer_replay.is_full:
            print("[SUCCESS] buffer is full")
            return

        #start trainiing/evaluation loop
        episode = 0
        while not self.buffer_replay.is_full:

            #initialise episode data
            step = 0
            ep_r = 0.
            done = False
            action_type = None
            previous_action_type = None
            is_success_grasp = False

            #initialise hld-net experience in one episode

            hld_depth_states      = []
            hld_action_types      = []
            hld_rewards           = []
            hld_next_depth_states = []
            hld_dones             = []

            #reset environment
            self.env.reset(reset_item = True)

            while not done and step < self.max_step and not self.buffer_replay.is_full:

                print(f"==== episode: {episode} ====")

                #action selection
                delta_moves_grasp = self.env.grasp_guidance_generation()
                delta_moves_push  = self.env.push_guidance_generation()

                #no grasp move or (previous grasp is fail and have push move)
                delta_moves = []
                if len(delta_moves_grasp) == 0:
                    print("[ACTION CHOICE] no grasp options => try to push!")
                    action_type = constants.PUSH
                    delta_moves = delta_moves_push
                elif (previous_action_type == constants.GRASP and not is_success_grasp and len(delta_moves_push) > 0):
                    print("[ACTION CHOICE] try to push after grasping fails")
                    action_type = constants.PUSH
                    delta_moves = delta_moves_push
                else:
                    print("[ACTION CHOICE] grasp!")
                    action_type = constants.GRASP
                    delta_moves = delta_moves_grasp

                if len(delta_moves) == 0:
                    done = True

                #initialise memory space for storing a complete set of actions
                depth_states         = []
                gripper_states       = []
                yaw_states           = []
                actions              = []
                gripper_actions      = []
                next_actions         = []
                next_gripper_actions = []
                action_types         = []
                rewards              = []
                next_depth_states    = []
                next_gripper_states  = []
                next_yaw_states      = []
                dones                = []
                
                for i in range(len(delta_moves)):

                    print(f"==== step: {step} N_pickable_item: {self.env.N_pickable_item} ====")

                    #get raw data
                    depth_img, gripper_state, yaw_state = self.get_raw_data(action_type)

                    #get state
                    state = self.preprocess_state(depth_img     = depth_img, 
                                                  gripper_state = gripper_state, 
                                                  yaw_ang       = yaw_state)
                    
                    #action selection
                    move   = np.array(delta_moves[i])
                    n_move = np.array(delta_moves[i+1] if i+1 < len(delta_moves) else [0,0,0,0,move[-2],move[-1]])

                    action,   action_type,  gripper_action             = np.array(move[0:self.N_action]), action_type,   np.array(move[-2:])
                    next_action, next_action_type, next_gripper_action = np.array(n_move[0:self.N_action]), action_type, np.array(n_move[-2:])

                    gripper_action = torch.FloatTensor(gripper_action).unsqueeze(0).to(self.device)
                    action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    
                    next_gripper_action = torch.FloatTensor(next_gripper_action).unsqueeze(0).to(self.device)
                    next_action = torch.FloatTensor(next_action).unsqueeze(0).to(self.device)

                    #step
                    reward, is_success_grasp = self.env.step(action_type, 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][0:3], 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][3], 
                                                             is_open_gripper = True if move[4] == 1 else False)

                    #get raw data after actions
                    next_depth_img, next_gripper_state, next_yaw_state = self.get_raw_data(action_type)

                    #print actions
                    if is_debug:
                        print(f"[STEP]: {step} [ACTION TYPE]: {action_type} [REWARD]: {reward}")  
                        print(f"[MOVE]: {action.to(torch.device('cpu')).detach().numpy()[0]}") 

                    #get next state
                    next_state = self.preprocess_state(depth_img     = next_depth_img, 
                                                       gripper_state = next_gripper_state, 
                                                       yaw_ang       = next_yaw_state)

                    #check if all items are picked
                    done = False if self.env.N_pickable_item > 0 else True
                    
                    # store experience in one action

                    depth_states.append(depth_img)
                    gripper_states.append(gripper_state)
                    yaw_states.append(yaw_state)
                    actions.append(action.to(torch.device('cpu')).detach().numpy()[0])
                    gripper_actions.append(gripper_action.to(torch.device('cpu')).detach().numpy()[0])
                    next_actions.append(next_action.to(torch.device('cpu')).detach().numpy()[0])
                    next_gripper_actions.append(next_gripper_action.to(torch.device('cpu')).detach().numpy()[0])
                    action_types.append(action_type)
                    rewards.append(reward)
                    next_depth_states.append(next_depth_img)
                    next_gripper_states.append(next_gripper_state)
                    next_yaw_states.append(next_yaw_state)
                    dones.append(True if (i == len(delta_moves) - 1 or done) else False)

                    #update history
                    ep_r += reward
                    step += 1

                    #check if done
                    if done:
                        print("[SUCCESS] finish one episode")
                        self.r_hist.append(ep_r)
                        self.step_hist.append(step)         
                        break 
                    else:

                        #return home position if grasp successfully
                        if is_success_grasp or i == len(delta_moves) - 1:
                            print("[SUCCESS] finish actions or grasp an item")
                            self.env.return_home(action_type)
                            print("[SUCCESS] return home position")

                        #check if out of working space
                        elif self.env.is_out_of_working_space:
                            print("[WARN] out of working space")
                            self.env.reset(reset_item = False)

                        #check if action executable
                        elif not self.env.can_execute_action:
                            print("[WARN] action is not executable")
                            self.env.reset(reset_item = False)

                        #check if collision to ground
                        elif self.env.is_collision_to_ground:
                            print("[WARN] collision to ground")
                            self.env.reset(reset_item = False)

                    #this set of motions is not executable => break it
                    if (self.env.is_out_of_working_space or 
                        not self.env.can_execute_action or
                        self.env.is_collision_to_ground):
                        print("[WARN] stop executing this action!")
                        break

                #update previous action type
                previous_action_type = action_type

                 #store transition experience of low-level behaviour  

                depth_states         = np.array(depth_states)
                gripper_states       = np.array(gripper_states)
                yaw_states           = np.array(yaw_states)
                actions              = np.array(actions)
                gripper_actions      = np.array(gripper_actions)
                next_actions         = np.array(next_actions)
                next_gripper_actions = np.array(next_gripper_actions)
                action_types         = np.array(action_types)
                rewards              = np.array(rewards)
                next_depth_states    = np.array(next_depth_states)
                dones                = np.array(dones)

                for i in range(len(depth_states)): 
                    self.buffer_replay.store_transition(True if i == 0 else False, #store if at home pos
                                                        depth_states[i], 
                                                        gripper_states[i],
                                                        yaw_states[i],
                                                        actions[i], 
                                                        gripper_actions[i], 
                                                        next_actions[i],
                                                        next_gripper_actions[i],
                                                        action_types[i],
                                                        rewards[i], 
                                                        next_depth_states[i], 
                                                        next_gripper_states[i],
                                                        next_yaw_states[i],
                                                        dones[i], 
                                                        0., 
                                                        0.,
                                                        True if rewards.sum() > 0 else False) 
                    
                print('[SUCCESS] store transition experience for low-level action')
                #TODO [NOTE 24 Aug 2024]: temporary design of hld reward
                #TODO [NOTE 24 Aug 2024]: add hld_reward function in env class
                #TODO [FINISH 25 Aug 2024]: handle sudden stop execution

                #compute the hld reward after completing one set of action
                hld_reward = 0.
                if action_type == constants.GRASP:
                    if np.sum(rewards) > 0:
                        hld_reward += 1.0
                elif action_type == constants.PUSH:                                
                    if np.sum(rewards) > 0:
                        hld_reward += 0.5

                #store hld-net data after completing one set of action
                hld_depth_states.append(depth_states[0])
                hld_action_types.append(action_type)
                hld_rewards.append(hld_reward)
                hld_next_depth_states.append(next_depth_states[-1])
                hld_dones.append(False if self.env.N_pickable_item > 0 else True)
                print('[SUCCESS] store hld-net data')

                #reset any items out of working space
                #TODO [NOTE 24 AUG 2024]: reset any items after one complete set of actions
                self.env.reset_item2workingspace()    
                time.sleep(0.5) 
                print('[SUCCESS] reset items to workplaces if any')
                
            #update hld-net q values
            hld_depth_states      = np.array(hld_depth_states)
            hld_action_types      = np.array(hld_action_types)
            hld_rewards           = np.array(hld_rewards)
            hld_next_depth_states = np.array(hld_next_depth_states)
            hld_dones             = np.array(hld_dones)

            print('[SUCCESS] compute backward estimation hld-net q values')

            #store transition experience of high-level behaviour
            for i in range(len(hld_rewards)):
                self.buffer_replay_hld.store_transition(hld_depth_states[i],
                                                        hld_action_types[i],
                                                        hld_rewards[i],
                                                        hld_next_depth_states[i],
                                                        hld_dones[i],
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.)
            
            print('[SUCCESS] store transition experience for high-level decision')

            #save buffer for each episode
            self.buffer_replay.save_buffer()
            print(f"[SUCCESS] save buffer {self.buffer_replay.memory_cntr+1}/{self.buffer_replay.max_memory_size}")
            self.buffer_replay_hld.save_buffer()
            print(f"[SUCCESS] save hld buffer {self.buffer_replay_hld.memory_size}/{self.buffer_replay_hld.memory_size}")
            
            #update episode
            episode += 1

        print("[SUCCESS] buffer is full")
        
            
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
    #             done = False if self.env.N_pickable_item > 0 else True

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
    #                 self.env.reset(reset_item = True)
    #                 print("[SUCCESS] finish one episode")
    #                 self.r_hist.append(ep_r)
    #                 self.step_hist.append(step)         
    #                 break 
    #             else:

    #                 #return home position if grasp successfully
    #                 if is_success_grasp:
    #                     print("[SUCCESS] grasp an item")
    #                     self.env.return_home()

    #                 #check if out of working space
    #                 elif self.env.is_out_of_working_space:
    #                     print("[WARN] out of working space")
    #                     self.env.reset(reset_item = False)

    #                 #check if action executable
    #                 elif not self.env.can_execute_action:
    #                     print("[WARN] action is not executable")
    #                     self.env.reset(reset_item = False)

    #                 #check if collision to ground
    #                 elif self.env.is_collision_to_ground:
    #                     print("[WARN] collision to ground")
    #                     self.env.reset(reset_item = False)


