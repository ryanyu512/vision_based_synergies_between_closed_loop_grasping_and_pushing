import os
import time
import pickle 
import random 

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
    def __init__(self, depth_imgs, next_depth_imgs, \
                 gripper_states, next_gripper_states, \
                 yaw_angs, next_yaw_angs, \
                 actions, next_actions, \
                 rewards, dones, success_masks):
        
        self.depth_imgs          = depth_imgs
        self.next_depth_imgs     = next_depth_imgs
        self.gripper_states      = gripper_states
        self.next_gripper_states = next_gripper_states
        self.yaw_angs            = yaw_angs
        self.next_yaw_angs       = next_yaw_angs
        self.actions             = actions
        self.next_actions        = next_actions
        self.rewards             = rewards
        self.dones               = dones
        self.success_masks       = success_masks

    def __len__(self):
        return len(self.grasp_states)

    def __getitem__(self, index):
        depth_img          = self.depth_imgs[index]
        next_depth_img     = self.next_depth_imgs[index]
        gripper_state      = self.gripper_states[index]
        next_gripper_state = self.next_gripper_states[index]
        yaw_ang            = self.yaw_angs[index]
        next_yaw_ang       = self.next_yaw_angs[index]
        action             = self.actions[index]
        next_action        = self.next_actions[index]
        reward             = self.rewards[index]
        done               = self.dones[index]
        success_masks      = self.success_masks[index]

        if gripper_state is not None and not np.isnan(gripper_state[0].item()):
            state = torch.concatenate([depth_img.unsqueeze(0), 
                                       gripper_state.expand(128, 128).unsqueeze(0), 
                                       yaw_ang.expand(128, 128).unsqueeze(0)], 
                                       dim=0)
            
            next_state = torch.concatenate([next_depth_img.unsqueeze(0), 
                                            next_gripper_state.expand(128, 128).unsqueeze(0), 
                                            next_yaw_ang.expand(128, 128).unsqueeze(0)], 
                                            dim=0)
            
            action_state = torch.concatenate([state, 
                                              (action[0]/constants.MAX_ACTION[0]).expand(128, 128).unsqueeze(0),
                                              (action[1]/constants.MAX_ACTION[1]).expand(128, 128).unsqueeze(0),
                                              (action[2]/constants.MAX_ACTION[2]).expand(128, 128).unsqueeze(0),
                                              (action[3]/constants.MAX_ACTION[3]).expand(128, 128).unsqueeze(0),
                                               action[4].expand(128, 128).unsqueeze(0)],
                                               dim = 0)
            
            next_action_state = torch.concatenate([next_state, 
                                                   (next_action[0]/constants.MAX_ACTION[0]).expand(128, 128).unsqueeze(0),
                                                   (next_action[1]/constants.MAX_ACTION[1]).expand(128, 128).unsqueeze(0),
                                                   (next_action[2]/constants.MAX_ACTION[2]).expand(128, 128).unsqueeze(0),
                                                   (next_action[3]/constants.MAX_ACTION[3]).expand(128, 128).unsqueeze(0),
                                                    next_action[4].expand(128, 128).unsqueeze(0)],
                                                    dim = 0)
        else:
            state = torch.concatenate([depth_img.unsqueeze(0), 
                                       yaw_ang.expand(128, 128).unsqueeze(0)], 
                                       dim=0)
            
            next_state = torch.concatenate([next_depth_img.unsqueeze(0), 
                                            next_yaw_ang.expand(128, 128).unsqueeze(0)], 
                                            dim=0)
            
            action_state = torch.concatenate([state, 
                                              (action[0]/constants.PUSH_MAX_ACTION[0]).expand(128, 128).unsqueeze(0),
                                              (action[1]/constants.PUSH_MAX_ACTION[1]).expand(128, 128).unsqueeze(0),
                                              (action[2]/constants.PUSH_MAX_ACTION[2]).expand(128, 128).unsqueeze(0),
                                              (action[3]/constants.PUSH_MAX_ACTION[3]).expand(128, 128).unsqueeze(0)],
                                               dim = 0)
            next_action_state = torch.concatenate([next_state, 
                                                  (next_action[0]/constants.PUSH_MAX_ACTION[0]).expand(128, 128).unsqueeze(0),
                                                  (next_action[1]/constants.PUSH_MAX_ACTION[1]).expand(128, 128).unsqueeze(0),
                                                  (next_action[2]/constants.PUSH_MAX_ACTION[2]).expand(128, 128).unsqueeze(0),
                                                  (next_action[3]/constants.PUSH_MAX_ACTION[3]).expand(128, 128).unsqueeze(0)],
                                                   dim = 0)

        return state, next_state, \
               action, next_action, \
               action_state, next_action_state, \
               reward, done, success_masks

class DemoDatasetHLDNet(Dataset):

    def __init__(self, states, next_states, action_types, next_action_types, rewards, dones):
        self.states            = states
        self.next_states       = next_states
        self.action_types      = action_types
        self.next_action_types = next_action_types
        self.rewards           = rewards
        self.dones             = dones

    def __len__(self):
        return len(self.grasp_states)

    def __getitem__(self, index):
        state             = self.states[index].unsqueeze(0)
        next_state        = self.next_states[index].unsqueeze(0)
        action_type       = self.action_types[index]
        next_action_types = self.next_action_types[index]
        reward            = self.rewards[index]
        done              = self.dones[index]

        return state, next_state, action_type, next_action_types, reward, done
    
class Agent():

    def __init__(self, 
                 env,
                 N_action = 4,
                 N_gripper_action = 2,
                 N_grasp_step_demo = constants.N_STEP_GRASP_DEMO,
                 N_push_step_demo  = constants.N_STEP_PUSH_DEMO,
                 N_push_step  = constants.N_STEP_PUSH,
                 N_grasp_step = constants.N_STEP_GRASP,
                 N_batch_hld  = 32,                     #batch for high-level decision network
                 N_batch      = 64,                     #batch for low-level action network
                 hld_lr       = 1e-5,                   #learning rate for high-level decision network
                 lr           = 1e-4,                   #learning rate for low-level action network
                 alpha        = 0.05,                   #for SAC entropy maxisation
                 tau          = 0.05,                   #for soft update of gripper and push net 
                 tau_hld      = 0.001,                  #for soft update of hld-net 
                 gamma        = 0.95,                   #discount rate for td-learning
                 bc_lambda    = 10.00,                  #factor for balancing rl loss and bc loss
                 bc_lambda_decay       = 0.999,         #factor to decay bc_lambda
                 min_bc_lambda         = 1.0,           #define min bc_lambda
                 max_memory_size       = 50000,
                 max_step              = 250,
                 save_all_exp_interval = 20,            #mainly used for update all experience's priority
                 max_result_window     = 100,           #the maximum record length that record if this action is successful or fail
                 gripper_loss_weight   = 1.0,           #used for preventing ce loss too dominating
                 is_debug              = False,
                 bc_offline            = False,
                 checkpt_dir           = 'logs/agent'):

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"device: {self.device}")

        #initialise env
        self.env = env
        self.env.N_grasp_step = N_grasp_step_demo
        self.env.N_push_step  = N_push_step_demo
        print("[SUCCESS] initialise environment")

        #initialise action and action_type dimension
        self.N_action         = N_action
        self.N_push_step      = N_push_step
        self.N_grasp_step     = N_grasp_step 
        self.N_gripper_action = N_gripper_action

        #initialise high level network
        self.hld_net        = HDL_Net(name = "hld_net", N_input_channels = 1, lr = lr)
        self.hld_net_target = HDL_Net(name = "hld_net_target", N_input_channels = 1, lr = lr)

        #initialise grasp actor
        self.grasp_actor   = Actor(name = "grasp_actor" if not bc_offline else "grasp_actor_bc_offline", 
                                   max_action = constants.MAX_ACTION,
                                   N_input_channels = 3, 
                                   lr = lr) #depth image + gripper status + yaw state
        #initialise grasp critic network 1
        self.grasp_critic1 = Critic(name = "grasp_critic1" if not bc_offline else "grasp_critic1_bc_offline", 
                                    N_input_channels = 8, 
                                    lr = lr) #depth image + gripper status + yaw state + dx + dy + dz + dyaw + gripper action
        #initialise grasp ciritc network 2
        self.grasp_critic2 = Critic(name = "grasp_critic2" if not bc_offline else "grasp_critic2_bc_offline", 
                                    N_input_channels = 8, 
                                    lr = lr) #depth image + gripper status + yaw state + dx + dy + dz + dyaw + gripper action
        #initialise grasp critic network target 1
        self.grasp_critic1_target = Critic(name = "grasp_critic1_target" if not bc_offline else "grasp_critic1_target_bc_offline", 
                                           N_input_channels = 8, 
                                           lr = lr) #depth image + gripper status + yaw state + dx + dy + dz + dyaw + gripper action
        #initialise grasp critic network target 2
        self.grasp_critic2_target = Critic(name = "grasp_critic2_target" if not bc_offline else "grasp_critic2_target_bc_offline", 
                                           N_input_channels = 8, 
                                           lr = lr) #depth image + gripper status + yaw state + dx + dy + dz + dyaw + gripper action

        #initialise grasp actor
        self.push_actor   = Actor(name = "push_actor" if not bc_offline else "push_actor_bc_offline", 
                                  max_action = constants.PUSH_MAX_ACTION,
                                  N_input_channels = 2, #depth image + yaw angle
                                  action_type="push", 
                                  lr = lr)  
        #initialise grasp critic network 1
        self.push_critic1 = Critic(name = "push_critic1" if not bc_offline else "push_critic1_bc_offline", 
                                   N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp ciritc network 2
        self.push_critic2 = Critic(name = "push_critic2" if not bc_offline else "push_critic2_bc_offline", 
                                   N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp critic network target 1
        self.push_critic1_target = Critic(name = "push_critic1_target" if not bc_offline else "push_critic1_target_bc_offline", 
                                          N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 
        #initialise grasp critic network target 2
        self.push_critic2_target = Critic(name = "push_critic2_target" if not bc_offline else "push_critic2_target_bc_offline", 
                                          N_input_channels = 6, lr = lr) #depth image + yaw angle + dx + dy + dz + dyaw 

        #soft update to make critic target align with critic
        self.soft_update(critic = self.hld_net, target_critic = self.hld_net_target)
        self.soft_update(critic = self.grasp_critic1, target_critic = self.grasp_critic1_target)
        self.soft_update(critic = self.grasp_critic2, target_critic = self.grasp_critic2_target)
        self.soft_update(critic = self.push_critic1, target_critic = self.push_critic1_target)
        self.soft_update(critic = self.push_critic2, target_critic = self.push_critic2_target)
        print("[SUCCESS] initialise networks")

        #initialise buffer replay
        self.bc_offline = bc_offline
        if not bc_offline:
            self.buffer_replay        = BufferReplay(max_memory_size = max_memory_size, 
                                                     checkpt_dir = 'logs/exp_rl')
            self.buffer_replay_expert = BufferReplay(max_memory_size = max_memory_size, 
                                                     checkpt_dir = 'logs/exp_expert')
        else:
            self.buffer_replay_expert_offline = BufferReplay(max_memory_size = max_memory_size, checkpt_dir = 'logs/exp_expert_offline')
        self.buffer_replay_hld    = BufferReplay_HLD(max_memory_size = int(max_memory_size/5))
        print("[SUCCESS] initialise memory buffer")
        #initialise batch size
        self.N_batch     = N_batch
        #initialise batch size for hld-net
        self.N_batch_hld = N_batch_hld
        #initialise small constant to prevent zero value
        self.sm_c      = 1e-6
        #initialise learning rate of low-level action learning rate
        self.lr        = lr
        #initialise learning rate of hld learning rate
        self.hld_lr    = hld_lr
        #initialise temperature factor
        self.alpha     = alpha
        #initialise discount factor
        self.gamma     = gamma
        #gripper action weights
        self.gripper_loss_weight = gripper_loss_weight
        #initialise rl-bc learning balancing factor
        self.bc_lambda         = bc_lambda
        self.min_bc_lambda     = min_bc_lambda
        self.bc_lambda_decay   = bc_lambda_decay
        #initialise soft update factor
        self.tau       = tau
        self.tau_hld   = tau_hld
        #initialise history 
        self.r_hist    = []
        self.step_hist = []

        #initialise if debug
        self.is_debug = is_debug

        #initialise save interval
        self.save_all_exp_interval = save_all_exp_interval

        #initialise max step per episode
        self.max_step = max_step

        #initialise check point directory
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)
        self.checkpt_dir = os.path.abspath(checkpt_dir)

        #initialise if the agent start reinforcement learning (rl) 
        self.enable_rl_critic = False
        self.enable_rl_actor  = False

        #initialise if the agent start behaviour cloning (bc) 
        self.enable_bc = False
        #the maximum record length that record if this action is successful or fail
        self.max_result_window = max_result_window

        self.grasp_record_list = [0]*self.max_result_window
        self.push_record_list  = [0]*self.max_result_window
        self.grasp_record_index = 0
        self.push_record_index  = 0

        self.best_grasp_success_rate = -np.inf
        self.best_push_success_rate  = -np.inf
        self.grasp_success_rate_hist = []
        self.push_success_rate_hist  = []

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
        
        #TODO [FINISH 29 AUG 2024]: yaw angle and gripper should not turn into image size in this stage

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

        #turn gripper state into image
        #GRIPPER_NON_CLOSE_NON_OPEN = 2, largest value
        max_gripper_action = constants.GRIPPER_CANNOT_OPERATE 
        in_gripper_state = None
        if gripper_state is not None and not np.isnan(gripper_state):
            in_gripper_state = gripper_state/max_gripper_action

        #turn yaw ang into image
        in_yaw_ang = None
        if yaw_ang is not None and not np.isnan(yaw_ang):
            in_yaw_ang = utils.wrap_ang(yaw_ang)/np.math.pi

        if self.is_debug:
            if in_gripper_state is not None and (in_gripper_state < 0 or in_gripper_state > 1):
                print("[ERROR] in_gripper_state < 0 or in_gripper_state > 1")
            if in_yaw_ang is not None and (in_yaw_ang < -1 or in_yaw_ang > 1):
                print("[ERROR] in_yaw_ang < -1 or in_yaw_ang > 1")
            if np.min(in_depth_img) < 0 or np.max(in_depth_img) > 1:
                print("[ERROR] np.min(in_depth_img) < 0 or np.max(in_depth_img) > 1") 

        return in_depth_img, in_gripper_state, in_yaw_ang

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
        #                 5,     6,          7,          8,               9,              10
        # next_action_types, dones, predict_qs, labeled_qs, predict_next_qs, labeled_next_qs

        #split experience into training (80%) and valuation (20%)
        indices       = np.arange(len(exp[0]))
        train_size    = int(train_ratio*len(indices))
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        #turn experience into compatible inputs to grasp network
        # state, next_state, action_type, reward, done, next_q_value
        states            = []
        next_states       = []
        action_types      = []
        next_action_types = []
        rewards           = []
        dones             = []

        # states, target_action_types, rewards, next_q_values 
        for i in range(len(indices)):
            #get state
            depth_img, gripper_state, yaw_ang = self.preprocess_state(depth_img     = exp[1][i], 
                                                                      gripper_state = None, 
                                                                      yaw_ang       = None)
            
            states.append(depth_img)

            next_depth_img, next_gripper_state, next_yaw_ang = self.preprocess_state(depth_img     = exp[4][i], 
                                                                                     gripper_state = None, 
                                                                                     yaw_ang       = None)
            
            next_states.append(next_depth_img)

            #get action types
            action_type = copy.copy(exp[2][i])
            action_types.append(action_type)

            next_action_type = copy.copy(exp[5][i])
            next_action_types.append(next_action_type)

            #get reward
            reward = copy.copy(exp[3][i])
            rewards.append(reward)

            #get dones
            done = copy.copy(exp[6][i])
            dones.append(done)

        states             = torch.FloatTensor(np.array(states))
        next_states        = torch.FloatTensor(np.array(next_states))
        action_types       = torch.FloatTensor(np.array(action_types))
        next_action_types  = torch.FloatTensor(np.array(next_action_types))
        rewards            = torch.FloatTensor(np.array(rewards))
        dones              = torch.FloatTensor(np.array(dones))

        dataset  = DemoDatasetHLDNet(states, next_states, action_types, next_action_types, rewards, dones)

        #create subset
        train_subset = Subset(dataset, train_indices)
        test_subset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size = self.N_batch_hld, shuffle = True, 
                                  drop_last=True if train_size > self.N_batch else False)
        test_loader  = DataLoader( test_subset, batch_size = self.N_batch_hld, shuffle = False)

        return train_loader, test_loader

    def get_train_test_dataloader(self,
                                  exp, 
                                  train_ratio = 0.9, 
                                  is_grasp = True):

        #            0,          1,            2,              3,          4,             
        # action_index, priorities, depth_states, gripper_states, yaw_states, 
        #       5,               6,            7,                    8,
        # actions, gripper_actions, next_actions, next_gripper_actions, 
        #       9,                10,                  11,              12,
        # rewards, next_depth_states, next_gripper_states, next_yaw_states,
        #    13,         14,         15,           16                      
        # dones, predict_qs, labeled_qs, success_mask 

        #split experience into training and valuation 
        indices       = np.arange(len(exp[0]))
        train_size    = int(train_ratio*len(indices))
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        #turn experience into compatible inputs to grasp network

        depth_imgs          = np.zeros((len(indices), 128, 128))
        next_depth_imgs     = np.zeros((len(indices), 128, 128))
        gripper_states      = np.zeros((len(indices), 1))
        next_gripper_states = np.zeros((len(indices), 1))
        yaw_angs            = np.zeros((len(indices), 1))
        next_yaw_angs       = np.zeros((len(indices), 1))
        actions             = np.zeros((len(indices), 5 if is_grasp else 4))
        next_actions        = np.zeros((len(indices), 5 if is_grasp else 4))
        
        for i in range(len(indices)):
            # #get state and next state
            depth_imgs[i], gripper_states[i], yaw_angs[i] = self.preprocess_state(depth_img     = exp[2][i], 
                                                                                  gripper_state = exp[3][i], 
                                                                                  yaw_ang       = exp[4][i])


            next_depth_imgs[i], next_gripper_states[i], next_yaw_angs[i] = self.preprocess_state(depth_img     = exp[10][i], 
                                                                                                 gripper_state = exp[11][i], 
                                                                                                 yaw_ang       = exp[12][i])
            
            #get action and next action
            if is_grasp:
                action      = np.hstack((exp[5][i], exp[6][i]))
                next_action = np.hstack((exp[7][i], exp[8][i]))
            else:
                action      = exp[5][i]
                next_action = exp[7][i]

            actions[i] = copy.copy(action)
            next_actions[i] = copy.copy(next_action)

        depth_imgs          = torch.FloatTensor(depth_imgs)
        gripper_states      = torch.FloatTensor(gripper_states)
        yaw_angs            = torch.FloatTensor(yaw_angs)
        next_depth_imgs     = torch.FloatTensor(next_depth_imgs)
        next_gripper_states = torch.FloatTensor(next_gripper_states)
        next_yaw_angs       = torch.FloatTensor(next_yaw_angs)
        actions             = torch.FloatTensor(actions)
        next_actions        = torch.FloatTensor(next_actions)
        rewards             = torch.FloatTensor(exp[9])
        dones               = torch.FloatTensor(exp[13])
        success_masks       = torch.FloatTensor(exp[16])

        dataset  = DemoDataset(depth_imgs, next_depth_imgs, 
                               gripper_states, next_gripper_states, 
                               yaw_angs, next_yaw_angs, 
                               actions, next_actions, 
                               rewards, dones, success_masks)
        
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

        #set target q network in eval mode
        hld_net_target.eval()

        for epoch in range(num_epochs):
            mean_hld_loss   = 0.

            batch_cnt = 0

            #set network in training mode
            hld_net.train()
               # state,  next_state,         action_type,        next_action_types,  reward,  done
            for states, next_states, target_action_types, next_target_action_types, rewards, dones in train_loader:            

                states                   = states.to(self.device)
                next_states              = next_states.to(self.device)
                target_action_types      = target_action_types.to(self.device)
                next_target_action_types = next_target_action_types.to(self.device)
                rewards                  = rewards.unsqueeze(1).to(self.device)
                dones                    = dones.unsqueeze(1).to(self.device)

                # compute current q - values
                batch_indices = torch.arange(states.size(0)).long().to(self.device)
                qs = hld_net(states)[batch_indices, target_action_types.long()]
                qs = qs.unsqueeze(1)

                # Compute target q - values
                with torch.no_grad():
                    # next_qs   = torch.max(hld_net_target(next_states), dim=1)[0]
                    next_qs   = hld_net_target(next_states)[batch_indices, next_target_action_types.long()]
                    next_qs   = next_qs.unsqueeze(1)
                    target_qs = rewards + self.gamma * (1 - dones) * next_qs

                if self.is_debug:
                    if len(qs.shape) != len(target_qs.shape):
                        print("[ERROR] len(qs.shape) != len(target_qs.shape)")
                        break
                    else:
                        if qs.shape[0] != target_qs.shape[0]:
                            print("[ERROR] qs.shape[0] != target_qs.sahpe[0]")
                            break
                        elif qs.shape[1] != target_qs.shape[1]:
                            print("[ERROR] qs.shape[1] != target_qs.sahpe[1]")
                            break

                hld_net_loss = mse_loss(qs, target_qs)

                # update hld-net
                hld_optimizer.zero_grad()
                hld_net_loss.backward()
                hld_optimizer.step()

                mean_hld_loss += hld_net_loss.item()
            
                batch_cnt += 1

                #soft update target network 
                self.soft_update(critic = hld_net, target_critic = hld_net_target, tau = self.tau_hld)
        
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
        
        #set target q network in eval mode
        hld_net_target.eval()

        #initialise loss funciton
        mse_loss  = nn.MSELoss()

        #initialise mean loss value
        mean_hld_loss = 0.

        #initialise batch counter
        batch_cnt = 0

        with torch.no_grad():

            for states, next_states, target_action_types, next_target_action_types, rewards, dones in test_loader:            

                states                   = states.to(self.device)
                next_states              = next_states.to(self.device)
                target_action_types      = target_action_types.to(self.device)
                next_target_action_types = next_target_action_types.to(self.device)
                rewards                  = rewards.unsqueeze(1).to(self.device)
                dones                    = dones.unsqueeze(1).to(self.device)

                # compute current q - values
                batch_indices = torch.arange(states.size(0)).long().to(self.device)
                qs = hld_net(states)[batch_indices, target_action_types.long()]
                qs = qs.unsqueeze(1)

                # Compute target q - values
                with torch.no_grad():
                    # next_qs   = torch.max(hld_net_target(next_states), dim=1)[0]
                    next_qs   = hld_net_target(next_states)[batch_indices, next_target_action_types.long()]
                    next_qs   = next_qs.unsqueeze(1)
                    target_qs = rewards + self.gamma * (1 - dones) * next_qs

                if self.is_debug:
                    if len(qs.shape) != len(target_qs.shape):
                        print("[ERROR] len(qs.shape) != len(target_qs.shape)")
                        break
                    else:
                        if qs.shape[0] != target_qs.shape[0]:
                            print("[ERROR] qs.shape[0] != target_qs.sahpe[0]")
                            break
                        elif qs.shape[1] != target_qs.shape[1]:
                            print("[ERROR] qs.shape[1] != target_qs.sahpe[1]")
                            break

                hld_net_loss = mse_loss(qs, target_qs)

                mean_hld_loss += hld_net_loss.item()
            
                batch_cnt += 1

        mean_hld_loss /= batch_cnt

        return mean_hld_loss

    def behaviour_cloning(self, train_loader, test_loader, critic1, critic2, target_critic1, target_critic2, actor, num_epochs=500, is_grasp = True):

        #set target_critic in eval mode
        target_critic1.eval()
        target_critic2.eval()

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
        if is_grasp:
            max_action = constants.MAX_ACTION
        else:
            max_action = constants.PUSH_MAX_ACTION

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

            for state, next_state, target_action, next_target_action, action_state, next_action_state, reward, done, success_mask in train_loader:

                state              = state.to(self.device)
                next_state         = next_state.to(self.device)
                target_action      = target_action.to(self.device)
                next_target_action = next_target_action.to(self.device)
                action_state       = action_state.to(self.device)
                next_action_state  = next_action_state.to(self.device)
                reward             = reward.unsqueeze(1).to(self.device)
                done               = done.unsqueeze(1).to(self.device)
                success_mask       = success_mask.to(self.device)

                #compute normalisation factor
                if is_grasp:
                    normalise_factor = torch.tensor(max_action + [1.]).to(self.device) 
                else:
                    normalise_factor = torch.tensor(max_action).to(self.device)

                #normalise target action
                normalise_target_action = target_action/normalise_factor.view(1, normalise_factor.shape[0])

                #forward pass through both critics
                with torch.no_grad():
                    next_q1  = target_critic1(next_action_state)
                    next_q2  = target_critic2(next_action_state)
                    next_q   = torch.min(next_q1, next_q2)
                    target_q = reward + (1 - done)*self.gamma*next_q
 
                q1 = critic1(action_state)
                q2 = critic2(action_state)
                
                # Use the actor to get the predicted actions
                actions, normalised_actions, gripper_action, z, normal, gripper_action_probs = actor.get_actions(state)

                # Compute loss for both critics
                if self.is_debug:
                    if len(q1.shape) != len(target_q.shape):
                        print("[ERROR] len(q1.shape) != len(target_q.shape)")
                        break
                    else:
                        if q1.shape[0] != target_q.shape[0]:
                            print("[ERROR] q1.shape[0] != target_q.sahpe[0]")
                            break
                        elif q1.shape[1] != target_q.shape[1]:
                            print("[ERROR] q1.shape[1] != target_q.sahpe[1]")
                            break

                    if len(q2.shape) != len(target_q.shape):
                        print("[ERROR] len(q2.shape) != len(target_q.shape)")
                        break
                    else:
                        if q2.shape[0] != target_q.shape[0]:
                            print("[ERROR] q2.shape[0] != target_q.sahpe[0]")
                            break
                        elif q2.shape[1] != target_q.shape[1]:
                            print("[ERROR] q2.shape[1] != target_q.sahpe[1]")
                            break

                critic1_loss = mse_loss(q1, target_q)
                critic2_loss = mse_loss(q2, target_q)

                # compute actor loss 
                if self.is_debug:
                    if len(normalised_actions.shape) != len(normalise_target_action[:,0:4].shape):
                        print("[ERROR] len(normalised_actions.shape) != len(normalise_target_action.shape)")
                        break
                    else:
                        if normalised_actions.shape[0] != normalise_target_action[:,0:4].shape[0]:
                            print("[ERROR] normalised_actions.shape[0] != normalise_target_action.sahpe[0]")
                            break
                        elif normalised_actions.shape[1] != normalise_target_action[:,0:4].shape[1]:
                            print("[ERROR] normalised_actions.shape[1] != normalise_target_action.sahpe[1]")
                            break

                move_loss  = mse_loss_no_reduce(normalised_actions.float(), normalise_target_action[:,0:4].float())
                move_loss  = torch.mean(move_loss, dim=1)* success_mask

                actor_loss = move_loss
                if is_grasp:

                    target_gripper_action_class = target_action[:, -1].long()
                    # print(target_gripper_action_class)
                    gripper_action_loss = ce_loss_no_reduce(gripper_action_probs, target_gripper_action_class)* success_mask *self.gripper_loss_weight
                    actor_loss += gripper_action_loss

                    if self.is_debug:
                        if len(gripper_action_probs.shape) != 2:
                            print("[ERROR] len(gripper_action_probs.shape) != 2")
                            break
                        else:
                            if gripper_action_probs.shape[0] != target_gripper_action_class.shape[0]:
                                print("[ERROR] gripper_action_probs.shape[0] != target_gripper_action_class.sahpe[0]")
                                break
                
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

                #soft update target critic network
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

        #set target_critic in eval mode
        target_critic1.eval()
        target_critic2.eval()

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
        if is_grasp:
            max_action = constants.MAX_ACTION
        else:
            max_action = constants.PUSH_MAX_ACTION

        with torch.no_grad():

            for state, next_state, target_action, next_target_action, action_state, next_action_state, reward, done, success_mask in test_loader:

                state              = state.to(self.device)
                next_state         = next_state.to(self.device)
                target_action      = target_action.to(self.device)
                next_target_action = next_target_action.to(self.device)
                action_state       = action_state.to(self.device)
                next_action_state  = next_action_state.to(self.device)
                reward             = reward.unsqueeze(1).to(self.device)
                done               = done.unsqueeze(1).to(self.device)
                success_mask       = success_mask.to(self.device)

                #compute normalisation factor
                if is_grasp:
                    normalise_factor    = torch.tensor(max_action + [1.]).to(self.device) 
                else:
                    normalise_factor    = torch.tensor(max_action).to(self.device)

                #compute action state
                normalise_target_action = target_action/normalise_factor.view(1, normalise_factor.shape[0])

                #forward pass through both critics
                with torch.no_grad():
                    next_q1  = target_critic1(next_action_state)
                    next_q2  = target_critic2(next_action_state)
                    next_q   = torch.min(next_q1, next_q2)
                    target_q = reward + (1 - done)*self.gamma*next_q
 
                q1 = critic1(action_state)
                q2 = critic2(action_state)

                # Use the actor to get the predicted actions
                actions, normalised_actions, gripper_action, z, normal, gripper_action_probs = actor.get_actions(state)

                # Compute loss for both critics

                if self.is_debug:
                    if len(q1.shape) != len(target_q.shape):
                        print("[ERROR] len(q1.shape) != len(target_q.shape)")
                        break
                    else:
                        if q1.shape[0] != target_q.shape[0]:
                            print("[ERROR] q1.shape[0] != target_q.sahpe[0]")
                            break
                        elif q1.shape[1] != target_q.shape[1]:
                            print("[ERROR] q1.shape[1] != target_q.sahpe[1]")
                            break

                    if len(q2.shape) != len(target_q.shape):
                        print("[ERROR] len(q2.shape) != len(target_q.shape)")
                        break
                    else:
                        if q2.shape[0] != target_q.shape[0]:
                            print("[ERROR] q2.shape[0] != target_q.sahpe[0]")
                            break
                        elif q2.shape[1] != target_q.shape[1]:
                            print("[ERROR] q2.shape[1] != target_q.sahpe[1]")
                            break

                critic1_loss = mse_loss(q1, target_q)
                critic2_loss = mse_loss(q2, target_q)

                # compute actor loss 
                if self.is_debug:
                    if len(normalised_actions.shape) != len(normalise_target_action[:,0:4].shape):
                        print("[ERROR] len(normalised_actions.shape) != len(normalise_target_action.shape)")
                        break
                    else:
                        if normalised_actions.shape[0] != normalise_target_action[:,0:4].shape[0]:
                            print("[ERROR] normalised_actions.shape[0] != normalise_target_action.sahpe[0]")
                            break
                        elif normalised_actions.shape[1] != normalise_target_action[:,0:4].shape[1]:
                            print("[ERROR] normalised_actions.shape[1] != normalise_target_action.sahpe[1]")
                            break

                move_loss  = mse_loss_no_reduce(normalised_actions.float(), normalise_target_action[:,0:4].float())
                move_loss  = torch.mean(move_loss, dim=1)* success_mask

                actor_loss = move_loss
                if is_grasp:

                    target_gripper_action_class = target_action[:, -1].long()
                    gripper_action_loss = ce_loss_no_reduce(gripper_action_probs, target_gripper_action_class)* success_mask *self.gripper_loss_weight
                    actor_loss += gripper_action_loss

                    if self.is_debug:
                        if len(gripper_action_probs.shape) != 2:
                            print("[ERROR] len(gripper_action_probs.shape) != 2")
                            break
                        else:
                            if gripper_action_probs.shape[0] != target_gripper_action_class.shape[0]:
                                print("[ERROR] gripper_action_probs.shape[0] != target_gripper_action_class.sahpe[0]")
                                break
                
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

    def gather_guidance_experience(self):

        if self.buffer_replay_expert_offline.is_full:
            print("[SUCCESS] buffer is full")
            return

        #start trainiing/evaluation loop
        episode = 0
        while not self.buffer_replay_expert_offline.is_full:

            #initialise episode data
            step = 0
            ep_r = 0.
            episode_done         = False
            action_type          = None
            previous_action_type = None
            is_success_grasp     = False
            is_env_reset         = None

            #initialise hld-net experience in one episode

            hld_depth_states      = []
            hld_action_types      = []
            hld_next_action_types = []
            hld_rewards           = []
            hld_next_depth_states = []
            hld_dones             = []

            #reset environment
            self.env.reset(reset_item = True)

            while not episode_done and step < self.max_step and not self.buffer_replay_expert_offline.is_full:

                print(f"==== episode: {episode} ====")

                self.env.return_home(is_env_reset, previous_action_type)
                is_env_reset = False
                #action selection
                target_item_pos = None
                delta_moves_grasp, target_item_pos_grasp = self.env.grasp_guidance_generation()
                delta_moves_push, target_item_pos_push  = self.env.push_guidance_generation()

                
                delta_moves = []

                #no grasp move or (previous grasp is fail and have push move)
                if len(delta_moves_grasp) == 0:
                    print("[ACTION CHOICE] no grasp options => try to push!")
                    action_type = constants.PUSH
                    delta_moves = delta_moves_push
                    target_item_pos = target_item_pos_push
                elif (previous_action_type == constants.GRASP and not is_success_grasp and len(delta_moves_push) > 0):
                    print("[ACTION CHOICE] try to push after grasping fails")
                    action_type = constants.PUSH
                    delta_moves = delta_moves_push
                    target_item_pos = target_item_pos_push
                else:
                    print("[ACTION CHOICE] grasp!")
                    action_type = constants.GRASP
                    delta_moves = delta_moves_grasp
                    target_item_pos = target_item_pos_grasp

                if len(delta_moves) == 0:
                    episode_done = True

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
                action_dones         = []
                
                for i in range(len(delta_moves)):

                    print(f"==== step: {step} N_pickable_item: {self.env.N_pickable_item} ====")

                    #get raw data
                    depth_img, gripper_state, yaw_state = self.get_raw_data(action_type)

                    if self.is_debug:
                        print("[SUCCESS] get raw data")
                    
                    #action selection
                    move   = np.array(delta_moves[i])
                    n_move = np.array(delta_moves[i+1] if i+1 < len(delta_moves) else [0,0,0,0,move[-2],move[-1]])

                    action,  gripper_action = np.array(move[0:self.N_action]), np.argmax(np.array(move[-2:]))
                    next_action, next_gripper_action = np.array(n_move[0:self.N_action]), np.argmax(np.array(n_move[-2:]))

                    gripper_action = torch.FloatTensor([gripper_action]).unsqueeze(0).to(self.device)
                    action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    
                    next_gripper_action = torch.FloatTensor([next_gripper_action]).unsqueeze(0).to(self.device)
                    next_action = torch.FloatTensor(next_action).unsqueeze(0).to(self.device)

                    if self.is_debug:
                        print("[SUCCESS] estimate actions from guidance") 

                    #step
                    reward, is_success_grasp = self.env.step(action_type, 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][0:3], 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][3], 
                                                             True if move[4] == 1 else False, 
                                                             target_item_pos)

                    #print actions
                    if self.is_debug:
                        print(f"[STEP]: {step} [ACTION TYPE]: {action_type} [REWARD]: {reward}")  
                        print(f"[MOVE]: {action.to(torch.device('cpu')).detach().numpy()[0]}") 

                    #update history
                    ep_r += reward
                    step += 1

                    #check if all items are picked
                    episode_done = False if self.env.N_pickable_item > 0 else True
                    if (i == len(delta_moves) - 1 or 
                        self.env.is_out_of_working_space or 
                        not self.env.can_execute_action or 
                        self.env.is_collision_to_ground or
                        self.env.gripper_cannot_operate):
                        action_done = True
                    else:
                        action_done = False 

                    #get raw data after actions
                    next_depth_img, next_gripper_state, next_yaw_state = self.get_raw_data(action_type)

                    if self.is_debug:
                        print("[SUCCESS] get next raw data")

                    # store experience during executing low-level action
                    depth_states.append(depth_img)
                    gripper_states.append(gripper_state)
                    yaw_states.append(yaw_state)

                    next_depth_states.append(next_depth_img)
                    next_gripper_states.append(next_gripper_state)
                    next_yaw_states.append(next_yaw_state)

                    actions.append(action.to(torch.device('cpu')).detach().numpy()[0])
                    gripper_actions.append(gripper_action.cpu())
                    next_actions.append(next_action.to(torch.device('cpu')).detach().numpy()[0])
                    next_gripper_actions.append(next_gripper_action.cpu())

                    action_types.append(action_type)
                    rewards.append(reward)
                    action_dones.append(action_done)

                    #check if episode_done
                    if episode_done:
                        print("[SUCCESS] finish one episode")
                        self.r_hist.append(ep_r)
                        self.step_hist.append(step)         
                        break 
                    else:

                        #return home position if grasp successfully
                        if is_success_grasp or i == len(delta_moves) - 1:
                            print("[SUCCESS] finish actions or grasp an item")

                        #check if out of working space
                        elif self.env.is_out_of_working_space:
                            print("[WARN] out of working space")
                            self.env.reset(reset_item = False)
                            is_env_reset = True

                        #check if action executable
                        elif not self.env.can_execute_action:
                            print("[WARN] action is not executable")
                            self.env.reset(reset_item = False)
                            is_env_reset = True

                        #check if collision to ground
                        elif self.env.is_collision_to_ground:
                            print("[WARN] collision to ground")
                            self.env.reset(reset_item = False)
                            is_env_reset = True

                        #check if gripper operate properly
                        elif self.env.gripper_cannot_operate:
                            print("[WARN] gripper cannot function properly")
                            self.env.reset(reset_item = False)
                            is_env_reset = True

                    #this set of motions is not executable => break it
                    if (self.env.is_out_of_working_space or 
                        not self.env.can_execute_action or
                        self.env.is_collision_to_ground or
                        self.env.gripper_cannot_operate):
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
                action_dones         = np.array(action_dones)

                for i in range(len(depth_states)): 
                    self.buffer_replay_expert_offline.store_transition(depth_states[i], 
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
                                                                        action_dones[i], 
                                                                        0., 0.,
                                                                        True if (rewards > 0).sum() > 0 else False, 
                                                                        0., 0.) 
                    
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
                        hld_reward += 1.0

                if self.env.N_pickable_item == 0:
                    hld_reward += 1.0

                #compute hld done
                if (self.env.N_pickable_item == 0 or step >= self.max_step or self.buffer_replay_expert_offline.is_full):
                    hld_done = True
                else:
                    hld_done = False

                print(f"hld_reward: {hld_reward}, hld_done: {hld_done}, N_pickable_item: {self.env.N_pickable_item}")
                print(f"N_action_step: {len(depth_states)}")

                #store hld-net data after completing one set of action
                if len(depth_states) > 0:
                    hld_depth_states.append(depth_states[0])
                    hld_action_types.append(action_type)
                    hld_rewards.append(hld_reward)
                    hld_next_depth_states.append(next_depth_states[-1])
                    hld_dones.append(hld_done)
                    print('[SUCCESS] store hld-net data')

                #reset any items out of working space
                self.env.reset_item2workingspace()    
                time.sleep(0.5) 
                
                if self.is_debug:
                    print('[SUCCESS] reset items to workplaces if any')
                
            #update hld-net q values
            for i in range(len(hld_action_types) - 1):
                hld_next_action_types.append(hld_action_types[i+1])
            #the last next action is meaningless, no matter which action type is fine
            hld_next_action_types.append(constants.GRASP)
            hld_depth_states      = np.array(hld_depth_states)
            hld_action_types      = np.array(hld_action_types)
            hld_rewards           = np.array(hld_rewards)
            hld_next_depth_states = np.array(hld_next_depth_states)
            hld_next_action_types = np.array(hld_next_action_types)
            hld_dones             = np.array(hld_dones)

            #store transition experience of high-level behaviour
            for i in range(len(hld_rewards)):
                self.buffer_replay_hld.store_transition(hld_depth_states[i],
                                                        hld_action_types[i],
                                                        hld_rewards[i],
                                                        hld_next_depth_states[i],
                                                        hld_next_action_types[i],
                                                        hld_dones[i],
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.)
            if self.is_debug:
                print('[SUCCESS] store transition experience for high-level decision')
            
            #update episode
            episode += 1

        print("[SUCCESS] buffer is full")

    def interact_test(self,
                      max_episode = 1,
                      is_train    = True):

        self.is_train = is_train

        #start trainiing/evaluation loop
        episode = 0

        #load check point
        # self.hld_net.load_checkpoint()

        try:
            #load agent data
            self.load_agent_data()
            print("[SUCCESS] load agent data") 
        except:
            print("[FAIL] load agent data") 

        self.load_models()

        while episode < max_episode:

            #initialise episode data
            step = 0
            ep_r = 0.
            episode_done         = False
            action_type          = None
            previous_action_type = None
            is_success_grasp     = False
            grasp_fail_counter   = 0
            push_fail_counter    = 0

            #reset environment
            self.env.reset(reset_item = True)
            is_env_reset = True

            while not episode_done and step < self.max_step:

                print(f"==== episode: {episode} ====")


                # #get raw data
                # depth_img, _, _ = self.get_raw_data(action_type)

                # #get state (no use in demo gathering)
                # depth_img, _, _ = self.preprocess_state(depth_img     = depth_img, 
                #                                         gripper_state = None, 
                #                                         yaw_ang       = None)
                
                # hld_state = torch.FloatTensor(depth_img).unsqueeze(0).unsqueeze(0)

                # self.hld_net.eval()
                # with torch.no_grad():
                #     action_type = self.hld_net.make_decisions(hld_state)

                self.env.return_home(is_env_reset, previous_action_type)
                is_env_reset = False

                if self.is_debug:
                    print("[SUCCESS] return home")

                time.sleep(0.5) 

                #reset any items out of working space
                self.env.reset_item2workingspace()    

                if self.is_debug:
                    print("[SUCCESS] reset items if any")

                time.sleep(0.5) 

                #demo action generation
                delta_moves, target_item_pos, action_type = self.env.demo_guidance_generation()

                #decide if it should enter demo mode
                self.enable_bc = True
                self.enable_rl_critic = True
                self.enable_rl_actor  = True

                if is_train and action_type == constants.PUSH and push_fail_counter >= 1:
                    expert_mode = True
                    buffer_replay = self.buffer_replay_expert
                    push_fail_counter = 0
                elif is_train and action_type == constants.GRASP and grasp_fail_counter >= 1:
                    expert_mode = True
                    buffer_replay = self.buffer_replay_expert
                    grasp_fail_counter = 0
                else:
                    expert_mode = False
                    buffer_replay = self.buffer_replay

                # if is_train and \
                #     (self.buffer_replay_expert.grasp_data_size < self.N_batch * 5 or \
                #      self.buffer_replay_expert.push_data_size  < self.N_batch * 5):
                #     expert_mode = True
                #     buffer_replay = self.buffer_replay_expert
                # elif is_train and action_type == constants.PUSH and push_fail_counter >= 1:
                #     expert_mode = True
                #     buffer_replay = self.buffer_replay_expert
                #     push_fail_counter = 0
                # elif is_train and action_type == constants.GRASP and grasp_fail_counter >= 1:
                #     expert_mode = True
                #     buffer_replay = self.buffer_replay_expert
                #     grasp_fail_counter = 0
                # else:
                #     expert_mode = False
                #     buffer_replay = self.buffer_replay

                # #decide if it should start bc
                # if (self.buffer_replay_expert.grasp_data_size >= self.N_batch and
                #     self.buffer_replay_expert.push_data_size  >= self.N_batch):
                #     self.enable_bc = True                
                #     #decide if actor should start rl
                #     if (self.buffer_replay.grasp_data_size >= self.N_batch and 
                #         self.buffer_replay.push_data_size  >= self.N_batch):
                #         self.enable_rl_critic = True

                #     if (self.buffer_replay.grasp_data_size >= 10*self.N_batch and 
                #         self.buffer_replay.push_data_size  >= 10*self.N_batch):
                #         self.enable_rl_actor = True

                if self.env.N_pickable_item <= 0:
                    episode_done = True
                    continue

                if action_type == constants.GRASP:
                    if expert_mode:
                        N_step_low_level = len(delta_moves)
                    else:
                        N_step_low_level = self.N_grasp_step
                else:
                    if expert_mode:
                        N_step_low_level = len(delta_moves)
                    else:    
                        N_step_low_level = self.N_push_step
                
                print(f"N_step_low_level: {N_step_low_level}")

                #initialise memory space for storing a complete set of actions
                depth_states           = []
                gripper_states         = []
                yaw_states             = []
                actions                = []
                gripper_actions        = []
                next_actions           = []
                next_gripper_actions   = []
                action_types           = []
                rewards                = []
                next_depth_states      = []
                next_gripper_states    = []
                next_yaw_states        = []
                action_dones           = []
                actor_losses           = []
                critic_losses          = []

                for i in range(N_step_low_level):
                    
                    update_mode_msg = f"ENABLE BC: {self.enable_bc} ENABLE RL CRITIC: {self.enable_rl_critic} ENABLE RL ACTOR: {self.enable_rl_actor}"
                    if expert_mode:
                        print("==== EXPERT MODE " + update_mode_msg + " ====") 
                    else:
                        print("==== AGENT MODE " + update_mode_msg + " ====") 

                    print(f"==== step: {step} N_pickable_item: {self.env.N_pickable_item} ====")

                    #get raw data
                    depth_img, gripper_state, yaw_state = self.get_raw_data(action_type)

                    if self.is_debug:
                        print("[SUCCESS] get raw data")

                    #preprocess raw data
                    in_depth_img, in_gripper_state, in_yaw_state = self.preprocess_state(depth_img     = depth_img, 
                                                                                         gripper_state = gripper_state, 
                                                                                         yaw_ang       = yaw_state, 
                                                                                         is_grasp      = action_type) 

                    if self.is_debug:
                        print("[SUCCESS] preprocess raw data")

                    #estimate actions
                    if action_type == constants.GRASP:
                        #estimate action from network
                        self.grasp_actor.eval()
                        with torch.no_grad():
                            state = torch.concatenate([torch.FloatTensor(in_depth_img).unsqueeze(0), 
                                                       torch.FloatTensor([in_gripper_state]).expand(128, 128).unsqueeze(0), 
                                                       torch.FloatTensor([in_yaw_state]).expand(128, 128).unsqueeze(0)], dim=0).unsqueeze(0)
                            action_est, normalised_action_est, gripper_action_est, z, normal, gripper_action_probs_est = self.grasp_actor.get_actions(state)
                    else:
                        #estimate action from network
                        self.push_actor.eval()
                        with torch.no_grad():
                            state = torch.concatenate([torch.FloatTensor(in_depth_img).unsqueeze(0), 
                                                       torch.FloatTensor([in_yaw_state]).expand(128, 128).unsqueeze(0)], dim=0).unsqueeze(0)
                            action_est, normalised_action_est, gripper_action_est, z, normal, gripper_action_probs_est = self.push_actor.get_actions(state)

                    if self.is_debug:
                        print("[SUCCESS] estimate actions from network") 

                    #action from demo
                    if expert_mode:
                        move = np.array(delta_moves[i])
                        action_demo, gripper_action_demo = np.array(move[0:self.N_action]), np.argmax(np.array(move[-2:]))
                        gripper_action_demo    = torch.FloatTensor([gripper_action_demo]).unsqueeze(0).to(self.device)
                        action_demo            = torch.FloatTensor(action_demo).unsqueeze(0).to(self.device)

                        if self.is_debug:
                            print("[SUCCESS] estimate actions from guidance") 

                        if action_type == constants.GRASP:
                            normalised_action_demo = action_demo/torch.FloatTensor(constants.MAX_ACTION).view(1, len(constants.MAX_ACTION)).to(self.device)
                        else:
                            normalised_action_demo = action_demo/torch.FloatTensor(constants.PUSH_MAX_ACTION).view(1, len(constants.PUSH_MAX_ACTION)).to(self.device)

                        action, normalised_action, gripper_action = action_demo, normalised_action_demo, gripper_action_demo

                        if self.is_debug:
                            print("[SUCCESS] normalise guidance action") 
                    else:
                        action, normalised_action, gripper_action =  action_est, normalised_action_est, gripper_action_est

                    #compute action state and current q value
                    if action_type == constants.GRASP:
                        self.grasp_critic1.eval()
                        self.grasp_critic2.eval()
                        with torch.no_grad():
                            action_state = torch.concatenate([state[0], 
                                                            torch.FloatTensor([normalised_action[0][0]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([normalised_action[0][1]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([normalised_action[0][2]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([normalised_action[0][3]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([gripper_action[0]]).expand(128, 128).unsqueeze(0)],
                                                            dim = 0).unsqueeze(0).to(self.device)        

                            current_q1 = self.grasp_critic1(action_state)
                            current_q2 = self.grasp_critic2(action_state)     
                    else:
                        self.push_critic1.eval()
                        self.push_critic2.eval()
                        with torch.no_grad():
                            action_state = torch.concatenate([state[0], 
                                                            torch.FloatTensor([normalised_action[0][0]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([normalised_action[0][1]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([normalised_action[0][2]]).expand(128, 128).unsqueeze(0),
                                                            torch.FloatTensor([normalised_action[0][3]]).expand(128, 128).unsqueeze(0)],
                                                            dim = 0).unsqueeze(0).to(self.device)

                            current_q1 = self.push_critic1(action_state)
                            current_q2 = self.push_critic2(action_state)

                    if self.is_debug:
                        print("[SUCCESS] compute current q value") 

                    #interact with env
                    reward, is_success_grasp = self.env.step(action_type, 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][0:3], 
                                                             action.to(torch.device('cpu')).detach().numpy()[0][3], 
                                                             True if gripper_action[0].item() == constants.GRASP else False,
                                                             target_item_pos)

                    #print actions

                    print(f"[STEP]: {step} [ACTION TYPE]: {action_type}")  
                    move_msg  = f"[MOVE] xyz: {action.to(torch.device('cpu')).detach().numpy()[0][0:3]}"
                    move_msg += f" yaw: {np.rad2deg(action.to(torch.device('cpu')).detach().numpy()[0][3])}"
                    print(move_msg)

                    #update history
                    ep_r += reward
                    step += 1

                    #check if all items are picked
                    episode_done = False if self.env.N_pickable_item > 0 else True
                    if (i == N_step_low_level - 1 or 
                        is_success_grasp or
                        self.env.is_out_of_working_space or 
                        not self.env.can_execute_action or 
                        self.env.is_collision_to_ground or
                        self.env.gripper_cannot_operate):
                        action_done = True
                    else:
                        action_done = False                       

                    #get next raw data
                    next_depth_img, next_gripper_state, next_yaw_state = self.get_raw_data(action_type)

                    if self.is_debug:
                        print("[SUCCESS] get next raw data")

                    #get next state (no use in demo gathering)
                    next_in_depth_img, next_in_gripper_state, next_in_yaw_state = self.preprocess_state(depth_img     = next_depth_img, 
                                                                                                        gripper_state = next_gripper_state, 
                                                                                                        yaw_ang       = next_yaw_state, 
                                                                                                        is_grasp      = action_type)     

                    if self.is_debug:
                        print("[SUCCESS] preprocess next raw data")

                    if action_type == constants.GRASP:
                        #estimate action from network
                        self.grasp_actor.eval()
                        with torch.no_grad():
                            next_state = torch.concatenate([torch.FloatTensor(next_in_depth_img).unsqueeze(0), 
                                                            torch.FloatTensor([next_in_gripper_state]).expand(128, 128).unsqueeze(0), 
                                                            torch.FloatTensor([next_in_yaw_state]).expand(128, 128).unsqueeze(0)], dim=0).unsqueeze(0)
                            next_action_est, next_normalised_action_est, next_gripper_action_est, next_z, next_normal, next_gripper_action_probs_est = self.grasp_actor.get_actions(next_state)
                    else:
                        #estimate action from network
                        self.push_actor.eval()
                        with torch.no_grad():
                            next_state = torch.concatenate([torch.FloatTensor(next_in_depth_img).unsqueeze(0), 
                                                           torch.FloatTensor([next_in_yaw_state]).expand(128, 128).unsqueeze(0)], dim=0).unsqueeze(0)
                            next_action_est, next_normalised_action_est, next_gripper_action_est, next_z, next_normal, next_gripper_action_probs_est = self.push_actor.get_actions(next_state)

                    if self.is_debug:
                        print("[SUCCESS] estimate next actions from network")

                    if expert_mode:

                        #action from demo
                        n_move = np.array(delta_moves[i+1] if i+1 < len(delta_moves) else [0,0,0,0,move[-2],move[-1]])
                        next_action_demo, next_gripper_action_demo = np.array(n_move[0:self.N_action]), np.argmax(np.array(n_move[-2:]))
                        next_gripper_action_demo = torch.FloatTensor([next_gripper_action_demo]).unsqueeze(0).to(self.device)
                        next_action_demo = torch.FloatTensor(next_action_demo).unsqueeze(0).to(self.device)

                        if self.is_debug:
                            print("[SUCCESS] estimate next actions from guidance")

                        if action_type == constants.GRASP:
                            next_normalised_action_demo = next_action_demo/torch.FloatTensor(constants.MAX_ACTION).view(1, len(constants.MAX_ACTION)).to(self.device)
                        else:
                            next_normalised_action_demo = next_action_demo/torch.FloatTensor(constants.PUSH_MAX_ACTION).view(1, len(constants.PUSH_MAX_ACTION)).to(self.device)

                        next_action, next_normalised_action, next_gripper_action = next_action_demo, next_normalised_action_demo, next_gripper_action_demo    

                        if self.is_debug:
                            print("[SUCCESS] normalise next actions from guidance")                        
                    else:
                        next_action, next_normalised_action, next_gripper_action = next_action_est, next_normalised_action_est, next_gripper_action_est

                    #compute next action state and next q value
                    if action_type == constants.GRASP:
                        self.grasp_critic1_target.eval()
                        self.grasp_critic2_target.eval()

                        with torch.no_grad():
                            next_action_state = torch.concatenate([next_state[0], 
                                                                torch.FloatTensor([next_normalised_action[0][0]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_normalised_action[0][1]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_normalised_action[0][2]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_normalised_action[0][3]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_gripper_action[0]]).expand(128, 128).unsqueeze(0)],
                                                                dim = 0).unsqueeze(0).to(self.device)        

                            next_q1 = self.grasp_critic1_target(next_action_state)
                            next_q2 = self.grasp_critic2_target(next_action_state)     
                        
                    else:
                        self.push_critic1_target.eval()
                        self.push_critic2_target.eval()

                        with torch.no_grad():
                            next_action_state = torch.concatenate([next_state[0], 
                                                                torch.FloatTensor([next_normalised_action[0][0]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_normalised_action[0][1]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_normalised_action[0][2]]).expand(128, 128).unsqueeze(0),
                                                                torch.FloatTensor([next_normalised_action[0][3]]).expand(128, 128).unsqueeze(0)],
                                                                dim = 0).unsqueeze(0).to(self.device)

                            next_q1 = self.push_critic1_target(next_action_state)
                            next_q2 = self.push_critic2_target(next_action_state)  

                    next_q   = torch.min(next_q1, next_q2)  
                    target_q = reward + (1 - action_done) * self.gamma * next_q
                    critic1_loss = nn.MSELoss()(current_q1, target_q)
                    critic2_loss = nn.MSELoss()(current_q2, target_q)
                    critic_loss  = (critic1_loss + critic2_loss)/2.

                    #compute bc loss
                    if expert_mode:
                        bc_loss_no_reduce = nn.MSELoss(reduction = 'none')(normalised_action_est.float(), normalised_action.float())
                        bc_loss_no_reduce = torch.mean(bc_loss_no_reduce, dim=1)

                        if action_type == constants.GRASP:
                            ce_loss_no_reduce  = nn.CrossEntropyLoss(reduction = 'none')(gripper_action_probs_est, gripper_action.long()[0])*self.gripper_loss_weight
                            bc_loss_no_reduce += ce_loss_no_reduce

                            print(f"[LOSS] ce_loss : {ce_loss_no_reduce.item()}, mse_loss: {bc_loss_no_reduce.item() - ce_loss_no_reduce.item()}")
                        actor_loss = torch.mean(bc_loss_no_reduce, dim = 0)
                    else:
                        actor_loss = None
                        
                    print(f"[LOSS] actor_loss: {actor_loss}, critic_loss: {critic_loss}")

                    # store experience during executing low-level action
                    depth_states.append(depth_img)
                    gripper_states.append(gripper_state)
                    yaw_states.append(yaw_state)

                    next_depth_states.append(next_depth_img)
                    next_gripper_states.append(next_gripper_state)
                    next_yaw_states.append(next_yaw_state)

                    actions.append(action.to(torch.device('cpu')).detach().numpy()[0])
                    gripper_actions.append(gripper_action.cpu())
                    next_actions.append(next_action.to(torch.device('cpu')).detach().numpy()[0])
                    next_gripper_actions.append(next_gripper_action.cpu())       

                    action_types.append(action_type)
                    rewards.append(reward)
                    action_dones.append(action_done)

                    actor_losses.append(actor_loss.cpu() if expert_mode else None)
                    critic_losses.append(critic_loss.to(torch.device('cpu')).detach().numpy())

                    if self.is_debug:
                        print("[SUCCESS] append transition experience")

                    #check if episode_done
                    if episode_done:
                        print("[SUCCESS] finish one episode")
                        self.r_hist.append(ep_r)
                        self.step_hist.append(step)         
                        break 
                    elif action_done:

                        
                        if  is_success_grasp or i == N_step_low_level - 1:
                            if is_success_grasp:
                                print("[SUCCESS] grasp an item")
                            else:
                                print("[SUCCESS] finish an action")
                                
                        #check if out of working space
                        if self.env.is_out_of_working_space:
                            print("[WARN] out of working space")
                        #check if action executable
                        elif not self.env.can_execute_action:
                            print("[WARN] action is not executable")
                        #check if collision to ground
                        elif self.env.is_collision_to_ground:
                            print("[WARN] collision to ground")                        
                        #check if the gripper can operate normally
                        elif self.env.gripper_cannot_operate:
                            print("[WARN] gripper cannot function properly")                        

                        #this set of motions is not executable => break it
                        if (self.env.is_out_of_working_space or 
                            not self.env.can_execute_action or 
                            self.env.is_collision_to_ground or 
                            self.env.gripper_cannot_operate):

                            self.env.reset(reset_item = False)
                            is_env_reset = True
                            print("[WARN] stop executing this action!")
                        
                        break
                
                print("=== end of action ===")

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
                action_dones         = np.array(action_dones)
                actor_losses         = np.array(actor_losses)
                critic_losses        = np.array(critic_losses)

                if not expert_mode and (rewards > 0).sum() <= 0:
                    if action_type == constants.GRASP:
                        grasp_fail_counter += 1
                    else:
                        push_fail_counter += 1                   
                elif not expert_mode and (rewards > 0).sum() > 0:
                    if action_type == constants.GRASP:
                        grasp_fail_counter = 0
                    else:
                        push_fail_counter = 0

                if (rewards > 0).sum() > 0:
                    print("[SUCCESS] GRASP OR PUSH ACTION")
                else:
                    print("[FAIL] GRASP OR PUSH ACTION")

                if is_train:
                    for i in range(len(depth_states)): 
                        #only save successful experience when in expert mode
                        if expert_mode and (rewards > 0).sum() <= 0:
                            continue
                        buffer_replay.store_transition(depth_states[i], 
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
                                                    action_dones[i], 
                                                    0., 
                                                    0.,
                                                    True if (rewards > 0).sum() > 0 else False,
                                                    actor_losses[i], 
                                                    critic_losses[i]) 
                    if self.is_debug:
                        print('[SUCCESS] store transition experience for low-level action')   

                #record if this action is successful or fail
                if is_train:

                    self.online_update(action_type = action_type)

                    if self.is_debug:
                        print('[SUCCESS] online update')

                    if not expert_mode and action_type == constants.GRASP:
                        self.grasp_record_list[self.grasp_record_index] = 1 if (rewards > 0).sum() > 0 else 0
                        self.grasp_record_index += 1
                        if self.grasp_record_index >= self.max_result_window:
                            self.grasp_record_index = 0
                    elif not expert_mode and action_type == constants.PUSH:
                        self.push_record_list[self.push_record_index] = 1 if (rewards > 0).sum() > 0 else 0
                        self.push_record_index += 1
                        if self.push_record_index >= self.max_result_window:
                            self.push_record_index = 0

                    #save model
                    self.save_models(action_type, episode_done, expert_mode)

                    #save agent data
                    self.save_agent_data()

                    if self.is_debug:
                        print("[SUCCESS] save agent data")

            if is_train and ((episode == max_episode - 1 or (episode % self.save_all_exp_interval) == 0)):
                self.buffer_replay_expert.save_all_exp_to_dir()
                self.buffer_replay.save_all_exp_to_dir()

                print("[SUCCESS] update all experience priorities")

            #update episode
            episode += 1

    def compute_state_batch_and_state_action_batch(self, action_type, depth_states, gripper_states, yaw_states, actions = None, gripper_actions = None):

        if action_type == constants.GRASP:
            state_batch = torch.zeros(depth_states.shape[0], 3, 128, 128).to(self.device)
            if actions is not None:
                action_state_batch = torch.zeros(depth_states.shape[0], 8, 128, 128).to(self.device)  
        else:
            state_batch = torch.zeros(depth_states.shape[0], 2, 128, 128).to(self.device)  
            if actions is not None:
                action_state_batch = torch.zeros(depth_states.shape[0], 6, 128, 128).to(self.device)  

        for i in range(depth_states.shape[0]):
            
            #preprocess states
            depth_state, gripper_state, yaw_state = self.preprocess_state(depth_img     = depth_states[i],
                                                                          gripper_state = gripper_states[i],
                                                                          yaw_ang       = yaw_states[i])

            #compute state and state action
            if action_type == constants.GRASP:
                
                state = torch.concatenate([torch.FloatTensor(depth_state).unsqueeze(0), 
                                           torch.FloatTensor([gripper_state]).expand(128, 128).unsqueeze(0), 
                                           torch.FloatTensor([yaw_state]).expand(128, 128).unsqueeze(0)], 
                                           dim=0)
                
                if actions is not None:
                    action_state = torch.concatenate([state, 
                                                    torch.FloatTensor([actions[i][0]/constants.MAX_ACTION[0]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][1]/constants.MAX_ACTION[1]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][2]/constants.MAX_ACTION[2]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][3]/constants.MAX_ACTION[3]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([gripper_actions[i]]).expand(128, 128).unsqueeze(0)],
                                                    dim = 0)                
                                
            else:
                state = torch.concatenate([torch.FloatTensor(depth_state).unsqueeze(0), 
                                           torch.FloatTensor([yaw_state]).expand(128, 128).unsqueeze(0)], 
                                           dim=0)

                if actions is not None:
                    action_state = torch.concatenate([state, 
                                                    torch.FloatTensor([actions[i][0]/constants.PUSH_MAX_ACTION[0]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][1]/constants.PUSH_MAX_ACTION[1]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][2]/constants.PUSH_MAX_ACTION[2]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][3]/constants.PUSH_MAX_ACTION[3]]).expand(128, 128).unsqueeze(0)],
                                                    dim = 0)

            state_batch[i] = state
            if actions is not None:
                action_state_batch[i] = action_state

        return state_batch, action_state_batch if actions is not None else None

    def online_update(self, action_type):
        
        #     0,                  1,                    2,                3,
        # batch, batch_depth_states, batch_gripper_states, batch_yaw_states, \
        #             4,                     5,                  6,                          7,
        # batch_actions, batch_gripper_actions, batch_next_actions, batch_next_gripper_actions, \
        #             8,                       9,                        10,                    11,
        # batch_rewards, batch_next_depth_states, batch_next_gripper_states, batch_next_yaw_states, \
        #          12,                 13
        # batch_dones, batch_success_mask
        
        if not self.enable_rl_critic and not self.enable_rl_actor and not self.enable_bc:
            return

        self.grasp_actor.train()
        self.grasp_critic1.train()
        self.grasp_critic2.train()
        self.grasp_critic1_target.eval()
        self.grasp_critic2_target.eval()

        self.push_actor.train()
        self.push_critic1.train()
        self.push_critic2.train()
        self.push_critic1_target.eval()
        self.push_critic2_target.eval()

        #sampling experience 
        if self.enable_rl_critic:
            exp_rl      = self.buffer_replay.sample_buffer(batch_size = self.N_batch, action_type = action_type)
            batch_index = exp_rl[0]
            rewards     = torch.FloatTensor(exp_rl[8]).unsqueeze(1).to(self.device)
            dones       = torch.FloatTensor(exp_rl[12]).unsqueeze(1).to(self.device)

            print(f"[ONLINE UPDATE] N_rl_exp: {len(batch_index)}")

            #compute rl - state and action state
            state_batch, action_state_batch = self.compute_state_batch_and_state_action_batch(action_type,  
                                                                                              depth_states    = exp_rl[1],  
                                                                                              gripper_states  = exp_rl[2],  
                                                                                              yaw_states      = exp_rl[3], 
                                                                                              actions         = exp_rl[4], 
                                                                                              gripper_actions = exp_rl[5])

            #compute rl - next state        
            next_state_batch, _ = self.compute_state_batch_and_state_action_batch(action_type, 
                                                                                  depth_states    = exp_rl[9], 
                                                                                  gripper_states  = exp_rl[10], 
                                                                                  yaw_states      = exp_rl[11],      
                                                                                  actions         = None,      
                                                                                  gripper_actions = None)
            
            #compute rl - next action state
            with torch.no_grad():
                if action_type == constants.GRASP:
                    next_action_state_batch        = torch.zeros(next_state_batch.shape[0], 8, 128, 128).to(self.device) 
                    next_action_batch, next_normalised_action_batch, next_gripper_action_batch, _, _, _ = self.grasp_actor.get_actions(next_state_batch)
                else:
                    next_action_state_batch        = torch.zeros(next_state_batch.shape[0], 6, 128, 128).to(self.device)
                    next_action_batch, next_normalised_action_batch, next_gripper_action_batch, _, _, _ = self.push_actor.get_actions(next_state_batch)
                
                for i in range(next_state_batch.shape[0]):
                    
                    #compute state and state action
                    if action_type == constants.GRASP:             
                        
                        next_action_state = torch.concatenate([next_state_batch[i], 
                                                              (next_normalised_action_batch[i][0]).expand(128, 128).unsqueeze(0),
                                                              (next_normalised_action_batch[i][1]).expand(128, 128).unsqueeze(0),
                                                              (next_normalised_action_batch[i][2]).expand(128, 128).unsqueeze(0),
                                                              (next_normalised_action_batch[i][3]).expand(128, 128).unsqueeze(0),
                                                              (next_gripper_action_batch[i]).expand(128, 128).unsqueeze(0)],
                                                               dim = 0)
                        
                    else:
                        
                        next_action_state = torch.concatenate([next_state_batch[i], 
                                                              (next_normalised_action_batch[i][0]).expand(128, 128).unsqueeze(0),
                                                              (next_normalised_action_batch[i][1]).expand(128, 128).unsqueeze(0),
                                                              (next_normalised_action_batch[i][2]).expand(128, 128).unsqueeze(0),
                                                              (next_normalised_action_batch[i][3]).expand(128, 128).unsqueeze(0)],
                                                               dim = 0)
            
                    next_action_state_batch[i] = next_action_state

        if self.enable_bc:
            exp_expert = self.buffer_replay_expert.sample_buffer(batch_size = self.N_batch, action_type = action_type)
            expert_batch_index = exp_expert[0]

            expert_action_batch         = torch.FloatTensor(exp_expert[4]).float().to(self.device)
            expert_gripper_action_batch = torch.FloatTensor(exp_expert[5]).long().to(self.device)

            expert_rewards = torch.FloatTensor(exp_expert[8]).unsqueeze(1).to(self.device) 
            expert_dones   = torch.FloatTensor(exp_expert[12]).unsqueeze(1).to(self.device)

            print(f"[ONLINE UPDATE] N_expert_exp: {len(expert_batch_index)}")
        
            #compute bc - state and action state
            expert_state_batch, expert_action_state_batch = self.compute_state_batch_and_state_action_batch(action_type,  
                                                                                                            depth_states    = exp_expert[1],  
                                                                                                            gripper_states  = exp_expert[2],  
                                                                                                            yaw_states      = exp_expert[3], 
                                                                                                            actions         = exp_expert[4], 
                                                                                                            gripper_actions = exp_expert[5])

            #compute bc - next state and action state
            _, next_expert_action_state_batch = self.compute_state_batch_and_state_action_batch(action_type,  
                                                                                                depth_states    = exp_expert[9],  
                                                                                                gripper_states  = exp_expert[10],  
                                                                                                yaw_states      = exp_expert[11], 
                                                                                                actions         = exp_expert[6], 
                                                                                                gripper_actions = exp_expert[7])
        
        if self.enable_rl_critic:
            #compute critic loss rl
            critic1_loss_rl, critic2_loss_rl, \
            critic1_loss_rl_no_reduce, critic2_loss_rl_no_reduce = self.compute_critic_loss(action_type, 
                                                                                            action_state_batch, 
                                                                                            next_action_state_batch,
                                                                                            rewards,
                                                                                            dones)
            
        if self.enable_bc:
            #compute critic loss bc
            critic1_loss_bc, critic2_loss_bc, \
            critic1_loss_bc_no_reduce, critic2_loss_bc_no_reduce = self.compute_critic_loss(action_type, 
                                                                                            expert_action_state_batch, 
                                                                                            next_expert_action_state_batch,
                                                                                            expert_rewards,
                                                                                            expert_dones)
            
        if self.enable_rl_critic and self.enable_bc:
            critic1_loss = critic1_loss_rl + critic1_loss_bc
            critic2_loss = critic2_loss_rl + critic2_loss_bc
        elif self.enable_bc:
            critic1_loss = critic1_loss_bc
            critic2_loss = critic2_loss_bc
        elif self.enable_rl_critic:
            critic1_loss = critic1_loss_rl 
            critic2_loss = critic2_loss_rl

        if action_type == constants.GRASP:
            self.grasp_critic1.optimiser.zero_grad()
            critic1_loss.backward()
            self.grasp_critic1.optimiser.step()
        
            self.grasp_critic2.optimiser.zero_grad()
            critic2_loss.backward()
            self.grasp_critic2.optimiser.step()

            self.soft_update(self.grasp_critic1, self.grasp_critic1_target, tau = self.tau)
            self.soft_update(self.grasp_critic2, self.grasp_critic2_target, tau = self.tau)
        else:
            self.push_critic1.optimiser.zero_grad()
            critic1_loss.backward()
            self.push_critic1.optimiser.step()
        
            self.push_critic2.optimiser.zero_grad()
            critic2_loss.backward()
            self.push_critic2.optimiser.step()

            self.soft_update(self.push_critic1, self.push_critic1_target, tau = self.tau)
            self.soft_update(self.push_critic2, self.push_critic2_target, tau = self.tau)

        print(f"[CRITIC UPDATE] critic1_loss: {critic1_loss.item()}, critic2_loss: {critic2_loss.item()}")

        #actor update
        if self.enable_rl_actor:
            #compute actor loss rl
            actor_loss_rl = self.compute_actor_loss_rl(action_type, state_batch)
        
        if self.enable_bc:
            #compute actor loss bc
            actor_loss_bc, actor_loss_bc_no_reduce = self.compute_actor_loss_bc(action_type, expert_state_batch, expert_action_batch, expert_gripper_action_batch)

        if self.enable_rl_actor and self.enable_bc:
            actor_loss = (actor_loss_rl + self.bc_lambda*actor_loss_bc)
            self.bc_lambda = np.max([self.min_bc_lambda, self.bc_lambda*self.bc_lambda_decay])
        elif self.enable_bc:
            actor_loss = actor_loss_bc
        elif self.enable_rl_actor:
            actor_loss = actor_loss_rl

        if self.enable_bc or self.enable_rl_actor:
            if action_type == constants.GRASP:
                self.grasp_actor.optimiser.zero_grad()
                actor_loss.backward()
                self.grasp_actor.optimiser.step()
            else:
                self.push_actor.optimiser.zero_grad()
                actor_loss.backward()
                self.push_actor.optimiser.step()        

            print(f"[ACTOR UPDATE] bc_lambda: {self.bc_lambda}")
            print(f"[ACTOR UPDATE] actor_loss: {actor_loss.item()}")
            if self.enable_rl_actor:
                print(f"[ACTOR UPDATE] actor_loss_rl: {actor_loss_rl.item()}")
            if self.enable_bc:
                print(f"[ACTOR UPDATE] actor_loss_bc: {actor_loss_bc.item()}")

        #update priority
        if self.enable_rl_critic:
            critic_loss_rl_no_reduce = (critic1_loss_rl_no_reduce + critic2_loss_rl_no_reduce)/2.
            self.buffer_replay.update_buffer(batch_index, 
                                             None,
                                             critic_loss_rl_no_reduce.to(torch.device('cpu')).detach().numpy())

        if self.enable_bc:
            critic_loss_bc_no_reduce = (critic1_loss_bc_no_reduce + critic2_loss_bc_no_reduce)/2.
            self.buffer_replay_expert.update_buffer(expert_batch_index, 
                                                    actor_loss_bc_no_reduce.to(torch.device('cpu')).detach().numpy(),
                                                    critic_loss_bc_no_reduce.to(torch.device('cpu')).detach().numpy())

    def compute_critic_loss(self, 
                            action_type, 
                            action_state_batch, 
                            next_action_state_batch,
                            rewards, 
                            dones):
        
        with torch.no_grad():

            if action_type == constants.GRASP:
                next_q1 = self.grasp_critic1_target(next_action_state_batch)
                next_q2 = self.grasp_critic2_target(next_action_state_batch)

            else:
                next_q1 = self.push_critic1_target(next_action_state_batch)
                next_q2 = self.push_critic2_target(next_action_state_batch)

            next_q = torch.min(next_q1, next_q2)

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # update critics
        if action_type == constants.GRASP:
            current_q1 = self.grasp_critic1(action_state_batch)
            current_q2 = self.grasp_critic2(action_state_batch)

        else:
            current_q1 = self.push_critic1(action_state_batch)
            current_q2 = self.push_critic2(action_state_batch)

        if self.is_debug:
            if len(current_q1.shape) != len(target_q.shape):
                print("[ERROR] len(current_q1.shape) != len(target_q.shape)")
            elif len(current_q2.shape) != len(target_q.shape):
                print("[ERROR] len(current_q2.shape) != len(target_q.shape)")
            else:
                for i in range(len(current_q1.shape)):
                    if current_q1.shape[i] != target_q.shape[i]:
                        print(f"[ERROR] current_q1.shape[{i}] != target_q.shape[{i}]")
                    if current_q2.shape[i] != target_q.shape[i]:
                        print(f"[ERROR] current_q2.shape[{i}] != target_q.shape[{i}]")

        critic1_loss_no_reduce = nn.MSELoss(reduction = 'none')(current_q1, target_q)
        critic2_loss_no_reduce = nn.MSELoss(reduction = 'none')(current_q2, target_q)

        critic1_loss = torch.mean(critic1_loss_no_reduce, dim = 0)
        critic2_loss = torch.mean(critic2_loss_no_reduce, dim = 0)

        return critic1_loss, critic2_loss, critic1_loss_no_reduce, critic2_loss_no_reduce

    def compute_actor_loss_bc(self, action_type, expert_state_batch, expert_actions, expert_gripper_actions):

        #compute actions 
        if action_type == constants.GRASP:
            #compute actions for bc loss
            _, normalised_actions_bc, _, _, _, gripper_action_probs_bc = self.grasp_actor.get_actions(expert_state_batch)
        else:
            #compute actions for bc loss
            # action, normalised_action, gripper_action, z, normal, gripper_action_prob
            _, normalised_actions_bc, _, _, _, gripper_action_probs_bc = self.push_actor.get_actions(expert_state_batch)

        #compute normalised action
        if action_type == constants.GRASP:
            normalised_expert_actions = expert_actions/torch.FloatTensor(constants.MAX_ACTION).view(1, len(constants.MAX_ACTION)).to(self.device)
        else: 
            normalised_expert_actions = expert_actions/torch.FloatTensor(constants.PUSH_MAX_ACTION).view(1, len(constants.PUSH_MAX_ACTION)).to(self.device)
        
        if self.is_debug:
            if len(normalised_actions_bc.shape) != len(normalised_expert_actions.shape):
                print("[ERROR] len(normalised_actions_bc.shape) != len(normalised_expert_actions.shape)")
            elif normalised_actions_bc.shape[1] != 4:
                 print("[ERROR] normalised_actions_bc.shape[1] != 4")
            elif normalised_expert_actions.shape[1] != 4:
                 print("[ERROR] normalised_expert_actions.shape[1] != 4")
            else:
                for i in range(len(normalised_actions_bc.shape)):
                    if normalised_actions_bc.shape[i] != normalised_expert_actions.shape[i]:
                        print(f"[ERROR] normalised_actions_bc.shape[{i}] != normalised_expert_actions.shape[{i}]")

        #compute bc loss
        bc_loss_no_reduce = nn.MSELoss(reduction = 'none')(normalised_actions_bc.float(), normalised_expert_actions.float())
        bc_loss_no_reduce = torch.mean(bc_loss_no_reduce, dim=1)

        if action_type == constants.GRASP:
            ce_loss_no_reduce  = nn.CrossEntropyLoss(reduction = 'none')(gripper_action_probs_bc, expert_gripper_actions.long())*self.gripper_loss_weight
            bc_loss_no_reduce += ce_loss_no_reduce

        bc_loss = torch.mean(bc_loss_no_reduce, dim = 0)

        return bc_loss, bc_loss_no_reduce

    def compute_actor_loss_rl(self, action_type, state_batch):

        #compute actions 
        if action_type == constants.GRASP:
            #compute actions for rl loss
            _, normalised_actions, gripper_actions, z, normal, _ = self.grasp_actor.get_actions(state_batch)
            log_probs = self.grasp_actor.compute_log_prob(normal, normalised_actions, z)

        else:
            #compute actions for rl loss
            _, normalised_actions, gripper_actions, z, normal, _ = self.push_actor.get_actions(state_batch)
            log_probs = self.push_actor.compute_log_prob(normal, normalised_actions, z)

        #compute normalised action
        if action_type == constants.GRASP:
            action_state_batch        = torch.zeros(state_batch.shape[0], 8, 128, 128).to(self.device)              
        else: 
            action_state_batch        = torch.zeros(state_batch.shape[0], 6, 128, 128).to(self.device)              

        for i in range(state_batch.shape[0]):
            
            if action_type == constants.GRASP:             
                #compute action state
                action_state = torch.concatenate([state_batch[i], 
                                                 (normalised_actions[i][0]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][1]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][2]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][3]).expand(128, 128).unsqueeze(0),
                                                 (gripper_actions[i]).expand(128, 128).unsqueeze(0)],
                                                  dim = 0)
            else:
                #compute action state
                action_state = torch.concatenate([state_batch[i], 
                                                 (normalised_actions[i][0]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][1]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][2]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][3]).expand(128, 128).unsqueeze(0)],
                                                  dim = 0)

            action_state_batch[i] = action_state
        
        #compute q value
        if action_type == constants.GRASP:
            q1 = self.grasp_critic1(action_state_batch)
            q2 = self.grasp_critic2(action_state_batch)
        else:
            q1 = self.push_critic1(action_state_batch)
            q2 = self.push_critic2(action_state_batch)
        min_q = torch.min(q1, q2)

        if self.is_debug:
            if len(log_probs.shape) != len(min_q.shape):
                print("[ERROR] len(log_probs.shape) != len(min_q.shape)")
            else:
                for i in range(len(log_probs.shape)):
                    if log_probs.shape[i] != min_q.shape[i]:
                        print(f"[ERROR] log_probs.shape[{i}] != min_q.shape[{i}]")

        #compute rl loss
        rl_loss = (self.alpha * log_probs - min_q).mean()  

        return rl_loss
        
    def save_models(self, action_type, episode_done, expert_mode):

        if action_type == constants.GRASP:
            #save grasp network
            grasp_success_rate = np.sum(self.grasp_record_list)/self.max_result_window
            print(f"[SUCCESS RATE] grasp_success_rate: {grasp_success_rate}/{self.best_grasp_success_rate}")
            if not expert_mode and (self.enable_rl_critic or self.enable_rl_actor or self.enable_bc):
                self.grasp_success_rate_hist.append(grasp_success_rate)
            
            if not expert_mode and self.best_grasp_success_rate < grasp_success_rate:
                self.best_grasp_success_rate = grasp_success_rate
                self.grasp_actor.save_checkpoint(True)
                self.grasp_critic1.save_checkpoint(True)
                self.grasp_critic2.save_checkpoint(True)
                self.grasp_critic1_target.save_checkpoint(True)
                self.grasp_critic2_target.save_checkpoint(True)
                print("[SUCCESS] save best grasp models")
        else:
            #save push network
            push_success_rate = np.sum(self.push_record_list)/self.max_result_window
            if not expert_mode and (self.enable_rl_critic or self.enable_rl_actor or self.enable_bc):            
                self.push_success_rate_hist.append(push_success_rate)
            print(f"[SUCCESS RATE] push_success_rate: {push_success_rate}/{self.best_push_success_rate}")

            if not expert_mode and self.best_push_success_rate < push_success_rate:
                self.best_push_success_rate = push_success_rate
                self.push_actor.save_checkpoint(True)
                self.push_critic1.save_checkpoint(True)
                self.push_critic2.save_checkpoint(True)
                self.push_critic1_target.save_checkpoint(True)
                self.push_critic2_target.save_checkpoint(True)
                print("[SUCCESS] save best push models")

        if episode_done:
            self.grasp_actor.save_checkpoint()
            self.grasp_critic1.save_checkpoint()
            self.grasp_critic2.save_checkpoint()
            self.grasp_critic1_target.save_checkpoint()
            self.grasp_critic2_target.save_checkpoint()
            print("[SUCCESS] save grasp models check point")

            self.push_actor.save_checkpoint()
            self.push_critic1.save_checkpoint()
            self.push_critic2.save_checkpoint()
            self.push_critic1_target.save_checkpoint()
            self.push_critic2_target.save_checkpoint()
            print("[SUCCESS] save push models check point")
        
    def load_models(self):
    
        try:

            if self.is_train:
                self.grasp_actor.load_checkpoint()
                self.grasp_critic1.load_checkpoint()
                self.grasp_critic2.load_checkpoint()
                self.grasp_critic1_target.load_checkpoint()
                self.grasp_critic2_target.load_checkpoint()
                print("[LOAD MODEL] load grasp check point")
            else:
                self.grasp_actor.load_checkpoint()
                self.grasp_critic1.load_checkpoint()
                self.grasp_critic2.load_checkpoint()
                self.grasp_critic1_target.load_checkpoint()
                self.grasp_critic2_target.load_checkpoint()
                print("[LOAD MODEL] load grasp best model")

            print("[SUCCESS] load grasp model")
        except:
            print("[FAIL] load grasp model")  

        try:
            if self.is_train:
                self.push_actor.load_checkpoint()
                self.push_critic1.load_checkpoint()
                self.push_critic2.load_checkpoint()
                self.push_critic1_target.load_checkpoint()
                self.push_critic2_target.load_checkpoint()
                print("[LOAD MODEL] load push check point")
            else:
                self.push_actor.load_checkpoint()
                self.push_critic1.load_checkpoint()
                self.push_critic2.load_checkpoint()
                self.push_critic1_target.load_checkpoint()
                self.push_critic2_target.load_checkpoint()
                print("[LOAD MODEL] load push best model")

            print("[SUCCESS] load push model")
        except:
            print("[FAIL] load push model")

    def save_agent_data(self):

        file_name = os.path.join(self.checkpt_dir, "agent_data.pkl")

        data_dict = {
            'bc_lambda': self.bc_lambda,
            'grasp_record_list': self.grasp_record_list,
            'push_record_list': self.push_record_list,
            'grasp_record_index': self.grasp_record_index, 
            'push_record_index': self.push_record_index,
            'best_grasp_success_rate': self.best_grasp_success_rate,
            'best_push_success_rate': self.best_push_success_rate,
            'grasp_success_rate_hist': self.grasp_success_rate_hist,
            'push_success_rate_hist': self.push_success_rate_hist
        }

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)

    def load_agent_data(self):
        file_name = os.path.join(self.checkpt_dir, "agent_data.pkl")
        with open(file_name, 'rb') as file:
            data_dict = pickle.load(file)

            self.bc_lambda               = data_dict['bc_lambda']     
            self.grasp_record_list       = data_dict['grasp_record_list']
            self.push_record_list        = data_dict['push_record_list']
            self.grasp_record_index      = data_dict['grasp_record_index']
            self.push_record_index       = data_dict['push_record_index']
            self.best_grasp_success_rate = data_dict['best_grasp_success_rate']
            self.best_push_success_rate  = data_dict['best_push_success_rate']
            self.grasp_success_rate_hist = data_dict['grasp_success_rate_hist']
            self.push_success_rate_hist  = data_dict['push_success_rate_hist']


