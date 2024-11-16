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
from network import Actor, Critic, HDL_Net, QNet

#DL related
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
    
class Agent():

    def __init__(self, 
                 env,
                 N_action = 4, #action dimension for low-level action networks
                 N_gripper_action = 2, #action dimension for gripper action (open and close)
                 N_grasp_step_demo = constants.N_STEP_GRASP_DEMO, #max number of grasping step for demonstration
                 N_push_step_demo = constants.N_STEP_PUSH_DEMO, #max number of pushing step for demonstration
                 N_push_step = constants.N_STEP_PUSH, #max number of pushing step allowable for pushing network
                 N_grasp_step = constants.N_STEP_GRASP, #max number of grasping step allowable for grasping network
                 N_batch_hld = 32, #batch for high-level decision network
                 N_batch = 64, #batch for low-level action network
                 hld_lr = 1e-5, #learning rate for high-level decision network
                 lr = 1e-4, #learning rate for low-level action network
                 alpha = 0.01, #for SAC entropy maxisation
                 tau  = 0.05, #for soft update of gripper and push net 
                 tau_hld = 0.001, #for soft update of hld-net 
                 gamma = 0.95, #discount rate for td-learning
                 bc_lambda = 1.00, #factor for balancing rl loss and bc loss
                 rl_lambda = 0., #factor to avoid rl loss dominates in the initial phase
                 bc_lambda_decay = 0.9999, #factor to decay bc_lambda
                 rl_lambda_upscale = 1.001,  #factor to upscale rl_lambda
                 min_bc_lambda = 1, #define min bc_lambda
                 max_rl_lambda = 0., #define max rl_lambda
                 max_memory_size = 25000, #define the max. size of buffer for bc (low-level action network)
                 max_memory_size_rl = 25000, #define the max. size of buffer for rl (low-level action network)
                 max_memory_size_hld = 25000, #define the max. size of buffer for hld network
                 max_action_taken = 50, #max number of action taken allowable for one episode
                 max_stage1_episode = 200, #define the max number of episodes for stage 1
                 success_rate_threshold = 0.7, #define the threshold for switching to full training mode
                 max_result_window = 500, #the maximum record length for low-level action networks
                 max_result_window_hld = 250, #the maximum record length for high-level decision network
                 max_result_window_eval = 100, #the maximum record length for high-level decision network (in eval)
                 gripper_loss_weight = 1.0, #used for preventing ce loss too dominating
                 is_debug = False, #define if show debug message
                 checkpt_dir_agent = 'logs/agent', #checkpoint directory for agent data
                 checkpt_dir_models = 'logs/models', #checkpoint directory for model data
                 exp_dir_expert = 'logs/exp_expert', #directory for expert experience
                 exp_dir_rl = 'logs/exp_rl', #directory for rl experience
                 exp_dir_hld = 'logs/exp_hld'): #directroy for hld experience

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"device: {self.device}")

        #initialise env
        self.env = env
        self.env.N_grasp_step = N_grasp_step_demo
        self.env.N_push_step  = N_push_step_demo
        print("[SUCCESS] initialise environment")

        #initialise action and action_type dimension
        self.N_action = N_action
        self.N_push_step = N_push_step
        self.N_grasp_step = N_grasp_step 
        self.N_gripper_action = N_gripper_action

        #initialise check point directory
        if not os.path.exists(checkpt_dir_agent):
            os.makedirs(checkpt_dir_agent)
        self.checkpt_dir_agent = os.path.abspath(checkpt_dir_agent)

        if not os.path.exists(checkpt_dir_models):
            os.makedirs(checkpt_dir_models)
        self.checkpt_dir_models = os.path.abspath(checkpt_dir_models)

        if not os.path.exists(exp_dir_expert):
            os.makedirs(exp_dir_expert)
        self.exp_dir_expert = os.path.abspath(exp_dir_expert)

        if not os.path.exists(exp_dir_rl):
            os.makedirs(exp_dir_rl)
        self.exp_dir_rl = os.path.abspath(exp_dir_rl)

        if not os.path.exists(exp_dir_hld):
            os.makedirs(exp_dir_hld)
        self.exp_dir_hld = os.path.abspath(exp_dir_hld)

        #initialise high level network
        self.hld_net = HDL_Net(name = "hld_net", N_input_channels = 1, lr = lr, checkpt_dir = self.checkpt_dir_models)
        self.hld_net_target = HDL_Net(name = "hld_net_target", N_input_channels = 1, lr = lr, checkpt_dir = self.checkpt_dir_models)
        self.hld_mode = constants.HLD_MODE

        #[RESEARCH 1.0]
        #initialise grasp actor
        self.grasp_actor = Actor(name = "grasp_actor", 
                                 max_action = constants.MAX_ACTION,
                                 N_input_channels = 2, #depth image + yaw state
                                 lr = lr,
                                 checkpt_dir = self.checkpt_dir_models) 

        #initialise grasp actor
        self.push_actor  = Actor(name = "push_actor", 
                                  max_action = constants.PUSH_MAX_ACTION,
                                  N_input_channels = 2, #depth image + yaw angle
                                  action_type="push", 
                                  lr = lr, 
                                  checkpt_dir = self.checkpt_dir_models)  

        #[RESEARCH 2.0]
        #initialise grasp Q network
        self.grasp_Q = QNet(name = "grasp_QNet", checkpt_dir = self.checkpt_dir_models)
        self.grasp_Q_target = QNet(name = "grasp_QNet_target", checkpt_dir = self.checkpt_dir_models)
        self.push_Q = QNet(name = "push_QNet", checkpt_dir = self.checkpt_dir_models)
        self.push_Q_target = QNet(name = "push_QNet_target", checkpt_dir = self.checkpt_dir_models)

        #soft update to make critic target align with critic
        self.soft_update(critic = self.hld_net, target_critic = self.hld_net_target)
        self.soft_update_Q(critic = self.grasp_Q, target_critic = self.grasp_Q_target)
        self.soft_update_Q(critic = self.push_Q, target_critic = self.push_Q_target)
        print("[SUCCESS] initialise networks")

        #initialise batch size
        self.N_batch = N_batch
        #initialise batch size for hld-net
        self.N_batch_hld = N_batch_hld
        #initialise small constant to prevent zero value
        self.sm_c = 1e-6
        #initialise learning rate of low-level action learning rate
        self.lr = lr
        #initialise learning rate of hld learning rate
        self.hld_lr = hld_lr
        #initialise temperature factor
        self.alpha = alpha
        #initialise discount factor
        self.gamma = gamma
        #gripper action weights
        self.gripper_loss_weight = gripper_loss_weight
        #initialise rl-bc learning balancing factor
        self.rl_lambda = rl_lambda
        self.bc_lambda = bc_lambda
        self.min_bc_lambda = min_bc_lambda
        self.max_rl_lambda = max_rl_lambda
        self.bc_lambda_decay = bc_lambda_decay
        self.rl_lambda_upscale = rl_lambda_upscale
        #initialise soft update factor
        self.tau = tau
        self.tau_hld = tau_hld

        #initialise buffer size
        self.max_memory_size = max_memory_size                  
        self.max_memory_size_rl = max_memory_size_rl              
        self.max_memory_size_hld = max_memory_size_hld

        #initialise if debug
        self.is_debug = is_debug

        #define if the agent is in evaluation mode
        self.is_eval = None

        #initialise max step per episode
        self.max_action_taken = max_action_taken

        #initialise if the agent start reinforcement learning (rl) 
        self.enable_rl = False

        #initialise if the agent start behaviour cloning (bc) 
        self.enable_bc = False

        #for evaluation record
        self.max_result_window_eval = max_result_window_eval
        self.CR_eval = [0]*self.max_result_window_eval
        self.AGS_eval = [0]*self.max_result_window_eval
        self.ATC_eval = [0]*self.max_result_window_eval

        self.eval_index = 0

        #for training record
        #the maximum record length that record if this action is successful or fail
        self.max_result_window = max_result_window

        #for grasping network training data
        self.grasp_record_list = [0]*self.max_result_window
        self.grasp_record_index = 0
        self.grasp_success_rate_hist = []
        self.best_grasp_success_rate = -np.inf

        #for pushing network training data
        self.push_record_list = [0]*self.max_result_window
        self.push_record_index = 0
        self.push_success_rate_hist = []
        self.best_push_success_rate = -np.inf

        #for hld training data
        self.max_result_window_hld = max_result_window_hld
        self.complete_record_train = [0]*self.max_result_window_hld
        self.action_taken_record_train = [0]*self.max_result_window_hld
        self.hld_record_index = 0
        self.best_CR_train = -np.inf
        self.best_ATC_mean_train = np.inf
        self.CR_train = []
        self.ATC_train = []

        #threshold for switching to full training mode (hld network + low-level action network)
        self.success_rate_threshold = success_rate_threshold

        #initialise grasp and push fail counter
        self.grasp_fail_counter = 0
        self.push_fail_counter = 0

        #initialise flag for saving stage 1, 2, 3 model
        self.is_save_stage1 = False
        self.max_stage1_episode = max_stage1_episode
        self.is_save_stage2 = False
        self.is_save_stage3 = False

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

        #turn yaw ang into image
        in_yaw_ang = None
        if yaw_ang is not None and not np.isnan(yaw_ang):
            in_yaw_ang = utils.wrap_ang(yaw_ang)/np.math.pi

        if self.is_debug:
            if in_yaw_ang is not None and (in_yaw_ang < -1 or in_yaw_ang > 1):
                print("[ERROR] in_yaw_ang < -1 or in_yaw_ang > 1")
            if np.min(in_depth_img) < 0 or np.max(in_depth_img) > 1:
                print("[ERROR] np.min(in_depth_img) < 0 or np.max(in_depth_img) > 1") 

        return in_depth_img, None, in_yaw_ang

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

    def soft_update_Q(self, critic, target_critic, tau = None):
        
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
        for target_layer, source_layer in zip(target_critic.Qx, critic.Qx):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                target_layer.running_mean.copy_(tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean)
                target_layer.running_var.copy_(tau * source_layer.running_var + (1 - tau) * target_layer.running_var)

        for target_layer, source_layer in zip(target_critic.Qy, critic.Qy):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                target_layer.running_mean.copy_(tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean)
                target_layer.running_var.copy_(tau * source_layer.running_var + (1 - tau) * target_layer.running_var)

        for target_layer, source_layer in zip(target_critic.Qz, critic.Qz):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                target_layer.running_mean.copy_(tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean)
                target_layer.running_var.copy_(tau * source_layer.running_var + (1 - tau) * target_layer.running_var)

        for target_layer, source_layer in zip(target_critic.Qyaw, critic.Qyaw):
            if isinstance(target_layer, torch.nn.BatchNorm1d):
                target_layer.running_mean.copy_(tau * source_layer.running_mean + (1 - tau) * target_layer.running_mean)
                target_layer.running_var.copy_(tau * source_layer.running_var + (1 - tau) * target_layer.running_var)

    def init_hld_exp(self):
        #initialise memory space for storing a complete set of actions
        self.hld_depth_states = []
        self.hld_action_types = [] 
        self.hld_rewards = []
        self.hld_next_depth_states = []
        self.hld_dones = []
        self.hld_critic_losses = []

    def init_lla_exp(self):

        self.depth_states = []
        self.gripper_states = []
        self.yaw_states = []
        self.actions = []
        self.gripper_actions = []
        self.next_actions = []
        self.next_gripper_actions = []
        self.action_types = []
        self.rewards = []
        self.next_depth_states = []
        self.next_gripper_states = []
        self.next_yaw_states = []
        self.action_dones = []
        self.priorities = []

    def append_hld_exp(self, depth_state, action_type, reward, next_depth_state, done, critic_loss):

        self.hld_depth_states.append(depth_state)
        self.hld_action_types.append(action_type)
        self.hld_rewards.append(reward)
        self.hld_next_depth_states.append(next_depth_state)
        self.hld_dones.append(done)
        self.hld_critic_losses.append(critic_loss)

    def append_lla_exp(self, depth_img, gripper_state, yaw_state, next_depth_img, next_gripper_state, next_yaw_state, 
                       action, next_action, action_type, reward, action_done, priority):
        
        self.depth_states.append(depth_img)
        self.gripper_states.append(gripper_state)
        self.yaw_states.append(yaw_state)

        self.next_depth_states.append(next_depth_img)
        self.next_gripper_states.append(next_gripper_state)
        self.next_yaw_states.append(next_yaw_state)

        try:
            self.actions.append(action.to(torch.device('cpu')).detach().numpy()[0])
        except:
            self.actions.append(action)
        try:
            self.next_actions.append(next_action.to(torch.device('cpu')).detach().numpy()[0])
        except:
            self.next_actions.append(next_action)
        
        self.gripper_actions.append(None)
        self.next_gripper_actions.append(None)       

        self.action_types.append(action_type)
        self.rewards.append(reward)
        self.action_dones.append(action_done)

        print(priority)
        self.priorities.append(priority.to(torch.device('cpu')).detach().numpy() if priority is not None else priority)

        if self.is_debug:
            print("[SUCCESS] append transition experience")

    def record_success_rate(self, action_type, rewards, delta_moves_grasp, delta_moves_push):
        if  action_type == constants.GRASP:
            
            #ensure 1) record the correct failure case and 2) success case
            #the HLD-net can choose grasping action when nothing can be grasped 
            if (len(delta_moves_grasp) > 0 and np.max(rewards) <= 0) or np.max(rewards) > 0:
                self.grasp_record_list[self.grasp_record_index] = 1. if np.max(rewards) > 0 else 0. 
                self.grasp_record_index += 1
                if self.grasp_record_index >= self.max_result_window:
                    self.grasp_record_index = 0

                #update fail counter
                self.grasp_fail_counter = self.grasp_fail_counter + 1 if np.max(rewards) <= 0 else 0 

            #record success rate
            grasp_success_rate = np.sum(self.grasp_record_list)/self.max_result_window
            print(f"[GRASP SUCCESS RATE] {grasp_success_rate*100}%/{self.best_grasp_success_rate*100}% [{self.max_result_window}]")
            self.grasp_success_rate_hist.append(grasp_success_rate)

        elif action_type == constants.PUSH and len(delta_moves_push) > 0:

            self.push_record_list[self.push_record_index] = 1. if np.max(rewards) > 0 else 0.
            self.push_record_index += 1
            if self.push_record_index >= self.max_result_window:
                self.push_record_index = 0

            #update fail counter
            self.push_fail_counter = self.push_fail_counter + 1 if np.max(rewards) <= 0 else 0

            #record success rate
            push_success_rate = np.sum(self.push_record_list)/self.max_result_window
            print(f"[PUSH SUCCESS RATE] {push_success_rate*100}%/{self.best_push_success_rate*100}% [{self.max_result_window}]")        
            self.push_success_rate_hist.append(push_success_rate)

    def record_grasp_attempt_data(self):
        if self.action_type == constants.GRASP:
            self.grasp_counter += 1
            if np.max(self.rewards) > 0:
                self.grasp_success_counter += 1

    def record_evaluation_data(self):

        if self.is_eval:
            if self.env.N_pickable_item <= 0:
                self.CR_eval[self.eval_index] = 1
            else:
                self.CR_eval[self.eval_index] = 0

            self.AGS_eval[self.eval_index] = self.grasp_success_counter/self.grasp_counter
            self.ATC_eval[self.eval_index] = self.N_action_taken

            #update hld record index
            self.eval_index += 1
            if self.eval_index >= self.max_result_window_eval:
                self.eval_index = 0

            #save agent data
            self.save_agent_data()

    def record_hld_data(self):
        if self.is_full_train:

            if self.hld_record_index >= self.max_result_window_hld:
                self.hld_record_index = 0

            if self.env.N_pickable_item <= 0:
                self.complete_record_train[self.hld_record_index] = 1
            else:
                self.complete_record_train[self.hld_record_index] = 0

            self.action_taken_record_train[self.hld_record_index] = self.N_action_taken

            #update hld record index
            self.hld_record_index += 1

            #save hld-net model
            self.save_models_hld()

            #save agent data
            self.save_agent_data()

    def reset_episode(self):
        #initialise episode data
        self.N_action_taken = 0
        self.episode_done = False
        self.action_type = None
        self.is_success_grasp = False
        self.grasp_counter = 0
        self.grasp_success_counter = 0
        self.grasp_fail_counter = 0
        self.push_fail_counter = 0

        #reset environment
        self.env.reset(reset_item = True)
        self.is_env_reset = True

        #init hld experience
        self.init_hld_exp()

    def init_interact(self, is_eval, lla_mode, hld_mode):

        #set evaluation mode and hld mode
        if is_eval:
            self.is_eval = is_eval
            self.is_full_train = False
            self.hld_mode = hld_mode
            self.lla_mode = lla_mode
        else:
            self.is_eval = is_eval
            self.is_full_train = False
            self.hld_mode = constants.HLD_MODE
            self.lla_mode = lla_mode

            #initialise buffer replay
            self.buffer_replay        = BufferReplay(max_memory_size = self.max_memory_size_rl, 
                                                     checkpt_dir = self.exp_dir_rl)
            self.buffer_replay_expert = BufferReplay(max_memory_size = self.max_memory_size, 
                                                     checkpt_dir = self.exp_dir_expert)
            self.buffer_replay_hld    = BufferReplay_HLD(max_memory_size = int(self.max_memory_size_hld), 
                                                         checkpt_dir = self.exp_dir_hld)
            print("[SUCCESS] initialise memory buffer")

        #set training mode
        #TODO: [NOTE 13/11/2024]: handle self.enable_rl flag
        if self.is_eval:
            self.enable_bc = False 
            self.enable_rl = False
        else:
            self.enable_bc = True 
            self.enable_rl = False

        #load agent data
        self.episode = 0
        try:
            self.load_agent_data()
            self.is_transform_to_full_train()
            if self.is_save_stage1 and not self.is_eval:
                self.enable_rl = True
            print("[SUCCESS] load agent data") 
        except:
            print("[FAIL] load agent data") 

        #load network model
        self.load_models()

    def get_hld_decision(self):

        #get demonstration action of pushing and grasping
        if self.lla_mode == constants.BC_ONLY:
            delta_moves_grasp, delta_moves_push, _, _ = self.env.demo_guidance_generation()
        elif self.lla_mode == constants.BC_RL:
            delta_moves_grasp, delta_moves_push, _, _ = self.env.demo_guidance_generation(is_continous_output = False)

        #get raw data
        _, hld_depth_img = self.env.get_rgbd_data()

        #determine the action type and demonstration action
        if (self.is_full_train or self.is_eval) and self.hld_mode == constants.HLD_MODE:

            #get current state
            in_hld_depth_img, _, _ = self.preprocess_state(depth_img = hld_depth_img, 
                                                           gripper_state = None, 
                                                           yaw_ang = None)
            
            hld_state = torch.FloatTensor(in_hld_depth_img).unsqueeze(0).unsqueeze(0)

            #make high-level decision
            self.hld_net.eval()
            with torch.no_grad():
                hld_q_values, self.action_type = self.hld_net.make_decisions(hld_state)

            #get demonstration movement
            demo_low_level_actions = delta_moves_grasp if self.action_type == constants.GRASP else delta_moves_push

        elif not self.is_full_train and not self.is_eval and self.hld_mode == constants.HLD_MODE:

            if len(delta_moves_grasp) > 0:
                demo_low_level_actions = delta_moves_grasp
                self.action_type = constants.GRASP
            else:
                demo_low_level_actions = delta_moves_push
                self.action_type = constants.PUSH

        elif self.is_eval and self.hld_mode == constants.GRASP_ONLY:

            self.action_type = constants.GRASP
            demo_low_level_actions = delta_moves_grasp

        elif self.is_eval and self.hld_mode == constants.SEQ_GRASP_PUSH:

            if self.action_type is None or self.action_type == constants.PUSH:
                self.action_type = constants.GRASP
                demo_low_level_actions = delta_moves_grasp
            elif self.action_type == constants.GRASP:
                self.action_type = constants.PUSH
                demo_low_level_actions = delta_moves_push

        return (hld_q_values if (self.is_full_train or self.is_eval) and self.hld_mode == constants.HLD_MODE else None), demo_low_level_actions, delta_moves_grasp, delta_moves_push, hld_depth_img

    def get_action_from_network(self, state):

        #estimate action
        if self.action_type == constants.GRASP:
            self.grasp_actor.eval()
            with torch.no_grad():
                if self.lla_mode == constants.BC_ONLY:
                    action, normalised_action, _, _ = self.grasp_actor.get_actions(state)
                elif self.lla_mode == constants.BC_RL:
                    Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind = self.grasp_Q.get_actions(state)
        else:
            self.push_actor.eval()
            with torch.no_grad():
                if self.lla_mode == constants.BC_ONLY:
                    action, normalised_action, _, _ = self.push_actor.get_actions(state)
                elif self.lla_mode == constants.BC_RL:
                    Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind = self.grasp_Q.get_actions(state)

        if self.is_debug:
            print("[SUCCESS] estimate actions from network") 

        if self.lla_mode == constants.BC_ONLY:
            return action, normalised_action
        elif self.lla_mode == constants.BC_RL:
            return Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind
        
    def convert_action_index_to_action(self, action_index, action_type, is_z = False, is_xyz = True):

        if is_xyz:
            if action_type == constants.GRASP:
                if is_z:
                    action = constants.GRASP_MAX_ACTION_Q[2]
                else:
                    action = constants.GRASP_MAX_ACTION_Q[0]
            elif action_type == constants.PUSH:
                if is_z:
                    action = constants.PUSH_MAX_ACTION_Q[2]
                else:
                    action = constants.PUSH_MAX_ACTION_Q[0]
        else:
            if action_type == constants.GRASP:
                action = constants.GRASP_MAX_ACTION_Q[-1]
            elif action_type == constants.PUSH:
                action = constants.PUSH_MAX_ACTION_Q[-1]

        if action_index == 0:
            return -action
        elif action_index == 1:
            return 0.
        elif action_index == 2:
            return action
                
    def convert_action_to_action_index(self, action, action_type, is_xyz = True):

        if action == 0:
            return 1

        return 0 if action < 0 else 2

    def get_action_from_network_Q(self, state):

        #estimate action
        if self.action_type == constants.GRASP:
            self.grasp_Q.eval()
            with torch.no_grad():
                Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind = self.grasp_Q.get_actions(state)
        else:
            self.push_Q.eval()
            with torch.no_grad():
                Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind = self.push_Q.get_actions(state)

        if self.is_debug:
            print("[SUCCESS] estimate actions from network") 

        return Qx, Qy, Qz, Qyaw, x_ind, y_ind, z_ind, yaw_ind

    def get_action_from_demo(self, move):

        action_demo, gripper_action_demo = np.array(move[0:self.N_action]), np.argmax(np.array(move[-2:]))
        gripper_action_demo = torch.FloatTensor([gripper_action_demo]).unsqueeze(0).to(self.device)
        action_demo = torch.FloatTensor(action_demo).unsqueeze(0).to(self.device)

        if self.action_type == constants.GRASP:
            normalised_action_demo = action_demo/torch.FloatTensor(constants.MAX_ACTION).view(1, len(constants.MAX_ACTION)).to(self.device)
        else:
            normalised_action_demo = action_demo/torch.FloatTensor(constants.PUSH_MAX_ACTION).view(1, len(constants.PUSH_MAX_ACTION)).to(self.device)

        return action_demo, normalised_action_demo

    def is_action_done(self, step_low_level, N_step_low_level, is_pushed):
        if (step_low_level == N_step_low_level - 1 or self.is_success_grasp or self.env.is_out_of_working_space or 
            not self.env.can_execute_action or self.env.is_collision_to_ground or self.env.gripper_cannot_operate or 
            is_pushed):
            return True
        else:
            return False   

    def get_q_value_from_critic(self, state, normalised_action, is_compute_target = False):

        #compute current action state
        action_state = torch.concatenate([state, 
                                          torch.FloatTensor([normalised_action[0]]).expand(128, 128).unsqueeze(0),
                                          torch.FloatTensor([normalised_action[1]]).expand(128, 128).unsqueeze(0),
                                          torch.FloatTensor([normalised_action[2]]).expand(128, 128).unsqueeze(0),
                                          torch.FloatTensor([normalised_action[3]]).expand(128, 128).unsqueeze(0)],
                                          dim = 0).unsqueeze(0).to(self.device)  

        #compute action state and current q value
        if self.action_type == constants.GRASP:
            if is_compute_target:
                self.grasp_critic1_target.eval()
                self.grasp_critic2_target.eval()

                with torch.no_grad(): 
                    q1 = self.grasp_critic1_target(action_state)
                    q2 = self.grasp_critic2_target(action_state)   
            else:
                self.grasp_critic1.eval()
                self.grasp_critic2.eval()

                with torch.no_grad(): 
                    q1 = self.grasp_critic1(action_state)
                    q2 = self.grasp_critic2(action_state)     
        else:
            if is_compute_target:
                self.push_critic1_target.eval()
                self.push_critic2_target.eval()

                with torch.no_grad():
                    q1 = self.push_critic1_target(action_state)
                    q2 = self.push_critic2_target(action_state)
            else:
                self.push_critic1.eval()
                self.push_critic2.eval()

                with torch.no_grad():
                    q1 = self.push_critic1(action_state)
                    q2 = self.push_critic2(action_state)

        if self.is_debug:
            print("[SUCCESS] compute current q value") 

        return q1, q2

    def is_expert_mode(self, episode, max_episode, demo_low_level_actions):

        if not self.is_save_stage1 and episode > self.max_stage1_episode:
            self.save_models(None, False, False, 'stage1')
            self.is_save_stage1 = True

        #decide if it should enter demo mode
        if not self.is_eval and \
           not self.is_full_train and \
           episode < self.max_stage1_episode:
            is_expert = True
        elif not self.is_eval and \
            ((self.push_fail_counter >= 1 and self.action_type == constants.PUSH) or \
             (self.grasp_fail_counter >= 1 and self.action_type == constants.GRASP)):

            if self.is_full_train: 
                if len(demo_low_level_actions) > 0:
                    is_expert = True
                else:
                    is_expert = False
            else:
                is_expert = True

            #reset fail counter
            if self.action_type == constants.PUSH and self.push_fail_counter >= 1: 
                self.push_fail_counter = 0
            if self.action_type == constants.GRASP and self.grasp_fail_counter >= 1:
                self.grasp_fail_counter = 0
        else:
            is_expert = False
        
        #decide N_step
        if self.action_type == constants.GRASP:
            N_step_low_level = len(demo_low_level_actions) if is_expert else self.N_grasp_step
        else:
            N_step_low_level = len(demo_low_level_actions) if is_expert else self.N_push_step

        print(f"N_step_low_level: {N_step_low_level}")

        return is_expert, N_step_low_level

    def compute_priority(self, current_q1, current_q2, next_q1, next_q2, reward, action_done, is_expert, action_est, action_target):
        #for research 1.0
        critic_loss = None
        if current_q1 is not None:
            next_q   = torch.min(next_q1, next_q2)  
            target_q = reward + (1 - action_done) * self.gamma * next_q
            critic1_loss = nn.MSELoss()(current_q1, target_q)
            critic2_loss = nn.MSELoss()(current_q2, target_q)
            critic_loss  = (critic1_loss + critic2_loss)/2.

        #compute bc loss
        if is_expert:
            bc_loss_no_reduce = nn.MSELoss(reduction = 'none')(action_est.float(), action_target.float())
            bc_loss_no_reduce = torch.mean(bc_loss_no_reduce, dim=1)
            actor_loss = torch.mean(bc_loss_no_reduce, dim = 0)
            priority = actor_loss
        else:
            actor_loss = None
            priority = critic_loss 
            
        print(f"[LOSS] actor_loss: {actor_loss}")

        if current_q1 is not None:
            print(f"[LOSS] critic_loss: {critic_loss}")
            print(f"[Q value] q1: {(target_q - current_q1)[0].item()}, q2: {(target_q - current_q2)[0].item()}, r: {reward}")

        return priority

    def compute_priority_Q(self, next_state, rewards, dones, Qx, Qy, Qz, Qyaw, nQx, nQy, nQz, nQyaw):
        # #for research 2.0
        # with torch.no_grad():
        #     if action_type == constants.GRASP:
        #         self.grasp_Q.eval()
        #         self.grasp_Q_target.eval()
        #         _, _, _, _, next_action_x, next_action_y, next_action_z, next_action_yaw = self.grasp_Q.get_actions(next_state, take_max = True)
        #         next_Qx, next_Qy, next_Qz, next_Qyaw = self.grasp_Q_target(next_state)
        #     elif action_type == constants.PUSH:
        #         self.push_Q.eval()
        #         self.push_Q_target.eval()
        #         _, _, _, _, next_action_x, next_action_y, next_action_z, next_action_yaw = self.push_Q.get_actions(next_state, take_max = True)
        #         next_Qx, next_Qy, next_Qz, next_Qyaw = self.push_Q_target(next_state)

        #     next_Qx = next_Qx[0][next_action_x] 
        #     next_Qy = next_Qy[0][next_action_y] 
        #     next_Qz = next_Qz[0][next_action_z] 
        #     next_Qyaw = next_Qyaw[0][next_action_yaw] 

        target_Qx = rewards + (1 - dones)*nQx
        target_Qy = rewards + (1 - dones)*nQy
        target_Qz = rewards + (1 - dones)*nQz
        target_Qyaw = rewards + (1 - dones)*nQyaw

        q_loss = nn.MSELoss()(Qx, target_Qx) + \
                 nn.MSELoss()(Qy, target_Qy) + \
                 nn.MSELoss()(Qz, target_Qz) + \
                 nn.MSELoss()(Qyaw, target_Qyaw)
        
        return q_loss

    def compute_priority_hld(self, next_hld_state, hld_reward, current_hld_q):
        #compute priority
        self.hld_net.eval()
        self.hld_net_target.eval()
        with torch.no_grad():
            _, next_action_type = self.hld_net.make_decisions(next_hld_state, take_max = True)

            next_hld_q_value, _ = self.hld_net_target.make_decisions(next_hld_state, take_max = True)
            next_hld_q_value = next_hld_q_value[0][next_action_type]

            target_hld_q_values = hld_reward + (1. - self.episode_done)*self.gamma*next_hld_q_value

            # current_hld_q = hld_q_values[0][self.action_type]
            hld_critic_loss = nn.MSELoss()(current_hld_q, target_hld_q_values).to(torch.device('cpu')).detach().numpy()

            print(f"[HLD reward] r: {hld_reward}")
            print(f"[HLD CRITIC LOSS] q: {current_hld_q}, target q: {target_hld_q_values}, action type: {self.action_type} next action type: {next_action_type}")
            print(f"[HLD CRITIC LOSS] hld critic loss: {hld_critic_loss}")

        return hld_critic_loss

    def is_reset_env(self, is_expert):
        #check if out of working space
        if self.env.is_out_of_working_space:
            print("[WARN] out of working space")
        #check if action executable
        if not self.env.can_execute_action:
            print("[WARN] action is not executable")
        #check if collision to ground
        if self.env.is_collision_to_ground:
            print("[WARN] collision to ground")                        
        #check if the gripper can operate normally
        if self.env.gripper_cannot_operate:
            print("[WARN] gripper cannot function properly")                        

        #this set of motions is not executable => break it
        if (self.env.is_out_of_working_space or 
            not self.env.can_execute_action or 
            self.env.is_collision_to_ground or 
            self.env.gripper_cannot_operate):

            if is_expert and not self.env.can_execute_action:
                #if expert mode's motion is not executable + agent fails => 
                #infinite loop => reset the items on the ground
                self.env.reset(reset_item = True)
            else:
                self.env.reset(reset_item = False)

            self.is_env_reset = True
            print("[WARN] stop executing this action!")

    def is_transform_to_full_train(self):
        if not self.is_eval:
            push_success_rate  = np.sum(self.push_record_list)/len(self.push_record_list)
            grasp_success_rate = np.sum(self.grasp_record_list)/len(self.grasp_record_list)

            if not self.is_full_train and push_success_rate >= self.success_rate_threshold and grasp_success_rate >= self.success_rate_threshold:
                self.is_full_train = True
                print("[TRAIN MODE] TRANSFORM TO FULL TRAIN MODE")

                if not self.is_save_stage2:
                    self.save_models(None, False, False, 'stage2')
                    self.is_save_stage2 = True

    def compute_state_batch_and_state_action_batch(self, action_type, depth_states, gripper_states, yaw_states, actions = None, gripper_actions = None):

        if action_type == constants.GRASP:
            state_batch = torch.zeros(depth_states.shape[0], 2, 128, 128).to(self.device)
            if actions is not None:
                action_state_batch = torch.zeros(depth_states.shape[0], 6, 128, 128).to(self.device)  
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
                                           torch.FloatTensor([yaw_state]).expand(128, 128).unsqueeze(0)], 
                                           dim=0)
                
                if actions is not None:

                    action_state = torch.concatenate([state, 
                                                    torch.FloatTensor([actions[i][0]/constants.MAX_ACTION[0]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][1]/constants.MAX_ACTION[1]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][2]/constants.MAX_ACTION[2]]).expand(128, 128).unsqueeze(0),
                                                    torch.FloatTensor([actions[i][3]/constants.MAX_ACTION[3]]).expand(128, 128).unsqueeze(0)],
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

    def online_update_hld(self):

        if self.buffer_replay_hld.N_data <= 1:
            return

        self.hld_net.train()
        self.hld_net_target.eval()

        # batch, batch_depth_states, batch_action_types, batch_rewards, batch_next_depth_states, batch_dones
        exp = self.buffer_replay_hld.sample_buffer(self.N_batch_hld)

        N_exp = len(exp[0])
        print(f"N_hld_exp: {N_exp}")

        action_batch = torch.FloatTensor(exp[2]).long().unsqueeze(0).to(self.device)
        reward_batch = torch.FloatTensor(exp[3]).unsqueeze(0).to(self.device)
        done_batch   = torch.FloatTensor(exp[5]).unsqueeze(0).to(self.device)

        state_batch      = torch.zeros(exp[1].shape[0], 1, 128, 128).to(self.device)
        next_state_batch = torch.zeros(exp[4].shape[0], 1, 128, 128).to(self.device)

        for i in range(len(exp[1])):
            #preprocess states
            depth_state, _, _ = self.preprocess_state(depth_img     = exp[1][i],
                                                      gripper_state = None,
                                                      yaw_ang       = None)
            
            state_batch[i] = torch.FloatTensor(depth_state).unsqueeze(0)

            #preprocess states
            next_depth_state, _, _ = self.preprocess_state(depth_img     = exp[4][i],
                                                           gripper_state = None,
                                                           yaw_ang       = None)
            
            next_state_batch[i] = torch.FloatTensor(next_depth_state).unsqueeze(0)

        q_values = self.hld_net(state_batch)[torch.arange(N_exp), action_batch]

        with torch.no_grad():
            next_actions = self.hld_net(next_state_batch).argmax(dim=1).unsqueeze(0).long()
            
            next_q_values = self.hld_net_target(next_state_batch)[torch.arange(N_exp), next_actions]
            target_q_values = reward_batch + (1 - done_batch)*(self.gamma * next_q_values)

        # Compute the loss (Mean Squared Error between the target and the predicted Q-values)
        loss = nn.MSELoss()(q_values, target_q_values)

        # Perform gradient descent to minimize the loss
        self.hld_net.optimiser.zero_grad()
        loss.backward()
        self.hld_net.optimiser.step()

        self.soft_update(self.hld_net, self.hld_net_target, self.tau_hld)

        # update buffer 
        # self.buffer_replay_hld.update_buffer(exp[0], 
        #                                      nn.MSELoss(reduction = 'none')(q_values, target_q_values).to(torch.device('cpu')).detach().numpy()[0])

        if self.is_debug:
            print('[SUCCESS] online update')

    #for research 2.0
    def online_update_lla_Q(self, action_type):

        if self.enable_rl and (not self.buffer_replay.have_grasp_data or not self.buffer_replay.have_push_data):
            return 
        
        if self.enable_bc and (not self.buffer_replay_expert.have_grasp_data or not self.buffer_replay_expert.have_push_data):
            return

        if not self.enable_rl and not self.enable_bc:
            return

        if action_type == constants.GRASP:
            self.grasp_Q.train()
        elif action_type == constants.PUSH:
            self.push_Q.train()

        if self.enable_bc:
            loss_bc = self.compute_lla_Q_loss(action_type, self.buffer_replay_expert, is_bc = True)
        
        if self.enable_rl:
            loss_rl = self.compute_lla_Q_loss(action_type, self.buffer_replay, is_bc = False)
        
        if self.enable_bc and self.enable_rl:
            loss = loss_bc + loss_rl
        elif self.enable_rl:
            loss = loss_rl
        elif self.enable_bc:
            loss = loss_bc
        
        if self.enable_bc:
            print(f"[Q LLA UPDATE] loss_bc: {loss_bc}")
        if self.enable_rl:
            print(f"[Q LLA UPDATE] loss_bc: {loss_rl}")
        print(f"[Q LLA UPDATE] loss: {loss}")

        if action_type == constants.GRASP:
            self.grasp_Q.zero_grad()
            loss.backward()
            self.grasp_Q.optimiser.step()

            self.soft_update_Q(self.grasp_Q, self.grasp_Q_target, self.tau)
        elif action_type == constants.PUSH:            
            self.push_Q.zero_grad()
            loss.backward()
            self.push_Q.optimiser.step()

            self.soft_update_Q(self.push_Q, self.push_Q_target, self.tau)

    #for research 1.0
    def online_update_lla(self, action_type):
        
        #     0,                  1,                    2,                3,
        # batch, batch_depth_states, batch_gripper_states, batch_yaw_states, \
        #             4,                     5,                  6,                          7,
        # batch_actions, batch_gripper_actions, batch_next_actions, batch_next_gripper_actions, \
        #             8,                       9,                        10,                    11,
        # batch_rewards, batch_next_depth_states, batch_next_gripper_states, batch_next_yaw_states, \
        #          12,                 13
        # batch_dones, batch_success_mask

        if self.enable_rl and (not self.buffer_replay.have_grasp_data or not self.buffer_replay.have_push_data):
            return 
        
        if self.enable_bc and (not self.buffer_replay_expert.have_grasp_data or not self.buffer_replay_expert.have_push_data):
            return

        if not self.enable_rl and not self.enable_bc:
            return

        if action_type == constants.GRASP:
            self.grasp_actor.train()
        elif action_type == constants.PUSH:
            self.push_actor.train()

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
                
        if self.enable_bc:
            #compute actor loss bc
            actor_loss_bc, actor_loss_bc_no_reduce = self.compute_actor_loss_bc(action_type, expert_state_batch, expert_action_batch, expert_gripper_action_batch)
            actor_loss = actor_loss_bc

        if self.enable_bc or self.enable_rl:
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
            if self.enable_bc:
                print(f"[ACTOR UPDATE] actor_loss_bc: {actor_loss_bc.item()}")

        if self.enable_bc:
            self.buffer_replay_expert.update_buffer(expert_batch_index, 
                                                    actor_loss_bc_no_reduce.to(torch.device('cpu')).detach().numpy(),
                                                    None)
            
        if self.is_debug:
            print('[SUCCESS] online update')

    def update_low_level_network(self, is_expert, is_sim_abnormal):
        
        #choose buffer replay
        buffer_replay = self.buffer_replay_expert if is_expert else self.buffer_replay

        for i in range(len(self.depth_states)): 
            #only save successful experience when in expert mode
            if (is_expert and (np.array(self.rewards) > 0).sum() <= 0) or is_sim_abnormal:
                continue
            
            #skip storing experience when rl mode doesn't turn on at all
            if not is_expert and not self.enable_rl:
                continue

            buffer_replay.store_transition(self.depth_states[i], 
                                           self.gripper_states[i], 
                                           self.yaw_states[i],
                                           self.actions[i], 
                                           self.gripper_actions[i], 
                                           self.next_actions[i], 
                                           self.next_gripper_actions[i],
                                           self.action_types[i], 
                                           self.rewards[i], 
                                           self.next_depth_states[i], 
                                           self.next_gripper_states[i], 
                                           self.next_yaw_states[i],
                                           self.action_dones[i], 
                                           True if (np.array(self.rewards) > 0).sum() > 0 else False,
                                           self.priorities[i]) 
        
        if self.is_debug:
            print('[SUCCESS] store transition experience for low-level action')   

        #update networks
        if self.lla_mode == constants.BC_ONLY:
            print("[UPDATE] GRASP NETWORK")
            self.online_update_lla(action_type = constants.GRASP)

            print("[UPDATE] PUSH NETWORK")
            self.online_update_lla(action_type = constants.PUSH)
        else:
            print("[UPDATE] GRASP NETWORK")
            self.online_update_lla_Q(action_type = constants.GRASP)

            print("[UPDATE] PUSH NETWORK")
            self.online_update_lla_Q(action_type = constants.PUSH)
        
        #save low-level network model
        self.save_models(self.action_type, self.episode_done, is_expert)

    def update_high_level_network(self, hld_depth_img, hld_q_values, is_sim_abnormal, delta_moves_grasp, delta_moves_push):

        if self.is_full_train:
            
            if not is_sim_abnormal:

                hld_reward = self.env.reward_hld(self.action_type, self.rewards, delta_moves_grasp, delta_moves_push)

                #get raw data
                _, next_hld_depth_img = self.env.get_rgbd_data()

                in_next_hld_depth_img, _, _ = self.preprocess_state(depth_img = next_hld_depth_img, 
                                                                    gripper_state = None, 
                                                                    yaw_ang = None)
                
                next_hld_state = torch.FloatTensor(in_next_hld_depth_img).unsqueeze(0).unsqueeze(0)

                #compute priority
                hld_critic_loss = self.compute_priority_hld(next_hld_state, hld_reward, hld_q_values[0][self.action_type])

                #append hld experience
                self.append_hld_exp(hld_depth_img, self.action_types[0], hld_reward, next_hld_depth_img, self.episode_done, hld_critic_loss)

                #store hld experience
                self.buffer_replay_hld.store_transition(self.hld_depth_states[-1], self.hld_action_types[-1], self.hld_rewards[-1],
                                                        self.hld_next_depth_states[-1], self.hld_dones[-1], self.hld_critic_losses[-1])
            
            #update hld-net
            self.online_update_hld()

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

    def compute_lla_Q_loss(self, action_type, buffer_replay, is_bc):

        #     0,                  1,                    2,                3,
        # batch, batch_depth_states, batch_gripper_states, batch_yaw_states, \
        #             4,                     5,                  6,                          7,
        # batch_actions, batch_gripper_actions, batch_next_actions, batch_next_gripper_actions, \
        #             8,                       9,                        10,                    11,
        # batch_rewards, batch_next_depth_states, batch_next_gripper_states, batch_next_yaw_states, \
        #          12,                 13
        # batch_dones, batch_success_mask

        exp = buffer_replay.sample_buffer(batch_size = self.N_batch, action_type = action_type)
        
        batch_index = exp[0]
        action_batch = torch.FloatTensor(exp[4]).long().to(self.device)
        next_action_batch = torch.FloatTensor(exp[6]).long().to(self.device)
        rewards = torch.FloatTensor(exp[8]).unsqueeze(1).to(self.device) 
        dones = torch.FloatTensor(exp[12]).unsqueeze(1).to(self.device)
        N_exp = len(batch_index)

        print(f"[ONLINE UPDATE] N_exp: {N_exp}")
    
        #compute bc - state and action state
        state_batch, _ = self.compute_state_batch_and_state_action_batch(action_type,  
                                                                         depth_states = exp[1],  
                                                                         gripper_states = exp[2],  
                                                                         yaw_states = exp[3], 
                                                                         actions = exp[4], 
                                                                         gripper_actions = exp[5])

        #compute bc - next state and action state
        next_state_batch, _ = self.compute_state_batch_and_state_action_batch(action_type,  
                                                                              depth_states = exp[9],  
                                                                              gripper_states = exp[10],  
                                                                              yaw_states = exp[11], 
                                                                              actions = exp[6], 
                                                                              gripper_actions = exp[7])

        if action_type == constants.GRASP:
            Qx, Qy, Qz, Qyaw = self.grasp_Q(state_batch)
        elif action_type == constants.PUSH:
            Qx, Qy, Qz, Qyaw = self.push_Q(state_batch)

        Qx = Qx[torch.arange(N_exp), action_batch[:, 0]].unsqueeze(1)
        Qy = Qy[torch.arange(N_exp), action_batch[:, 1]].unsqueeze(1)
        Qz = Qz[torch.arange(N_exp), action_batch[:, 2]].unsqueeze(1)
        Qyaw = Qyaw[torch.arange(N_exp), action_batch[:, 3]].unsqueeze(1)

        with torch.no_grad():
            if action_type == constants.GRASP:
                # next_actions = self.hld_net(next_state_batch).argmax(dim=1).unsqueeze(0).long()
                if not is_bc:
                    next_Qx, next_Qy, next_Qz, next_Qyaw = self.grasp_Q(next_state_batch)
                    next_actions_x = next_Qx.argmax(dim=1).long()
                    next_actions_y = next_Qy.argmax(dim=1).long()
                    next_actions_z = next_Qz.argmax(dim=1).long()
                    next_actions_yaw = next_Qyaw.argmax(dim=1).long()
                else:
                    next_actions_x = next_action_batch[:, 0]
                    next_actions_y = next_action_batch[:, 1]
                    next_actions_z = next_action_batch[:, 2]
                    next_actions_yaw = next_action_batch[:, 3]

                next_Qx, next_Qy, next_Qz, next_Qyaw = self.grasp_Q_target(next_state_batch)
            elif action_type == constants.PUSH:
                if not is_bc:
                    next_Qx, next_Qy, next_Qz, next_Qyaw = self.push_Q(next_state_batch)
                    next_actions_x = next_Qx.argmax(dim=1).long()
                    next_actions_y = next_Qy.argmax(dim=1).long()
                    next_actions_z = next_Qz.argmax(dim=1).long()
                    next_actions_yaw = next_Qyaw.argmax(dim=1).long()
                else:
                    next_actions_x = next_action_batch[:, 0]
                    next_actions_y = next_action_batch[:, 1]
                    next_actions_z = next_action_batch[:, 2]
                    next_actions_yaw = next_action_batch[:, 3]

                next_Qx, next_Qy, next_Qz, next_Qyaw = self.push_Q_target(next_state_batch)

            next_Qx = next_Qx[torch.arange(N_exp), next_actions_x].unsqueeze(1)
            next_Qy = next_Qy[torch.arange(N_exp), next_actions_y].unsqueeze(1)
            next_Qz = next_Qz[torch.arange(N_exp), next_actions_z].unsqueeze(1)
            next_Qyaw = next_Qyaw[torch.arange(N_exp), next_actions_yaw].unsqueeze(1)

            target_Qx = rewards + (1 - dones)*next_Qx
            target_Qy = rewards + (1 - dones)*next_Qy
            target_Qz = rewards + (1 - dones)*next_Qz
            target_Qyaw = rewards + (1 - dones)*next_Qyaw

        loss = nn.MSELoss()(Qx, target_Qx) + \
               nn.MSELoss()(Qy, target_Qy) + \
               nn.MSELoss()(Qz, target_Qz) + \
               nn.MSELoss()(Qyaw, target_Qyaw)
        
        if self.is_debug:
            if target_Qx.shape[1] != 1:
                print("[ERROR] target_Qx.shape[1] != 1") 
            if target_Qy.shape[1] != 1:
                print("[ERROR] target_Qy.shape[1] != 1") 
            if target_Qz.shape[1] != 1:
                print("[ERROR] target_Qz.shape[1] != 1") 
            if target_Qyaw.shape[1] != 1:
                print("[ERROR] target_Qyaw.shape[1] != 1") 

        return loss

    def compute_actor_loss_bc(self, action_type, expert_state_batch, expert_actions, expert_gripper_actions):

        #compute actions 
        if action_type == constants.GRASP:
            #compute actions for bc loss
            # action, normalised_action, z, normal
            # _, normalised_actions_bc, _, _, _, gripper_action_probs_bc = self.grasp_actor.get_actions(expert_state_batch)
            _, normalised_actions_bc, _, _ = self.grasp_actor.get_actions(expert_state_batch)
        else:
            #compute actions for bc loss
            # action, normalised_action, gripper_action, z, normal, gripper_action_prob
            # _, normalised_actions_bc, _, _, _, gripper_action_probs_bc = self.push_actor.get_actions(expert_state_batch)
            _, normalised_actions_bc, _, _ = self.push_actor.get_actions(expert_state_batch)

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

        bc_loss = torch.mean(bc_loss_no_reduce, dim = 0)

        return bc_loss, bc_loss_no_reduce

    def compute_actor_loss_rl(self, action_type, state_batch):

        #compute actions 
        if action_type == constants.GRASP:
            #compute actions for rl loss
            # _, normalised_actions, gripper_actions, z, normal, _ = self.grasp_actor.get_actions(state_batch)

            # action, normalised_action, z, normal
            _, normalised_actions, z, normal = self.grasp_actor.get_actions(state_batch)
            log_probs = self.grasp_actor.compute_log_prob(normal, normalised_actions, z)

        else:
            #compute actions for rl loss
            # _, normalised_actions, gripper_actions, z, normal, _ = self.push_actor.get_actions(state_batch)

            # action, normalised_action, z, normal
            _, normalised_actions, z, normal = self.push_actor.get_actions(state_batch)
            log_probs = self.push_actor.compute_log_prob(normal, normalised_actions, z)

        #compute normalised action
        if action_type == constants.GRASP:
            action_state_batch        = torch.zeros(state_batch.shape[0], 6, 128, 128).to(self.device)              
        else: 
            action_state_batch        = torch.zeros(state_batch.shape[0], 6, 128, 128).to(self.device)              

        for i in range(state_batch.shape[0]):
            
            if action_type == constants.GRASP:             
                action_state = torch.concatenate([state_batch[i], 
                                                 (normalised_actions[i][0]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][1]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][2]).expand(128, 128).unsqueeze(0),
                                                 (normalised_actions[i][3]).expand(128, 128).unsqueeze(0)],
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
        
    def save_models_hld(self):

        #save completion rate
        complete_rate_train = np.sum(self.complete_record_train)/self.max_result_window_hld
        print(f"[HLD COMPLETE RATE] complete_rate: {complete_rate_train*100.}%/{self.best_CR_train*100.}% [{self.max_result_window_hld}]")
        self.CR_train.append(complete_rate_train)
        
        #save average action taken for one episode
        ATC_mean_train = np.sum(self.action_taken_record_train)/(np.array(self.action_taken_record_train) > 0).sum()
        print(f"[HLD ATC] ATC mean: {ATC_mean_train}/{self.best_ATC_mean_train} [{self.max_result_window_hld}]")
        if (np.array(self.action_taken_record_train) > 0).sum() >= self.max_result_window_hld:
            self.ATC_train.append(ATC_mean_train)
        else:
            self.ATC_train.append(0.)

        if self.best_CR_train < complete_rate_train:
            self.best_CR_train = complete_rate_train
            
            self.hld_net.save_checkpoint(True)
            self.hld_net_target.save_checkpoint(True)

            print("[SUCCESS] save best hld models")

        elif self.best_CR_train == complete_rate_train and ATC_mean_train < self.best_ATC_mean_train:
            self.best_ATC_mean_train = ATC_mean_train
            self.best_CR_train = complete_rate_train
            self.hld_net.save_checkpoint(True)
            self.hld_net_target.save_checkpoint(True)

            print("[SUCCESS] save best hld models")

        self.hld_net.save_checkpoint()
        self.hld_net_target.save_checkpoint()
        print("[SUCCESS] save hld models check point")

    def save_models(self, action_type, episode_done, is_expert, save_name = None):

        if action_type == constants.GRASP:
            #save grasp network
            grasp_success_rate = np.sum(self.grasp_record_list)/self.max_result_window
            
            if not is_expert and self.best_grasp_success_rate < grasp_success_rate:
                self.best_grasp_success_rate = grasp_success_rate
                if self.lla_mode == constants.BC_ONLY:
                    self.grasp_actor.save_checkpoint(True)
                elif self.lla_mode == constants.BC_RL:
                    self.grasp_Q.save_checkpoint(True)
                    self.grasp_Q_target.save_checkpoint(True)
                print("[SUCCESS] save best grasp models")
        elif action_type == constants.PUSH:
            #save push network
            push_success_rate = np.sum(self.push_record_list)/self.max_result_window
 
            if not is_expert and self.best_push_success_rate < push_success_rate:
                self.best_push_success_rate = push_success_rate
                if self.lla_mode == constants.BC_ONLY:
                    self.push_actor.save_checkpoint(True)
                elif self.lla_mode == constants.BC_RL:
                    self.push_Q.save_checkpoint(True)
                    self.push_Q_target.save_checkpoint(True)
                print("[SUCCESS] save best push models")

        if episode_done or save_name is not None:
            if self.lla_mode == constants.BC_ONLY:
                self.grasp_actor.save_checkpoint(name=save_name)
                print("[SUCCESS] save grasp models check point")
                self.push_actor.save_checkpoint(name=save_name)
                print("[SUCCESS] save push models check point")
            elif self.lla_mode == constants.BC_RL:
                self.grasp_Q.save_checkpoint(name=save_name)
                self.grasp_Q_target.save_checkpoint(name=save_name)
                print("[SUCCESS] save grasp models check point")
                self.push_Q.save_checkpoint(name=save_name)
                self.push_Q_target.save_checkpoint(name=save_name)
                print("[SUCCESS] save push models check point")    
        
    def load_models(self):
        
        #load hld-net
        try:
            if self.is_eval:
                self.hld_net.load_checkpoint(True)
                self.hld_net_target.load_checkpoint(True)
                print("[LOAD MODEL] load hld-net best model")
            else:
                self.hld_net.load_checkpoint()
                self.hld_net_target.load_checkpoint()
                print("[LOAD MODEL] load hld-net check point")

            print("[SUCCESS] load hld-net model")
        except:
            print("[FAIL] load hld-net model")  

        #load grasp-net
        try:
            if not self.is_eval:
                if self.lla_mode == constants.BC_ONLY:
                    self.grasp_actor.load_checkpoint()
                elif self.lla_mode == constants.BC_RL:
                    self.grasp_Q.load_checkpoint()
                    self.grasp_Q_target.load_checkpoint()
                print("[LOAD MODEL] load grasp check point")
            else:
                if self.lla_mode == constants.BC_ONLY:
                    self.grasp_actor.load_checkpoint(True)
                elif self.lla_mode == constants.BC_RL:
                    self.grasp_Q.load_checkpoint(True)
                    self.grasp_Q_target.load_checkpoint(True)
                print("[LOAD MODEL] load grasp best model")

            print("[SUCCESS] load grasp model")
        except:
            print("[FAIL] load grasp model")  

        #load push-net
        try:
            if not self.is_eval:
                if self.lla_mode == constants.BC_ONLY:
                    self.push_actor.load_checkpoint()
                elif self.lla_mode == constants.BC_RL:
                    self.push_Q.load_checkpoint()
                    self.push_Q_target.load_checkpoint()
                print("[LOAD MODEL] load push check point")
            else:
                if self.lla_mode == constants.BC_ONLY:
                    self.push_actor.load_checkpoint(True)
                elif self.lla_mode == constants.BC_RL:
                    self.push_Q.load_checkpoint(True)
                    self.push_Q_target.load_checkpoint(True)
                print("[LOAD MODEL] load push best model")

            print("[SUCCESS] load push model")
        except:
            print("[FAIL] load push model")

    def save_agent_data(self):

        if self.is_eval:
            file_name = os.path.join(self.checkpt_dir_agent, f"agent_data_eval_{self.hld_mode}.pkl")

            data_dict = {
                'max_result_window_eval': self.max_result_window_eval,

                'CR_eval': self.CR_eval,
                'AGS_eval': self.AGS_eval,
                'ATC_eval': self.ATC_eval,

                'eval_index': self.eval_index, 
            }
        else:
            file_name = os.path.join(self.checkpt_dir_agent, "agent_data.pkl")

            data_dict = {

                'episode': self.episode,

                'bc_lambda': self.bc_lambda,

                'grasp_record_list': self.grasp_record_list,
                'push_record_list': self.push_record_list,
                'complete_record_train': self.complete_record_train,
                'action_taken_record_train': self.action_taken_record_train,

                'grasp_record_index': self.grasp_record_index, 
                'push_record_index': self.push_record_index,
                'hld_record_index': self.hld_record_index,

                'best_grasp_success_rate': self.best_grasp_success_rate,
                'best_push_success_rate': self.best_push_success_rate,
                'best_CR_train': self.best_CR_train,
                'best_ATC_mean_train': self.best_ATC_mean_train,

                'grasp_success_rate_hist': self.grasp_success_rate_hist,
                'push_success_rate_hist': self.push_success_rate_hist,
                'CR_train': self.CR_train,
                'ATC_train': self.ATC_train,

                'is_save_stage1': self.is_save_stage1,
                'is_save_stage2': self.is_save_stage2,
                'is_full_train': self.is_full_train,

            }

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)

        if self.is_debug:
            print("[SUCCESS] save agent data")

    def load_agent_data(self):

        if self.is_eval:
            file_name = os.path.join(self.checkpt_dir_agent, f"agent_data_eval_{self.hld_mode}.pkl")
            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.max_result_window_eval  = data_dict['max_result_window_eval']     
                
                self.CR_eval = data_dict['CR_eval']
                self.AGS_eval = data_dict['AGS_eval']
                self.ATC_eval = data_dict['ATC_eval']
                self.eval_index = data_dict['eval_index']
                
        else:
            file_name = os.path.join(self.checkpt_dir_agent, "agent_data.pkl")
            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.episode = data_dict['episode']
                self.bc_lambda = data_dict['bc_lambda']     
                
                self.grasp_record_list = data_dict['grasp_record_list']
                self.push_record_list = data_dict['push_record_list']
                self.complete_record_train = data_dict['complete_record_train']
                self.action_taken_record_train = data_dict['action_taken_record_train']
                
                self.grasp_record_index = data_dict['grasp_record_index']
                self.push_record_index = data_dict['push_record_index']
                self.hld_record_index = data_dict['hld_record_index']
                
                self.best_grasp_success_rate = data_dict['best_grasp_success_rate']
                self.best_push_success_rate  = data_dict['best_push_success_rate']
                self.best_CR_train = data_dict['best_CR_train']
                self.best_ATC_mean_train = data_dict['best_ATC_mean_train']

                self.grasp_success_rate_hist = data_dict['grasp_success_rate_hist']
                self.push_success_rate_hist  = data_dict['push_success_rate_hist']
                self.CR_train = data_dict['CR_train']
                self.ATC_train = data_dict['ATC_train']

                self.is_save_stage1 = data_dict['is_save_stage1']
                self.is_save_stage2 = data_dict['is_save_stage2']
                self.is_full_train = data_dict['is_full_train'] 

    def interact(self,
                 max_episode = 1,
                 hld_mode = constants.HLD_MODE,
                 lla_mode = constants.BC_ONLY,
                 is_eval = False):

        #initialise interact 
        self.init_interact(is_eval, lla_mode, hld_mode)

        while self.episode < max_episode:

            self.reset_episode()

            while not self.episode_done and self.N_action_taken < self.max_action_taken:

                if self.env.N_pickable_item <= 0:
                    self.episode_done = True
                    continue

                print(f"==== episode: {self.episode} ====")

                #reset any items out of working space
                self.env.reset_item2workingspace()    
                time.sleep(0.5) 

                #get high - level decision (grasp or push)
                hld_q_values, demo_low_level_actions, delta_moves_grasp, delta_moves_push, hld_depth_img = self.get_hld_decision()
                
                #decide if it should enter demo mode
                is_expert, N_step_low_level = self.is_expert_mode(self.episode, max_episode, demo_low_level_actions)

                #initialise memory space for storing a complete set of actions
                self.init_lla_exp()

                for i in range(N_step_low_level):
                    
                    update_mode_msg = f"ENABLE BC: {self.enable_bc} ENABLE RL: {self.enable_rl} FULL TRAIN: {self.is_full_train}"
                    if is_expert:
                        print("==== EXPERT MODE " + update_mode_msg + " ====") 
                    else:
                        print("==== AGENT MODE " + update_mode_msg + " ====") 

                    print(f"==== low level action taken: {self.N_action_taken} N_pickable_item: {self.env.N_pickable_item} ====")

                    #get raw data
                    depth_img, gripper_state, yaw_state = self.env.get_raw_data(self.action_type)

                    if self.is_debug:
                        print("[SUCCESS] get raw data")

                    #preprocess raw data
                    in_depth_img, _, in_yaw_state = self.preprocess_state(depth_img = depth_img, 
                                                                          gripper_state = gripper_state, 
                                                                          yaw_ang = yaw_state, 
                                                                          is_grasp = self.action_type) 

                    #get state
                    state = torch.concatenate([torch.FloatTensor(in_depth_img).unsqueeze(0), 
                                               torch.FloatTensor([in_yaw_state]).expand(128, 128).unsqueeze(0)], dim=0).unsqueeze(0)

                    #estimate actions by network
                    if self.lla_mode == constants.BC_ONLY:
                        action_est, normalised_action_est = self.get_action_from_network(state)

                        #action from demo
                        if is_expert:
                            move = np.array(demo_low_level_actions[i])
                            action, normalised_action = self.get_action_from_demo(move)
                        else:
                            action, normalised_action = action_est, normalised_action_est

                        #interact with env
                        action_to_env_step = action.to(torch.device('cpu')).detach().numpy()[0]

                    elif self.lla_mode == constants.BC_RL:
                        Qx, Qy, Qz, Qyaw, action_x, action_y, action_z, action_yaw = self.get_action_from_network_Q(state)

                        if is_expert:
                            move = np.array(demo_low_level_actions[i])
                            action4d, normalised_action = self.get_action_from_demo(move)

                            action_x = self.convert_action_to_action_index(action4d[0][0], self.action_type, True)
                            action_y = self.convert_action_to_action_index(action4d[0][1], self.action_type, True)
                            action_z = self.convert_action_to_action_index(action4d[0][2], self.action_type, True)
                            action_yaw = self.convert_action_to_action_index(action4d[0][3], self.action_type, False)

                            action = [action_x, action_y, action_z, action_yaw]  
                        else:
                            action = [action_x, action_y, action_z, action_yaw]
                            action4d = [self.convert_action_index_to_action(action[0], self.action_type, is_z = False, is_xyz = True), 
                                        self.convert_action_index_to_action(action[1], self.action_type, is_z = False, is_xyz = True),
                                        self.convert_action_index_to_action(action[2], self.action_type, is_z = True, is_xyz = True),
                                        self.convert_action_index_to_action(action[3], self.action_type, is_z = False, is_xyz = False)]
                            action4d = torch.FloatTensor(action4d).unsqueeze(0).to(self.device)

                        #interact with env
                        action_to_env_step = action4d.to(torch.device('cpu')).detach().numpy()[0]

                    reward, self.is_success_grasp, is_grasped, is_pushed, next_depth_img, next_gripper_state, next_yaw_state, is_sim_abnormal = self.env.step(self.action_type, 
                                                                                                                                                              action_to_env_step[0:3], 
                                                                                                                                                              action_to_env_step[3], 
                                                                                                                                                              None)

                    #print actions
                    print(f"[ACTION TYPE]: {self.action_type}")  
                    move_msg  = f"[MOVE] move: {action_to_env_step}"
                    move_msg += f" move index: {action}"

                    #check if episode is done
                    self.episode_done = False if self.env.N_pickable_item > 0 else True

                    #check if action is done
                    action_done = self.is_action_done(i, N_step_low_level, is_pushed)                    
                    if action_done:
                        if self.action_type == constants.GRASP and not self.is_success_grasp:
                            reward = -1.0
                        elif self.action_type == constants.PUSH and not is_pushed:
                            reward = -1.0

                    #get next state 
                    next_in_depth_img, _, next_in_yaw_state = self.preprocess_state(depth_img = next_depth_img, 
                                                                                    gripper_state = next_gripper_state, 
                                                                                    yaw_ang = next_yaw_state, 
                                                                                    is_grasp = self.action_type)     

                    #get next state
                    next_state = torch.concatenate([torch.FloatTensor(next_in_depth_img).unsqueeze(0), 
                                                    torch.FloatTensor([next_in_yaw_state]).expand(128, 128).unsqueeze(0)], dim=0).unsqueeze(0)

                    #estimate next actions
                    if self.lla_mode == constants.BC_ONLY:
                        next_action_est, next_normalised_action_est = self.get_action_from_network(next_state)

                        if is_expert:                        
                            n_move = np.array(demo_low_level_actions[i+1] if i+1 < len(demo_low_level_actions) else [0,0,0,0,move[-2],move[-1]])
                            next_action, next_normalised_action = self.get_action_from_demo(n_move)
                        else:
                            next_action, next_normalised_action = next_action_est, next_normalised_action_est

                        #compute priority for experience
                        priority = self.compute_priority(None, None, None, None, 
                                                         reward, action_done, is_expert, normalised_action_est, normalised_action)

                        #store experience during executing low-level action
                        self.append_lla_exp(depth_img, gripper_state, yaw_state, next_depth_img, next_gripper_state, next_yaw_state, 
                                            action, next_action, 
                                            self.action_type, reward, action_done, priority)

                    elif self.lla_mode == constants.BC_RL:

                        if is_expert:
                            n_move = np.array(demo_low_level_actions[i+1] if i+1 < len(demo_low_level_actions) else [0,0,0,0,move[-2],move[-1]])
                            next_action4d, next_normalised_action = self.get_action_from_demo(n_move)

                            next_action_x = self.convert_action_to_action_index(next_action4d[0][0], self.action_type, True)
                            next_action_y = self.convert_action_to_action_index(next_action4d[0][1], self.action_type, True)
                            next_action_z = self.convert_action_to_action_index(next_action4d[0][2], self.action_type, True)
                            next_action_yaw = self.convert_action_to_action_index(next_action4d[0][3], self.action_type, False)

                            next_action = [next_action_x, next_action_y, next_action_z, next_action_yaw]
                            move_msg += f"\nnext move: {next_action4d[0]}"
                            move_msg += f" next move index: {next_action}"
                            print(move_msg)

                            with torch.no_grad():
                                if self.action_type == constants.GRASP:
                                    self.grasp_Q.eval()
                                    self.grasp_Q_target.eval()
                                    next_Qx, next_Qy, next_Qz, next_Qyaw = self.grasp_Q_target(next_state)
                                elif self.action_type == constants.PUSH:
                                    self.push_Q.eval()
                                    self.push_Q_target.eval()
                                    next_Qx, next_Qy, next_Qz, next_Qyaw = self.push_Q_target(next_state)
                        else:
                            with torch.no_grad():
                                if self.action_type == constants.GRASP:
                                    self.grasp_Q.eval()
                                    self.grasp_Q_target.eval()
                                    _, _, _, _, next_action_x, next_action_y, next_action_z, next_action_yaw = self.grasp_Q.get_actions(next_state, take_max = True)
                                    next_Qx, next_Qy, next_Qz, next_Qyaw = self.grasp_Q_target(next_state)
                                elif self.action_type == constants.PUSH:
                                    self.push_Q.eval()
                                    self.push_Q_target.eval()
                                    _, _, _, _, next_action_x, next_action_y, next_action_z, next_action_yaw = self.push_Q.get_actions(next_state, take_max = True)
                                    next_Qx, next_Qy, next_Qz, next_Qyaw = self.push_Q_target(next_state)

                            next_action = [next_action_x, next_action_y, next_action_z, next_action_yaw]
                            move_msg += f" next move index: {next_action}"
                            print(move_msg)

                        
                        #compute priority for experience
                        priority = self.compute_priority_Q(next_state, reward, action_done, 
                                                           Qx[0][action_x], Qy[0][action_y], Qz[0][action_z], Qyaw[0][action_yaw], 
                                                           next_Qx[0][next_action_x], next_Qy[0][next_action_y], next_Qz[0][next_action_z], next_Qyaw[0][next_action_yaw])

                        #store experience during executing low-level action
                        self.append_lla_exp(depth_img, gripper_state, yaw_state, next_depth_img, next_gripper_state, next_yaw_state, 
                                            action, next_action, 
                                            self.action_type, reward, action_done, priority)

                    print(f"[PRIORITY] {priority}")

                    #check if episode_done
                    if self.episode_done:
                        print("[SUCCESS] finish one episode")   

                        break 
                    elif action_done:

                        if self.is_success_grasp or i == N_step_low_level - 1 or is_pushed:
                            if self.is_success_grasp:
                                print("[SUCCESS] grasp an item successfully")
                            elif is_grasped:
                                print("[SUCCESS] perform grasp action")
                            elif is_pushed:
                                print("[SUCCESS] perform push action")
                            else:
                                print("[SUCCESS] finish an action")
                                
                        self.is_reset_env(is_expert)
                        
                        break
                
                print("=== end of action ===")
                #update number of action taken
                self.N_action_taken += 1

                #return home
                self.env.return_home(self.is_env_reset, self.action_type)
                self.is_env_reset = False

                if self.is_debug:
                    print("[SUCCESS] return home")

                time.sleep(0.5) 

                #store transition experience of low-level behaviour  
                if (np.array(self.rewards) > 0).sum() > 0:
                    print("[SUCCESS] GRASP OR PUSH ACTION")
                else:
                    print("[FAIL] GRASP OR PUSH ACTION")

                #record success rate for low-level action network
                if not is_sim_abnormal and not is_expert:

                    #record success rate of low-level actions network
                    self.record_success_rate(self.action_type, self.rewards, delta_moves_grasp, delta_moves_push)

                    #record grasping efficiency
                    self.record_grasp_attempt_data()

                    #check if we should change to full training mode
                    self.is_transform_to_full_train()

                if not self.is_eval:
                    #update low-level network
                    self.update_low_level_network(is_expert, is_sim_abnormal)

                    #update high-level network
                    if hld_q_values is not None:
                        self.update_high_level_network(hld_depth_img, hld_q_values, is_sim_abnormal, delta_moves_grasp, delta_moves_push)

                #save agent data
                self.save_agent_data()

            print("=== end of episode ===")

            #save evaluation data
            self.record_evaluation_data()

            #save hld data
            self.record_hld_data()

            #update episode
            self.episode += 1

            #save agent data
            self.save_agent_data()