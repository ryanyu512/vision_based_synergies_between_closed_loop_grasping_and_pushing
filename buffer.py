import os
import copy
import pickle 
import random
import constants
import numpy as np

class BufferReplay():

    def __init__(self, 
                 max_memory_size        = int(1000), 
                 img_size               = 128, 
                 N_action               = 4, 
                 N_gripper_action_type  = 2,
                 alpha                  = 0.6,
                 checkpt_dir            = 'logs/exp',
                 load_checkpt_dir       = None,
                 prioritised_prob       = 0.8,
                 is_debug               = True): 
      
        self.have_grasp_data        = False
        self.have_push_data         = False
        self.grasp_data_size        = 0
        self.push_data_size         = 0
        self.max_memory_size        = max_memory_size
        self.memory_cntr            = 0
        self.img_size               = img_size
        self.N_action               = N_action
        self.N_gripper_action_type  = N_gripper_action_type
        #initialise if the memory is full
        self.is_full                = False
        #initialise power value for prioritisation
        self.alpha                  = alpha
        #initialise small constant to prevent division by zero
        self.sm_c                   = 1e-6

        #current depth state
        self.depth_states         = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current gripper state 
        self.gripper_states       = np.zeros(self.max_memory_size)
        #current yaw angles
        self.yaw_states           = np.zeros(self.max_memory_size)
        #current action
        self.actions              = np.zeros((self.max_memory_size, self.N_action))
        #current gripper action
        self.gripper_actions      = np.zeros((self.max_memory_size))
        #next action
        self.next_actions         = np.zeros((self.max_memory_size, self.N_action))
        #next gripper action
        self.next_gripper_actions = np.zeros((self.max_memory_size))
        #current action type
        self.action_types         = np.ones(self.max_memory_size)*-1
        #next reward
        self.rewards              = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states    = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current gripper state 
        self.next_gripper_states  = np.zeros(self.max_memory_size)
        #current yaw angles
        self.next_yaw_states      = np.zeros(self.max_memory_size)
        #is done in the next state
        self.dones                = np.zeros(self.max_memory_size, dtype = bool)
        #predicted q value: 
        self.predict_qs           = np.zeros(self.max_memory_size)
        #labeled q value: reward + gamma*min(target_q1, target_q2)
        self.labeled_qs           = np.zeros(self.max_memory_size)
        #indicate if this experience is a successful experience
        self.success_mask         = np.zeros(self.max_memory_size, dtype = bool)
        #surprise value
        self.priority             = np.ones(self.max_memory_size)
        #initialise prioritised sampling probability
        self.prioritised_prob     = prioritised_prob

        #initialise check point directory
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)
        self.checkpt_dir = os.path.abspath(checkpt_dir)

        #check the data size in hardware storage
        self.data_length = len(os.listdir(self.checkpt_dir))

        self.is_debug    = is_debug

        try:
            if load_checkpt_dir is None:
                self.load_exp_from_dir()
            else:
                self.load_exp_from_dir(load_checkpt_dir)
                print("[SUCCESS] load low-level buffer")
        except:
            print("[FAIL] cannot load low-level buffer")

    def init_mem_size(self, max_memory_size):

        self.max_memory_size      = max_memory_size
        self.memory_cntr          = 0

        #current depth state
        self.depth_states         = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current gripper state 
        self.gripper_states       = np.zeros(self.max_memory_size)
        #current yaw angles
        self.yaw_states           = np.zeros(self.max_memory_size)
        #current action
        self.actions              = np.zeros((self.max_memory_size, self.N_action))
        #current gripper action
        self.gripper_actions      = np.zeros((self.max_memory_size))
        #next action
        self.next_actions         = np.zeros((self.max_memory_size, self.N_action))
        #next gripper action
        self.next_gripper_actions = np.zeros((self.max_memory_size))
        #current action type
        self.action_types         = np.ones(self.max_memory_size)*-1
        #next reward
        self.rewards              = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states    = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current gripper state 
        self.next_gripper_states  = np.zeros(self.max_memory_size)
        #current yaw angles
        self.next_yaw_states      = np.zeros(self.max_memory_size)
        #is done in the next state
        self.dones                = np.zeros(self.max_memory_size, dtype = bool)
        #predicted q value: 
        self.predict_qs           = np.zeros(self.max_memory_size)
        #labeled q value: reward + gamma*min(target_q1, target_q2)
        self.labeled_qs           = np.zeros(self.max_memory_size)
        #indicate if this experience is a successful experience
        self.success_mask         = np.zeros(self.max_memory_size, dtype = bool)
        #surprise value
        self.priority             = np.ones(self.max_memory_size)

    def store_transition(self, 
                         depth_state, gripper_state, yaw_state, 
                         action, gripper_action, 
                         next_action, next_gripper_action,
                         action_type,reward, 
                         next_depth_state, next_gripper_state, 
                         next_yaw_state, done, 
                         predict_q, labeled_q, is_success, 
                         priority,
                         is_save_to_dir = True):

        #update memory
        if self.memory_cntr >= self.max_memory_size:
            self.is_full = True
            self.memory_cntr = 0

        #minus action type data size when the buffer is full before storing new experience
        if self.is_full:
            if self.action_types[self.memory_cntr] == constants.GRASP:
                self.grasp_data_size -= 1
            else:
                self.push_data_size -= 1

        #compute priority
        priority_ = np.abs(priority + self.sm_c)**self.alpha

        #plus action type data size
        if action_type == constants.GRASP:
            self.have_grasp_data = True
            self.grasp_data_size += 1                
        else:
            self.have_push_data  = True
            self.push_data_size += 1

        print(f"[BUFFER] grasp_data_size: {self.grasp_data_size} push_data_size: {self.push_data_size}")

        self.depth_states[self.memory_cntr]           = depth_state
        self.gripper_states[self.memory_cntr]         = gripper_state
        self.yaw_states[self.memory_cntr]             = yaw_state
        self.actions[self.memory_cntr]                = action
        self.gripper_actions[self.memory_cntr]        = gripper_action
        self.next_actions[self.memory_cntr]           = next_action
        self.next_gripper_actions[self.memory_cntr]   = next_gripper_action
        self.action_types[self.memory_cntr]           = action_type
        self.rewards[self.memory_cntr]                = reward
        self.next_depth_states[self.memory_cntr]      = next_depth_state
        self.next_gripper_states[self.memory_cntr]    = next_gripper_state
        self.next_yaw_states[self.memory_cntr]        = next_yaw_state
        self.dones[self.memory_cntr]                  = done
        self.predict_qs[self.memory_cntr]             = predict_q
        self.labeled_qs[self.memory_cntr]             = labeled_q
        self.success_mask[self.memory_cntr]           = is_success
        self.priority[self.memory_cntr]               = priority_

        if is_save_to_dir:
            data_dict = {
                'depth_state': depth_state,
                'gripper_state': gripper_state,
                'yaw_state': yaw_state,
                'action': action,
                'gripper_action': gripper_action,
                'next_action': next_action,
                'next_gripper_action': next_gripper_action,
                'action_type': action_type,
                'reward': reward,
                'next_depth_state': next_depth_state,
                'next_gripper_state': next_gripper_state,
                'next_yaw_state': next_yaw_state,
                'done': done,
                'predict_q':  predict_q,
                'labeled_q':  labeled_q,
                'success_mask':  is_success,
                'priority': priority,
            }

            self.save_one_exp_to_dir(data_dict)

        #update memory counter
        self.memory_cntr += 1

    def get_experience_by_action_type(self, action_type):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        #get index related to the action_type
        action_index = ((self.action_types == action_type).nonzero())[0]

        #get priorities
        priorities             = self.priority[:max_index][action_index]

        depth_states           = self.depth_states[:max_index][action_index]
        gripper_states         = self.gripper_states[:max_index][action_index]
        yaw_states             = self.yaw_states[:max_index][action_index]
        actions                = self.actions[:max_index][action_index]
        gripper_actions        = self.gripper_actions[:max_index][action_index]
        next_actions           = self.next_actions[:max_index][action_index]
        next_gripper_actions   = self.next_gripper_actions[:max_index][action_index]
        rewards                = self.rewards[:max_index][action_index]
        next_depth_states      = self.next_depth_states[:max_index][action_index]
        next_gripper_states    = self.next_gripper_states[:max_index][action_index]
        next_yaw_states        = self.next_yaw_states[:max_index][action_index]
        dones                  = self.dones[:max_index][action_index]
        predict_qs             = self.predict_qs[:max_index][action_index]
        labeled_qs             = self.labeled_qs[:max_index][action_index]
        success_mask           = self.success_mask[:max_index][action_index]

        return action_index, priorities, depth_states, gripper_states, yaw_states, \
               actions, gripper_actions, next_actions, next_gripper_actions, \
               rewards, next_depth_states, next_gripper_states, next_yaw_states, \
               dones, predict_qs, labeled_qs, success_mask

    def sample_buffer(self, batch_size, action_type):
        #            0,          1,            2,              3,          4,             
        # action_index, priorities, depth_states, gripper_states, yaw_states, 
        #       5,               6,            7,                    8,
        # actions, gripper_actions, next_actions, next_gripper_actions, 
        #       9,                10,                  11,              12,
        # rewards, next_depth_states, next_gripper_states, next_yaw_states,
        #    13,         14,         15,           16                      
        # dones, predict_qs, labeled_qs, success_mask 

        experience = self.get_experience_by_action_type(action_type)

        if experience[1].sum() == 0 or random.random() >= self.prioritised_prob:
            priorities = np.ones_like(experience[1])
        else:
            priorities = copy.copy(experience[1])
        probs = priorities/(priorities.sum())

        batch   = np.random.choice(len(experience[0]), 
                                   np.min([batch_size, len(experience[0])]),
                                   replace = False, 
                                   p       = probs)

        if self.is_debug:
            if np.unique(batch).shape[0] != np.min([batch_size, len(experience[0])]):
                print("[ERROR] np.unique(batch).shape[0] != np.min([batch_size, len(experience[0])]")

        batch_depth_states           = experience[2][batch]
        batch_gripper_states         = experience[3][batch]
        batch_yaw_states             = experience[4][batch]
        batch_actions                = experience[5][batch]
        batch_gripper_actions        = experience[6][batch]
        batch_next_actions           = experience[7][batch]
        batch_next_gripper_actions   = experience[8][batch]
        batch_rewards                = experience[9][batch]
        batch_next_depth_states      = experience[10][batch]
        batch_next_gripper_states    = experience[11][batch]
        batch_next_yaw_states        = experience[12][batch]
        batch_dones                  = experience[13][batch]
        batch_success_mask           = experience[16][batch]
        
        return batch, batch_depth_states, batch_gripper_states, batch_yaw_states, \
               batch_actions, batch_gripper_actions, batch_next_actions, batch_next_gripper_actions, \
               batch_rewards, batch_next_depth_states, batch_next_gripper_states, batch_next_yaw_states, \
               batch_dones, batch_success_mask

    def save_all_exp_to_dir(self):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        for i in range(max_index):
            data_dict = {
                'depth_state': self.depth_states[i],
                'gripper_state': self.gripper_states[i],
                'yaw_state': self.yaw_states[i],
                'action': self.actions[i],
                'gripper_action': self.gripper_actions[i],
                'next_action': self.next_actions[i],
                'next_gripper_action': self.next_gripper_actions[i],
                'action_type': self.action_types[i],
                'reward': self.rewards[i],
                'next_depth_state': self.next_depth_states[i],
                'next_gripper_state': self.next_gripper_states[i],
                'next_yaw_state': self.next_yaw_states[i],
                'done': self.dones[i],
                'predict_q':  self.predict_qs[i],
                'labeled_q':  self.labeled_qs[i],
                'success_mask':  self.success_mask[i],
                'priority': self.priority[i],
            }

            file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{i}" + ".pkl")
            with open(file_name, 'wb') as file:
                pickle.dump(data_dict, file)

    def save_one_exp_to_dir(self, data_dict):

        if self.data_length >= self.max_memory_size:
            self.data_length  = 0

        file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{self.data_length}" + ".pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)
            print(f"[BUFFER] data saved {self.data_length+1}/{self.max_memory_size}")

            self.data_length += 1

    def load_exp_from_dir(self, checkpt_dir = None):

        if checkpt_dir is None:
            checkpt_dir = self.checkpt_dir

        exp_dir = os.listdir(checkpt_dir)

        #check the data size in hardware storage
        data_length = len(exp_dir)

        print(f"[LOAD BUFFER] data_length: {data_length}")
        #reinitialise memory size if the max memory size is less than data_length
        if self.max_memory_size <= data_length:
            self.init_mem_size(data_length) if self.max_memory_size < data_length else None
            self.max_memory_size = data_length
            self.is_full         = True 
            self.memory_cntr     = self.max_memory_size
        elif self.max_memory_size > data_length:
            self.is_full         = False 
            self.memory_cntr     = data_length

        for i in range(data_length):
            file_name = os.path.join(checkpt_dir, exp_dir[i])
            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.depth_states[i]           = data_dict['depth_state']   
                self.gripper_states[i]         = data_dict['gripper_state']
                self.yaw_states[i]             = data_dict['yaw_state']   
                self.actions[i]                = data_dict['action']            
                self.gripper_actions[i]        = data_dict['gripper_action']   
                self.next_actions[i]           = data_dict['next_action']            
                self.next_gripper_actions[i]   = data_dict['next_gripper_action'] 
                self.action_types[i]           = data_dict['action_type']      
                self.rewards[i]                = data_dict['reward']            
                self.next_depth_states[i]      = data_dict['next_depth_state'] 
                self.next_gripper_states[i]    = data_dict['next_gripper_state']
                self.next_yaw_states[i]        = data_dict['next_yaw_state']
                self.dones[i]                  = data_dict['done']              
                self.predict_qs[i]             = data_dict['predict_q']         
                self.labeled_qs[i]             = data_dict['labeled_q'] 
                self.success_mask[i]           = data_dict['success_mask'] 
                self.priority[i]               = data_dict['priority']  

                if self.action_types[i] == constants.GRASP:
                    self.have_grasp_data = True
                    self.grasp_data_size += 1
                else:
                    self.have_push_data = True
                    self.push_data_size += 1

        print(f"[BUFFER] grasp_data_size: {self.grasp_data_size}")
        print(f"[BUFFER] push_data_size: {self.push_data_size}")

    def update_buffer(self, sample_inds, actor_loss, critic_loss):

        if self.is_debug:
            if actor_loss is not None and len(actor_loss.shape) != len(critic_loss.shape):
                print("[ERROR] len(actor_loss.shape) != len(critic_loss.shape)")
                print("actor_loss.shape: ", actor_loss.shape, "critic_loss.shape: ", critic_loss.shape)
            else:
                if actor_loss is not None and actor_loss.shape[0] != critic_loss.shape[0]:
                    print("[ERROR] actor_loss.shape[0] != critic_loss.shape[0]")
                elif actor_loss is not None and actor_loss.shape[1] != critic_loss.shape[1]:
                    print("[ERROR] actor_loss.shape[1] != critic_loss.shape[1]")

        for i, sample_ind in enumerate(sample_inds):
            if actor_loss is not None:
                self.priority[sample_ind]  = np.abs(actor_loss[i] + self.sm_c)**self.alpha
            else:
                self.priority[sample_ind]  = np.abs(critic_loss[i] + self.sm_c)**self.alpha

        print("[SUCCESS] update experience priorities")