import os
import copy
import pickle 
import random
import numpy as np

class BufferReplay_HLD():

    def __init__(self, 
                 max_memory_size  = int(1000), 
                 img_size         = 128, 
                 alpha            = 0.6,
                 prioritised_prob = 0.8,
                 checkpt_dir      = 'logs/exp_hld',
                 is_debug         = True): 
      
        self.max_memory_size = max_memory_size
        self.memory_cntr     = 0
        self.img_size        = img_size
        #initialise if the memory is full
        self.is_full         = False
        #initialise power value for prioritisation
        self.alpha           = alpha
        #initialise small constant to prevent division by zero
        self.sm_c            = 1e-6

        self.N_data          = 0

        #current depth state
        self.depth_states      = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current action type
        self.action_types      = np.ones(self.max_memory_size)*-1
        #next reward
        self.rewards           = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #is done in the next state
        self.dones             = np.zeros(self.max_memory_size, dtype = bool)
        #surprise value
        self.priority          = np.ones(self.max_memory_size)
        #initialise prioritised sampling probability
        self.prioritised_prob  = prioritised_prob

        #initialise check point directory
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)
        self.checkpt_dir       = os.path.abspath(checkpt_dir)

        #check the data size in hardware storage
        self.data_length = len(os.listdir(self.checkpt_dir))

        self.is_debug    = is_debug

        try:
            self.load_exp_from_dir()
            print("[SUCCESS] load previous buffer")
        except:
            print("[FAIL] cannot load previous buffer")

    def init_mem_size(self, max_memory_size):

        self.max_memory_size      = max_memory_size
        self.memory_cntr          = 0

        #current depth state
        self.depth_states      = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current action type
        self.action_types      = np.ones(self.max_memory_size)*-1
        #next reward
        self.rewards           = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #is done in the next state
        self.dones             = np.zeros(self.max_memory_size, dtype = bool)
        #surprise value
        self.priority          = np.ones(self.max_memory_size)

    def store_transition(self, 
                         depth_state, action_type, reward, 
                         next_depth_state, 
                         done, critic_loss,
                         is_save_to_dir = True):

        #update memory
        if self.memory_cntr >= self.max_memory_size:
            self.is_full = True
            self.memory_cntr = 0

        self.N_data += 1
        if self.N_data >= self.max_memory_size:
            self.N_data = self.max_memory_size

        priority = np.abs(critic_loss + self.sm_c)**self.alpha
        self.depth_states[self.memory_cntr]      = depth_state
        self.action_types[self.memory_cntr]      = action_type
        self.rewards[self.memory_cntr]           = reward
        self.next_depth_states[self.memory_cntr] = next_depth_state
        self.dones[self.memory_cntr]             = done
        self.priority[self.memory_cntr]          = priority

        if is_save_to_dir:
            data_dict = {
                'depth_state': depth_state,
                'action_type': action_type,
                'reward': reward,
                'next_depth_state': next_depth_state,
                'done': done,
                'priority': priority,
            }

            self.save_one_exp_to_dir(data_dict)

        #update memory counter
        self.memory_cntr += 1

    def get_experience(self):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        #get priorities
        priorities        = self.priority[:max_index]

        depth_states      = self.depth_states[:max_index]
        action_types      = self.action_types[:max_index]
        rewards           = self.rewards[:max_index]
        next_depth_states = self.next_depth_states[:max_index]
        dones             = self.dones[:max_index]

        return priorities, depth_states, action_types, rewards, next_depth_states, \
               dones
    
    def sample_buffer(self, batch_size):

        #          0,            1,            2,       3,                 4,     5
        # priorities, depth_states, action_types, rewards, next_depth_states, dones

        experience = self.get_experience()

        if experience[0].sum() == 0 or random.random() >= self.prioritised_prob:
            priorities = np.ones_like(experience[0])
        else:
            priorities = copy.copy(experience[0])
        probs = priorities/(priorities.sum())

        batch   = np.random.choice(len(experience[0]), 
                                   np.min([batch_size, len(experience[0])]),
                                   replace = False, 
                                   p       = probs)

        batch_depth_states      = experience[1][batch]
        batch_action_types      = experience[2][batch]
        batch_rewards           = experience[3][batch]
        batch_next_depth_states = experience[4][batch]
        batch_dones             = experience[5][batch]

        return batch, batch_depth_states, batch_action_types, batch_rewards, batch_next_depth_states, batch_dones

    def save_all_exp_to_dir(self):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        for i in range(max_index):
            data_dict = dict()

            data_dict['depth_state']      = self.depth_states[i]
            data_dict['action_type']      = self.action_types[i]
            data_dict['reward']           = self.rewards[i]
            data_dict['next_depth_state'] = self.next_depth_states[i]
            data_dict['done']             = self.dones[i]
            data_dict['priority']         = self.priority[i]

            file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{i}" + ".pkl")
            with open(file_name, 'wb') as file:
                pickle.dump(data_dict, file)

    def save_one_exp_to_dir(self, data_dict):

        if self.data_length >= self.max_memory_size:
            self.data_length  = 0

        file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{self.data_length}" + ".pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)
            self.data_length += 1

    def load_exp_from_dir(self, checkpt_dir = None):

        if checkpt_dir is None:
            checkpt_dir = self.checkpt_dir

        #get all the file names in the checkpoint directory
        exp_dir     = os.listdir(self.checkpt_dir)

        #check the data size in hardware storage
        data_length = len(exp_dir)

        print(f"[LOAD HLD BUFFER] data_length: {data_length}")
        #reinitialise memory size if the max memory size is less than data_length
        if self.max_memory_size <= data_length:
            if self.max_memory_size < data_length:
                self.init_mem_size(data_length)  
            self.max_memory_size = data_length
            self.is_full         = True 
            self.memory_cntr     = self.max_memory_size
        elif self.max_memory_size > data_length:
            self.is_full         = False 
            self.memory_cntr     = data_length

        for i in range(data_length):
            file_name = os.path.join(self.checkpt_dir, exp_dir[i])
            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.depth_states[i]      = data_dict['depth_state']   
                self.action_types[i]      = data_dict['action_type']      
                self.rewards[i]           = data_dict['reward']            
                self.next_depth_states[i] = data_dict['next_depth_state'] 
                self.dones[i]             = data_dict['done']   
                self.priority[i]          = data_dict['priority'] 

            self.N_data += 1

    def update_buffer(self, sample_inds, critic_loss):

        for i, sample_ind in enumerate(sample_inds):
                self.priority[sample_ind]  = np.abs(critic_loss[i] + self.sm_c)**self.alpha

        print("[SUCCESS] update experience priorities")
