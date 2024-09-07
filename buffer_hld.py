import os
import pickle 
import numpy as np

class BufferReplay_HLD():

    def __init__(self, 
                 max_memory_size = int(1000), 
                 img_size        = 128, 
                 alpha           = 0.6,
                 checkpt_dir     = 'logs/demo_exp_hld'): 
      
        self.max_memory_size = max_memory_size
        self.memory_cntr     = 0
        self.img_size        = img_size
        #initialise if the memory is full
        self.is_full         = False
        #initialise power value for prioritisation
        self.alpha           = alpha
        #initialise small constant to prevent division by zero
        self.sm_c            = 1e-6

        #current depth state
        self.depth_states      = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current action type
        self.action_types      = np.ones(self.max_memory_size)*-1
        #next action type
        self.next_action_types = np.ones(self.max_memory_size)*-1
        #next reward
        self.rewards           = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #is done in the next state
        self.dones             = np.zeros(self.max_memory_size, dtype = bool)
        #predicted q value: 
        self.predict_qs        = np.zeros(self.max_memory_size)
        #labeled q value: 
        self.labeled_qs        = np.zeros(self.max_memory_size)
        #predicted next q value: 
        self.predict_next_qs   = np.zeros(self.max_memory_size)
        #labeled next q value: 
        self.labeled_next_qs   = np.zeros(self.max_memory_size)
        #surprise value
        self.priority          = np.ones(self.max_memory_size)


        #initialise check point directory
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)
        self.checkpt_dir       = os.path.abspath(checkpt_dir)

        #check the data size in hardware storage
        self.data_length = len(os.listdir(self.checkpt_dir))

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
        #next action type
        self.next_action_types = np.ones(self.max_memory_size)*-1
        #next reward
        self.rewards           = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #is done in the next state
        self.dones             = np.zeros(self.max_memory_size, dtype = bool)
        #predicted q value: 
        self.predict_qs        = np.zeros(self.max_memory_size)
        #labeled q value: 
        self.labeled_qs        = np.zeros(self.max_memory_size)
        #predicted next q value: 
        self.predict_next_qs   = np.zeros(self.max_memory_size)
        #labeled next q value: 
        self.labeled_next_qs   = np.zeros(self.max_memory_size)
        #surprise value
        self.priority          = np.ones(self.max_memory_size)

    def store_transition(self, 
                         depth_state, action_type, reward, 
                         next_depth_state, next_action_type, done, 
                         predict_q, labeled_q,
                         predict_next_q, labeled_next_q, 
                         is_save_to_dir = True):

        #update memory
        if self.memory_cntr >= self.max_memory_size:
            self.is_full = True
            self.memory_cntr = 0

        priority = np.abs(predict_q - labeled_q + self.sm_c)**self.alpha
        self.depth_states[self.memory_cntr]      = depth_state
        self.action_types[self.memory_cntr]      = action_type
        self.rewards[self.memory_cntr]           = reward
        self.next_depth_states[self.memory_cntr] = next_depth_state
        self.next_action_types[self.memory_cntr] = next_action_type
        self.dones[self.memory_cntr]             = done
        self.predict_qs[self.memory_cntr]        = predict_q
        self.labeled_qs[self.memory_cntr]        = labeled_q
        self.predict_next_qs[self.memory_cntr]   = predict_next_q
        self.labeled_next_qs[self.memory_cntr]   = labeled_next_q
        self.priority[self.memory_cntr]          = priority

        if is_save_to_dir:
            data_dict = {
                'depth_state': depth_state,
                'action_type': action_type,
                'reward': reward,
                'next_depth_state': next_depth_state,
                'next_action_type': next_action_type,
                'done': done,
                'predict_q':  predict_q,
                'labeled_q':  labeled_q,
                'predict_next_q':  predict_next_q,
                'labeled_next_q':  labeled_next_q,
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
        next_action_types = self.next_action_types[:max_index]
        dones             = self.dones[:max_index]
        predict_qs        = self.predict_qs[:max_index]
        labeled_qs        = self.labeled_qs[:max_index]
        predict_next_qs   = self.predict_next_qs[:max_index]
        labeled_next_qs   = self.labeled_next_qs[:max_index]

        return priorities, depth_states, action_types, rewards, next_depth_states, \
               next_action_types, dones, predict_qs, labeled_qs, predict_next_qs, \
               labeled_next_qs

    def sample_buffer(self, batch_size):

        #          0,            1,            2,       3,                 4,     
        # priorities, depth_states, action_types, rewards, next_depth_states, 
        #                 5,     6,          7,          8,               9,              10
        # next_action_types, dones, predict_qs, labeled_qs, predict_next_qs, labeled_next_qs
        experience = self.get_experience()
        priorities = experience[0]
        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)
        probs = priorities/(priorities.sum())

        batch   = np.random.choice(len(experience[0]), 
                                   batch_size,
                                   replace = False, 
                                   p       = probs)

        batch_depth_states      = experience[1][batch]
        batch_action_types      = experience[2][batch]
        batch_rewards           = experience[3][batch]
        batch_next_depth_states = experience[4][batch]
        batch_next_action_types = experience[5][batch]
        batch_dones             = experience[6][batch]
        batch_predict_qs        = experience[7][batch]
        batch_labeled_qs        = experience[8][batch]
        batch_predict_next_qs   = experience[9][batch]
        batch_labeled_next_qs   = experience[10][batch]      

        return batch, batch_depth_states, batch_action_types, batch_rewards, \
               batch_next_depth_states, batch_next_action_types, batch_dones, batch_predict_qs, \
               batch_labeled_qs, batch_predict_next_qs, batch_labeled_next_qs

    def save_all_exp_to_dir(self):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        for i in range(max_index):
            data_dict = dict()

            data_dict['depth_state']      = self.depth_states[i]
            data_dict['action_type']      = self.action_types[i]
            data_dict['reward']           = self.rewards[i]
            data_dict['next_depth_state'] = self.next_depth_states[i]
            data_dict['next_action_type'] = self.next_action_types[i]
            data_dict['done']             = self.dones[i]
            data_dict['predict_q']        = self.predict_qs[i]
            data_dict['labeled_q']        = self.labeled_qs[i]
            data_dict['predict_next_q']   = self.predict_next_qs[i]
            data_dict['labeled_next_q']   = self.labeled_next_qs[i]
            data_dict['priority']         = self.priority[i]

            file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{i}" + ".pkl")
            with open(file_name, 'wb') as file:
                pickle.dump(data_dict, file)

    def save_one_exp_to_dir(self, data_dict):

        file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{self.data_length}" + ".pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)
            self.data_length += 1

    def load_exp_from_dir(self):

        #get all the file names in the checkpoint directory
        exp_dir     = os.listdir(self.checkpt_dir)

        #check the data size in hardware storage
        data_length = len(exp_dir)

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
                self.next_action_types[i] = data_dict['next_action_type']
                self.dones[i]             = data_dict['done']   
                self.predict_qs[i]        = data_dict['predict_q']         
                self.labeled_qs[i]        = data_dict['labeled_q'] 
                self.predict_next_qs[i]   = data_dict['predict_next_q']         
                self.labeled_next_qs[i]   = data_dict['labeled_next_q'] 
                self.priority[i]          = data_dict['priority'] 

    def update_buffer(self, sample_ind, predict_q):
        
        self.predict_qs[sample_ind] = predict_q
        self.priority[sample_ind]  = np.abs(self.predict_qs[sample_ind] - self.labeled_qs[sample_ind] + self.sm_c)**self.alpha
