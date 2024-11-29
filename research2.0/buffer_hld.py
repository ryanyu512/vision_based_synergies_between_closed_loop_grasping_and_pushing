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
        self.memory_cntr = 0
        self.img_size = img_size
        #initialise if the memory is full
        self.is_full = False
        #initialise power value for prioritisation
        self.alpha = alpha
        #initialise small constant to prevent division by zero
        self.sm_c = 1e-6
        #initialise number of data in buffer now
        self.N_data = 0
        #initialise prioritised sampling probability
        self.prioritised_prob = prioritised_prob

        #surprise value
        self.priority = np.ones(self.max_memory_size)

        #initialise check point directory
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)
        self.checkpt_dir = os.path.abspath(checkpt_dir)

        #check the data size in hardware storage
        self.data_length = len(os.listdir(self.checkpt_dir))

        self.is_debug = is_debug

        try:
            self.load_exp_from_dir()
            print("[SUCCESS] load previous buffer")
        except:
            print("[FAIL] cannot load previous buffer")

    def init_mem_size(self, max_memory_size):

        self.max_memory_size      = max_memory_size
        self.memory_cntr          = 0

        #surprise value
        self.priority = np.ones(self.max_memory_size)

    def store_transition(self, 
                         depth_state, 
                         action_type, 
                         reward, 
                         next_depth_state, 
                         done, 
                         critic_loss,
                         is_save_to_dir = True):

        self.N_data += 1
        if self.N_data >= self.max_memory_size:
            self.N_data = self.max_memory_size

        priority = np.abs(critic_loss + self.sm_c)**self.alpha
        self.priority[self.memory_cntr] = priority

        if is_save_to_dir:
            data_dict = {
                'depth_state': depth_state,
                'action_type': action_type,
                'reward': reward,
                'next_depth_state': next_depth_state,
                'done': done,
                'priority': priority,
            }

            self.add_one_exp_to_dir(data_dict)

    def get_experience(self):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        #get priorities
        priorities = self.priority[:max_index]

        return priorities
    
    def sample_buffer(self, batch_size):

        #          0
        # priorities

        experience = self.get_experience()

        if experience.sum() == 0 or random.random() >= self.prioritised_prob:
            priorities = np.ones_like(experience)
        else:
            priorities = copy.copy(experience)
        probs = priorities/(priorities.sum())

        batch = np.random.choice(len(experience), 
                                 np.min([batch_size, len(experience)]),
                                 replace = False, 
                                 p = probs)

        self.load_batch_exp_from_dir(batch)

        return batch, \
            self.batch_depth_states, \
            self.batch_action_types, \
            self.batch_rewards, \
            self.batch_next_depth_states, \
            self.batch_dones

    def add_one_exp_to_dir(self, data_dict):

        file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{self.memory_cntr}" + ".pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)
            print(f"[HLD BUFFER] data saved {self.memory_cntr+1}/{self.max_memory_size}")

        #update memory counter
        self.memory_cntr += 1

        if self.memory_cntr >= self.max_memory_size:
            self.is_full = True
            self.memory_cntr = 0

        #save memory counter
        data_dict_mem_counter = {'memory_cntr': self.memory_cntr}
        file_name = os.path.join(self.checkpt_dir, "memory_cntr.pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict_mem_counter, file)

    def update_one_exp_to_dir(self, data_dict, sample_index):

        file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{sample_index}" + ".pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)

    def load_batch_exp_from_dir(self, batch_index, checkpt_dir = None):
        if checkpt_dir is None:
            checkpt_dir = self.checkpt_dir

        # #get all the file names in the checkpoint directory
        # exp_dir = os.listdir(self.checkpt_dir)
        # try:
        #     exp_dir.remove('memory_cntr.pkl')
        # except:
        #     print("no memory_cntr.pkl")
        # exp_sort_index = np.argsort([int(e.split('.')[0].split('_')[-1]) for e in exp_dir])
        # sort_exp_dir = np.array(exp_dir)[exp_sort_index]

        #current depth state
        self.batch_depth_states = np.zeros((len(batch_index), self.img_size, self.img_size))
        #current action type
        self.batch_action_types = np.ones(len(batch_index))*-1
        #next reward
        self.batch_rewards = np.zeros(len(batch_index))
        #next state
        self.batch_next_depth_states = np.zeros((len(batch_index), self.img_size, self.img_size))
        #is done in the next state
        self.batch_dones = np.zeros(len(batch_index), dtype = bool)

        for i in range(len(batch_index)):
            # file_name = os.path.join(self.checkpt_dir, sort_exp_dir[batch_index[i]])
            file_name = os.path.join(checkpt_dir, f"experience_data_{batch_index[i]}.pkl")

            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.batch_depth_states[i] = data_dict['depth_state']   
                self.batch_action_types[i] = data_dict['action_type']      
                self.batch_rewards[i] = data_dict['reward']            
                self.batch_next_depth_states[i] = data_dict['next_depth_state'] 
                self.batch_dones[i] = data_dict['done']   

    def load_exp_from_dir(self, checkpt_dir = None):

        if checkpt_dir is None:
            checkpt_dir = self.checkpt_dir

        #get all the file names in the checkpoint directory
        exp_dir = os.listdir(self.checkpt_dir)
        try:
            exp_dir.remove('memory_cntr.pkl')
        except:
            print("no memory_cntr.pkl")
        exp_sort_index = np.argsort([int(e.split('.')[0].split('_')[-1]) for e in exp_dir])
        sort_exp_dir = np.array(exp_dir)[exp_sort_index]

        #check the data size in hardware storage
        data_length = len(sort_exp_dir)

        print(f"[LOAD HLD BUFFER] data_length: {data_length}")
        #reinitialise memory size if the max memory size is less than data_length
        if self.max_memory_size == data_length:  
            self.is_full = True 
            try:
                file_name = os.path.join(checkpt_dir, "memory_cntr.pkl")
                with open(file_name, 'rb') as file:
                    data_dict = pickle.load(file)
                    self.memory_cntr = data_dict['memory_cntr']
                    print(f"hld memory_cntr: {self.memory_cntr}")
            except:
                self.memory_cntr  = 0
        elif self.max_memory_size < data_length:
            self.init_mem_size(data_length)  
            self.max_memory_size = data_length
            self.is_full = True 
            self.memory_cntr = self.max_memory_size
        elif self.max_memory_size > data_length:
            self.is_full = False 
            self.memory_cntr = data_length

        for i in range(data_length):
            file_name = os.path.join(self.checkpt_dir, sort_exp_dir[i])
            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)
 
                self.priority[i] = data_dict['priority'] 

            self.N_data += 1

    def update_buffer(self, sample_inds, critic_loss):

        for i, sample_ind in enumerate(sample_inds):
            self.priority[sample_ind]  = np.abs(critic_loss[i] + self.sm_c)**self.alpha

            data_dict = {
                'depth_state': self.batch_depth_states[i],
                'action_type': self.batch_action_types[i],
                'reward': self.batch_rewards[i],
                'next_depth_state': self.batch_next_depth_states[i],
                'done': self.batch_dones[i],
                'priority': self.priority[sample_ind],
            }

            self.update_one_exp_to_dir(data_dict, sample_ind)

        print("[SUCCESS] update experience priorities")

    # def save_all_exp_to_dir(self):

    #     #get max_ind for sampling range
    #     max_index = self.max_memory_size if self.is_full else self.memory_cntr

    #     for i in range(max_index):
    #         data_dict = dict()

    #         data_dict['depth_state']      = self.depth_states[i]
    #         data_dict['action_type']      = self.action_types[i]
    #         data_dict['reward']           = self.rewards[i]
    #         data_dict['next_depth_state'] = self.next_depth_states[i]
    #         data_dict['done']             = self.dones[i]
    #         data_dict['priority']         = self.priority[i]

    #         file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{i}" + ".pkl")
    #         with open(file_name, 'wb') as file:
    #             pickle.dump(data_dict, file)