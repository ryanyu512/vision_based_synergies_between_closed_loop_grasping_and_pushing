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
      
        self.have_grasp_data = False
        self.have_push_data = False
        self.grasp_data_size = 0
        self.push_data_size = 0
        self.max_memory_size = max_memory_size
        self.memory_cntr = 0
        self.img_size = img_size
        self.N_action = N_action
        self.N_gripper_action_type  = N_gripper_action_type
        #initialise if the memory is full
        self.is_full = False
        #initialise power value for prioritisation
        self.alpha = alpha
        #initialise small constant to prevent division by zero
        self.sm_c = 1e-6
        #initialise prioritised sampling probability
        self.prioritised_prob = prioritised_prob

        #current action type
        self.action_types = np.ones(self.max_memory_size)*-1
        #surprise value
        self.priority = np.ones(self.max_memory_size)

        #initialise check point directory
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)
        self.checkpt_dir = os.path.abspath(checkpt_dir)

        #check the data size in hardware storage
        self.data_length = len(os.listdir(self.checkpt_dir))

        self.is_debug = is_debug

        # try:
        #     if load_checkpt_dir is None:
        self.load_exp_from_dir()
        #     else:
        #         self.load_exp_from_dir(load_checkpt_dir)
        #         print("[SUCCESS] load low-level buffer")
        # except:
        #     print("[FAIL] cannot load low-level buffer")

    def init_mem_size(self, max_memory_size):

        self.max_memory_size = max_memory_size
        self.memory_cntr = 0

        #current action type
        self.action_types = np.ones(self.max_memory_size)*-1
        #surprise value
        self.priority = np.ones(self.max_memory_size)

    def store_transition(self, 
                         depth_state, 
                         gripper_state, 
                         yaw_state, 
                         action, 
                         gripper_action, 
                         next_action, 
                         next_gripper_action,
                         action_type,reward, 
                         next_depth_state, 
                         next_gripper_state, 
                         next_yaw_state, 
                         done, 
                         is_success, 
                         priority,
                         is_save_to_dir = True):

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

        self.action_types[self.memory_cntr] = action_type
        self.priority[self.memory_cntr] = priority_

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
                'success_mask':  is_success,
                'priority': priority,
            }

            self.add_one_exp_to_dir(data_dict)

    def get_experience_by_action_type(self, action_type):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        #get index related to the action_type
        action_index = ((self.action_types == action_type).nonzero())[0]

        #get priorities
        priorities = self.priority[:max_index][action_index]

        return action_index, priorities

    def sample_buffer(self, batch_size, action_type):
        #            0,          1             
        # action_index, priorities

        experience = self.get_experience_by_action_type(action_type)

        if experience[1].sum() == 0 or random.random() >= self.prioritised_prob:
            priorities = np.ones_like(experience[1])
        else:
            priorities = copy.copy(experience[1])
        probs = priorities/(priorities.sum())

        batch = np.random.choice(len(experience[0]), 
                                 np.min([batch_size, len(experience[0])]),
                                 replace = False, 
                                 p = probs)

        if self.is_debug:
            if np.unique(batch).shape[0] != np.min([batch_size, len(experience[0])]):
                print("[ERROR] np.unique(batch).shape[0] != np.min([batch_size, len(experience[0])]")

        self.load_batch_exp_from_dir(experience[0], batch)
        
        return self.batch, self.batch_depth_states, self.batch_gripper_states, self.batch_yaw_states, \
               self.batch_actions, self.batch_gripper_actions, self.batch_next_actions, self.batch_next_gripper_actions, \
               self.batch_rewards, self.batch_next_depth_states, self.batch_next_gripper_states, self.batch_next_yaw_states, \
               self.batch_dones, self.batch_success_mask

    def add_one_exp_to_dir(self, data_dict):

        file_name = os.path.join(self.checkpt_dir, "experience_data" + f"_{self.memory_cntr}" + ".pkl")

        with open(file_name, 'wb') as file:
            pickle.dump(data_dict, file)
            print(f"[BUFFER] data saved {self.memory_cntr+1}/{self.max_memory_size}")

        #update memory counter
        self.memory_cntr += 1

        #update memory
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

    def load_batch_exp_from_dir(self, action_index, batch_index, checkpt_dir = None):

        if checkpt_dir is None:
            checkpt_dir = self.checkpt_dir

        # #get experience directory
        # exp_dir = os.listdir(checkpt_dir)
        # try:
        #     exp_dir.remove('memory_cntr.pkl')
        # except:
        #     print("no memory_cntr.pkl")
        # exp_sort_index = np.argsort([int(e.split('.')[0].split('_')[-1]) for e in exp_dir])
        # sort_exp_dir = np.array(exp_dir)[exp_sort_index]

        #get batch size
        batch_size = len(batch_index)
        
        #initialise batch index
        self.batch = np.ones(batch_size, dtype=int)*-1
        #current depth state
        self.batch_depth_states = np.zeros((batch_size, self.img_size, self.img_size))
        #current gripper state 
        self.batch_gripper_states = np.zeros(batch_size)
        #current yaw angles
        self.batch_yaw_states = np.zeros(batch_size)
        #current action
        self.batch_actions = np.zeros((batch_size, self.N_action))
        #current gripper action
        self.batch_gripper_actions = np.zeros((batch_size))
        #next action
        self.batch_next_actions = np.zeros((batch_size, self.N_action))
        #next gripper action
        self.batch_next_gripper_actions = np.zeros((batch_size))
        # current action type
        self.batch_action_types = np.ones(batch_size)*-1
        #next reward
        self.batch_rewards = np.zeros(batch_size)
        #next state
        self.batch_next_depth_states = np.zeros((batch_size, self.img_size, self.img_size))
        #current gripper state 
        self.batch_next_gripper_states = np.zeros(batch_size)
        #current yaw angles
        self.batch_next_yaw_states = np.zeros(batch_size)
        #is done in the next state
        self.batch_dones = np.zeros(batch_size, dtype = bool)
        #indicate if this experience is a successful experience
        self.batch_success_mask = np.zeros(batch_size, dtype = bool)

        # is_same = True
        for i in range(len(batch_index)):
            self.batch[i] = action_index[batch_index[i]]
            # file_name_check = os.path.join(checkpt_dir, sort_exp_dir[self.batch[i]])
            file_name = os.path.join(checkpt_dir, f"experience_data_{self.batch[i]}.pkl")

            # if file_name != file_name_check:
            #     is_same = False
            #     print("[ERROR] is_same = False")

            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.batch_depth_states[i] = data_dict['depth_state']   
                self.batch_gripper_states[i] = data_dict['gripper_state']
                self.batch_yaw_states[i] = data_dict['yaw_state']   
                self.batch_actions[i] = data_dict['action']            
                self.batch_gripper_actions[i] = data_dict['gripper_action']   
                self.batch_next_actions[i] = data_dict['next_action']            
                self.batch_next_gripper_actions[i] = data_dict['next_gripper_action'] 
                self.batch_action_types[i] = data_dict['action_type']      
                self.batch_rewards[i] = data_dict['reward']            
                self.batch_next_depth_states[i] = data_dict['next_depth_state'] 
                self.batch_next_gripper_states[i] = data_dict['next_gripper_state']
                self.batch_next_yaw_states[i] = data_dict['next_yaw_state']
                self.batch_dones[i] = data_dict['done']              
                self.batch_success_mask[i] = data_dict['success_mask'] 

    def load_exp_from_dir(self, checkpt_dir = None):

        if checkpt_dir is None:
            checkpt_dir = self.checkpt_dir

        exp_dir = os.listdir(checkpt_dir)
        try:
            exp_dir.remove('memory_cntr.pkl')
        except:
            print("no memory_cntr.pkl")
        exp_sort_index = np.argsort([int(e.split('.')[0].split('_')[-1]) for e in exp_dir])
        sort_exp_dir = np.array(exp_dir)[exp_sort_index]

        #check the data size in hardware storage
        data_length = len(sort_exp_dir)

        print(f"[LOAD BUFFER] data_length: {data_length}")
        #reinitialise memory size if the max memory size is less than data_length
        if self.max_memory_size == data_length:
            self.is_full         = True
            try:
                file_name = os.path.join(checkpt_dir, "memory_cntr.pkl")
                with open(file_name, 'rb') as file:
                    data_dict = pickle.load(file)
                    self.memory_cntr = data_dict['memory_cntr']
                    print(f"lla memory_cntr: {self.memory_cntr}")
            except:
                self.max_memory_cntr = 0
        elif self.max_memory_size < data_length:
            self.init_mem_size(data_length)
            self.max_memory_size = data_length
            self.is_full         = True 
            self.memory_cntr     = self.max_memory_size
        elif self.max_memory_size > data_length:
            self.is_full         = False 
            self.memory_cntr     = data_length

        for i in range(data_length):
            file_name = os.path.join(checkpt_dir, sort_exp_dir[i])
            with open(file_name, 'rb') as file:
                data_dict = pickle.load(file)

                self.action_types[i] = data_dict['action_type']      
                self.priority[i] = data_dict['priority']  

                if self.action_types[i] == constants.GRASP:
                    self.have_grasp_data = True
                    self.grasp_data_size += 1
                else:
                    self.have_push_data = True
                    self.push_data_size += 1

        print(f"[BUFFER] grasp_data_size: {self.grasp_data_size}")
        print(f"[BUFFER] push_data_size: {self.push_data_size}")

    def update_buffer(self, sample_inds, actor_loss, critic_loss):

        # if self.is_debug:
        #     if actor_loss is not None and len(actor_loss.shape) != len(critic_loss.shape):
        #         print("[ERROR] len(actor_loss.shape) != len(critic_loss.shape)")
        #         print("actor_loss.shape: ", actor_loss.shape, "critic_loss.shape: ", critic_loss.shape)
        #     else:
        #         if actor_loss is not None and actor_loss.shape[0] != critic_loss.shape[0]:
        #             print("[ERROR] actor_loss.shape[0] != critic_loss.shape[0]")
        #         elif actor_loss is not None and actor_loss.shape[1] != critic_loss.shape[1]:
        #             print("[ERROR] actor_loss.shape[1] != critic_loss.shape[1]")

        for i, sample_ind in enumerate(sample_inds):

            #update priority
            if actor_loss is not None:
                self.priority[sample_ind] = np.abs(actor_loss[i] + self.sm_c)**self.alpha
            else:
                self.priority[sample_ind] = np.abs(critic_loss[i] + self.sm_c)**self.alpha
            
            if self.batch_action_types[0] != self.batch_action_types[i]:
                print("[ERROR] not the same action type")

            #update experience to harddisk
            data_dict = {
                'depth_state': self.batch_depth_states[i],
                'gripper_state': self.batch_gripper_states[i],
                'yaw_state': self.batch_yaw_states[i],
                'action': self.batch_actions[i],
                'gripper_action': self.batch_gripper_actions[i],
                'next_action': self.batch_next_actions[i],
                'next_gripper_action': self.batch_next_gripper_actions[i],
                'action_type': self.batch_action_types[i],
                'reward': self.batch_rewards[i],
                'next_depth_state': self.batch_next_depth_states[i],
                'next_gripper_state': self.batch_next_gripper_states[i],
                'next_yaw_state': self.batch_next_yaw_states[i],
                'done': self.batch_dones[i],
                'success_mask':  self.batch_success_mask[i],
                'priority': self.priority[sample_ind],
            }

            self.update_one_exp_to_dir(data_dict, sample_ind)

        print("[SUCCESS] update experience priorities")