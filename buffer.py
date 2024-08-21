import numpy as np
import pickle 

class BufferReplay():

    def __init__(self, 
                 max_memory_size        = int(1000), 
                 img_size               = 128, 
                 N_action               = 4, 
                 N_gripper_action_type  = 2,
                 alpha                  = 0.6): 
      
        self.max_memory_size        = max_memory_size
        self.memory_cntr            = 0
        self.img_size               = img_size
        self.N_action               = N_action
        self.N_gripper_action_type  = N_gripper_action_type

        #record if this state taken at home position 
        self.is_home           = np.zeros(self.max_memory_size, dtype = bool)
        #current depth state
        self.depth_states      = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #current gripper state 
        self.gripper_states    = np.zeros(self.max_memory_size)
        #current yaw angles
        self.yaw_states        = np.zeros(self.max_memory_size)
        #current action
        self.actions           = np.zeros((self.max_memory_size, self.N_action))
        #current gripper action
        self.gripper_actions   = np.zeros((self.max_memory_size, self.N_gripper_action_type))
        #current action type
        self.action_types      = np.ones((self.max_memory_size))*-1
        #next reward
        self.rewards           = np.zeros(self.max_memory_size)
        #next state
        self.next_depth_states = np.zeros((self.max_memory_size, self.img_size, self.img_size))
        #is done in the next state
        self.dones             = np.zeros(self.max_memory_size, dtype = bool)
        #predicted q value: (q1 + q2)/2.
        self.predict_qs        = np.zeros(self.max_memory_size)
        #labeled q value: reward + gamma*min(target_q1, target_q2)
        self.labeled_qs        = np.zeros(self.max_memory_size)
        #surprise value
        self.priority          = np.ones(self.max_memory_size)
        #initialise if the memory is full
        self.is_full           = False
        #initialise power value for prioritisation
        self.alpha             = alpha
        #initialise small constant to prevent division by zero
        self.sm_c              = 1e-6

    def store_transition(self, is_home, depth_state, gripper_state, yaw_state, action, gripper_action, action_type, reward, next_depth_state, done, predict_q, labeled_q):

        #update memory
        if self.memory_cntr >= self.max_memory_size:
            self.is_full = True
            self.memory_cntr = 0

        self.is_home[self.memory_cntr]           = is_home
        self.depth_states[self.memory_cntr]      = depth_state
        self.gripper_states[self.memory_cntr]    = gripper_state
        self.yaw_states[self.memory_cntr]        = yaw_state
        self.actions[self.memory_cntr]           = action
        self.gripper_actions[self.memory_cntr]   = gripper_action
        self.action_types[self.memory_cntr]      = action_type
        self.rewards[self.memory_cntr]           = reward
        self.next_depth_states[self.memory_cntr] = next_depth_state
        self.dones[self.memory_cntr]             = done
        self.predict_qs[self.memory_cntr]        = predict_q
        self.labeled_qs[self.memory_cntr]        = labeled_q
        self.priority[self.memory_cntr]          = np.abs(predict_q - labeled_q + self.sm_c)**self.alpha

        #update memory counter
        self.memory_cntr += 1

    def get_experience_by_home_type(self):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        #get index related to the action_type
        action_index = ((self.is_home == 1).nonzero())[0]
        depth_states = self.depth_states[:max_index][action_index]
        action_type  = self.action_types[:max_index][action_index]

        return action_index, depth_states, action_type

    def get_experience_by_action_type(self, action_type):

        #get max_ind for sampling range
        max_index = self.max_memory_size if self.is_full else self.memory_cntr

        #get index related to the action_type
        action_index = ((self.action_types == action_type).nonzero())[0]

        #get priorities
        priorities        = self.priority[:max_index][action_index]

        depth_states      = self.depth_states[:max_index][action_index]
        gripper_states    = self.gripper_states[:max_index][action_index]
        yaw_states        = self.yaw_states[:max_index][action_index]
        actions           = self.actions[:max_index][action_index]
        gripper_actions   = self.gripper_actions[:max_index][action_index]
        rewards           = self.rewards[:max_index][action_index]
        next_depth_states = self.next_depth_states[:max_index][action_index]
        dones             = self.dones[:max_index][action_index]
        predict_qs        = self.predict_qs[:max_index][action_index]
        labeled_qs        = self.labeled_qs[:max_index][action_index]

        return action_index, priorities, depth_states, gripper_states, yaw_states, actions, gripper_actions, rewards, next_depth_states, dones, predict_qs, labeled_qs

    def sample_buffer(self, batch_size, action_type):
        #            0,          1,            2,              3,          4,       5,               6,       7,                 8,     9,         10,         11
        # action_index, priorities, depth_states, gripper_states, yaw_states, actions, gripper_actions, rewards, next_depth_states, dones, predict_qs, labeled_qs
        experience = self.get_experience_by_action_type(action_type)

        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)
        probs = priorities/(priorities.sum())

        batch   = np.random.choice(len(experience[0]), 
                                   batch_size,
                                   replace = False, 
                                   p       = probs)

        batch_depth_states      = experience[2][batch]
        batch_gripper_states    = experience[3][batch]
        batch_yaw_states        = experience[4][batch]
        batch_actions           = experience[5][batch]
        batch_gripper_actions   = experience[6][batch]
        batch_rewards           = experience[7][batch]
        batch_next_depth_states = experience[8][batch]
        batch_dones             = experience[9][batch]

        return batch, batch_depth_states, batch_actions, batch_gripper_actions, batch_rewards, batch_next_depth_states, batch_dones

    def save_buffer(self):
        
        data_dict = dict()
        data_dict['is_home']           = self.is_home
        data_dict['depth_states']      = self.depth_states
        data_dict['gripper_states']    = self.gripper_states
        data_dict['yaw_states']        = self.yaw_states
        data_dict['actions']           = self.actions
        data_dict['gripper_actions']   = self.gripper_actions
        data_dict['action_types']      = self.action_types
        data_dict['rewards']           = self.rewards
        data_dict['next_depth_states'] = self.next_depth_states
        data_dict['dones']             = self.dones
        data_dict['predict_qs']        = self.predict_qs
        data_dict['labeled_qs']        = self.labeled_qs
        data_dict['priority']          = self.priority
        data_dict['is_full']           = self.is_full
        data_dict['memory_cntr']       = self.memory_cntr

        with open("experience_data.pkl", 'wb') as file:
            pickle.dump(data_dict, file)

    def load_buffer(self):

        data_dict = dict()
        
        with open("experience_data.pkl", 'rb') as file:
            data_dict = pickle.load(file)

            self.is_home           = data_dict['is_home']   
            self.depth_states      = data_dict['depth_states']   
            self.gripper_states    = data_dict['gripper_states']
            self.yaw_states        = data_dict['yaw_states']   
            self.actions           = data_dict['actions']            
            self.gripper_actions   = data_dict['gripper_actions']   
            self.action_types      = data_dict['action_types']      
            self.rewards           = data_dict['rewards']            
            self.next_depth_states = data_dict['next_depth_states'] 
            self.dones             = data_dict['dones']              
            self.predict_qs        = data_dict['predict_qs']         
            self.labeled_qs        = data_dict['labeled_qs'] 
            self.priority          = data_dict['priority']  
            self.is_full           = data_dict['is_full']          
            self.memory_cntr       = data_dict['memory_cntr']   

    def update_buffer(self, sample_ind, predict_q):
        
        self.predict_q[sample_ind] = predict_q
        self.priority[sample_ind]  = np.abs(self.predict_q[sample_ind] - self.labeled_q[sample_ind] + self.sm_c)**self.alpha


