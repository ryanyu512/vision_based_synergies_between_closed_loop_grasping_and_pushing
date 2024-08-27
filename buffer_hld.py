import os
import numpy as np
import pickle 

class BufferReplay_HLD():

    def __init__(self, 
                 alpha = 0.6): 
      
        #current depth state
        self.depth_states      = None
        #current action type
        self.action_types      = None
        #next reward
        self.rewards           = None
        #next state
        self.next_depth_states = None
        #is done in the next state
        self.dones             = None
        #predicted q value: 
        self.predict_qs        = None
        #labeled q value: 
        self.labeled_qs        = None
        #predicted next q value: 
        self.predict_next_qs   = None
        #labeled next q value: 
        self.labeled_next_qs   = None
        #surprise value
        self.priority          = None
        #initialise power value for prioritisation
        self.alpha             = alpha
        #initialise small constant to prevent division by zero
        self.sm_c              = 1e-6
        #initialise memory size
        self.memory_size       = 0

        try:
            self.load_buffer()
            print("[SUCCESS] load previous buffer")
        except Exception as e:
            print("[FAIL] cannot load previous buffer")
            print(f"[FAIL] {str(e)}")

    def store_transition(self, 
                         depth_state, action_type, reward, 
                         next_depth_state, done, 
                         predict_q, labeled_q,
                         predict_next_q, labeled_next_q):

        if self.depth_states is None:        
            self.depth_states      = np.array([depth_state])
            self.action_types      = np.array([action_type])
            self.rewards           = np.array([reward])
            self.next_depth_states = np.array([next_depth_state])
            self.dones             = np.array([done])
            self.predict_qs        = np.array([predict_q])
            self.labeled_qs        = np.array([labeled_q])
            self.predict_next_qs   = np.array([predict_next_q])
            self.labeled_next_qs   = np.array([labeled_next_q])
            self.priority          = np.array([np.abs(predict_q - labeled_q + self.sm_c)**self.alpha])
        else:
            self.depth_states      = np.concatenate((self.depth_states, np.array([depth_state])))
            self.action_types      = np.concatenate((self.action_types, np.array([action_type])))
            self.rewards           = np.concatenate((self.rewards, np.array([reward])))
            self.next_depth_states = np.concatenate((self.next_depth_states, np.array([next_depth_state])))
            self.dones             = np.concatenate((self.dones, np.array([done])))
            self.predict_qs        = np.concatenate((self.predict_qs, np.array([predict_q])))
            self.labeled_qs        = np.concatenate((self.labeled_qs, np.array([labeled_q])))
            self.predict_next_qs   = np.concatenate((self.predict_next_qs, np.array([predict_next_q])))
            self.labeled_next_qs   = np.concatenate((self.labeled_next_qs, np.array([labeled_next_q])))
            self.priority          = np.concatenate((self.priority, np.array([np.abs(predict_q - labeled_q + self.sm_c)**self.alpha])))

        self.memory_size = self.depth_states.shape[0]

    def get_experience(self):

        return self.priority, self.depth_states, self.action_types, self.rewards, \
               self.next_depth_states, self.dones, self.predict_qs, self.labeled_qs, \
               self.predict_next_qs, self.labeled_next_qs

    def sample_buffer(self, batch_size):

        #          0,            1,            2,       3,                 4,     
        # priorities, depth_states, action_types, rewards, next_depth_states, 
        #     5,          6,          7,               8,               9
        # dones, predict_qs, labeled_qs, predict_next_qs, labeled_next_qs
        experience = self.get_experience()
        priorities = experience[0]

        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)
        probs = priorities/(priorities.sum())

        batch_size = min(batch_size, len(experience[0]))

        batch   = np.random.choice(len(experience[0]), 
                                   batch_size,
                                   replace = False, 
                                   p       = probs)

        batch_depth_states      = experience[1][batch]
        batch_action_types      = experience[2][batch]
        batch_rewards           = experience[3][batch]
        batch_next_depth_states = experience[4][batch]
        batch_dones             = experience[5][batch]
        batch_predict_qs        = experience[6][batch]
        batch_labeled_qs        = experience[7][batch]
        batch_predict_next_qs   = experience[8][batch]
        batch_labeled_next_qs   = experience[9][batch]      

        return batch, batch_depth_states, batch_action_types, batch_rewards, batch_next_depth_states, batch_dones, batch_predict_qs, batch_labeled_qs, batch_predict_next_qs, batch_labeled_next_qs

    def save_buffer(self):
        
        data_dict = dict()

        data_dict['depth_states']      = self.depth_states
        data_dict['action_types']      = self.action_types
        data_dict['rewards']           = self.rewards
        data_dict['next_depth_states'] = self.next_depth_states
        data_dict['dones']             = self.dones
        data_dict['predict_qs']        = self.predict_qs
        data_dict['labeled_qs']        = self.labeled_qs
        data_dict['predict_next_qs']   = self.predict_next_qs
        data_dict['labeled_next_qs']   = self.labeled_next_qs
        data_dict['priority']          = self.priority

        with open("hld_experience_data.pkl", 'wb') as file:
            pickle.dump(data_dict, file)

    def load_buffer(self):

        if not os.path.exists("hld_experience_data.pkl"):
            print("[FAIL] Buffer file not found.")
            return
        
        with open("hld_experience_data.pkl", 'rb') as file:
            data_dict = pickle.load(file)

            self.depth_states      = data_dict['depth_states']   
            self.action_types      = data_dict['action_types']      
            self.rewards           = data_dict['rewards']            
            self.next_depth_states = data_dict['next_depth_states'] 
            self.dones             = data_dict['dones']              
            self.predict_qs        = data_dict['predict_qs']         
            self.labeled_qs        = data_dict['labeled_qs'] 
            self.predict_next_qs   = data_dict['predict_next_qs']         
            self.labeled_next_qs   = data_dict['labeled_next_qs'] 
            self.priority          = data_dict['priority'] 

    def update_buffer(self, sample_ind, predict_q):
        
        self.predict_qs[sample_ind] = predict_q
        self.priority[sample_ind]  = np.abs(self.predict_qs[sample_ind] - self.labeled_qs[sample_ind] + self.sm_c)**self.alpha
