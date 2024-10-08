import os
import cv2
import sys
import time
import copy
import utils
import constants
import numpy as np
import matplotlib.pyplot as plt

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Env():
    def __init__(self, 
                 item_dir, 
                 N_item,           
                 workspace_lim, 
                 is_test = False, 
                 use_preset_test = False,
                 N_step_grasp             =  constants.N_STEP_GRASP_DEMO, 
                 N_step_push              =  constants.N_STEP_PUSH_DEMO,
                 can_execute_action_thres =  0.010,
                 push_reward_thres        =  0.010,
                 ground_collision_thres   =  0.00,
                 lift_z_after_grasp       =  0.05,
                 return_home_pos_thres    =  0.025,
                 return_home_ori_thres    =  np.deg2rad(2.5),
                 gripper_open_force       =  20.,
                 gripper_close_force      =  100.,
                 gripper_velocity         =  0.15,
                 gripper_joint_open       =  0.045,
                 gripper_joint_close      = -0.045,
                 cluttered_mode           = False, 
                 is_debug                 = True):

        #define workingspace limitation
        self.workspace_lim = workspace_lim

        #compute working space dimension        
        self.x_length = self.workspace_lim[0][1] - self.workspace_lim[0][0]
        self.y_length = self.workspace_lim[1][1] - self.workspace_lim[1][0]
        self.x_center = self.x_length/2. + self.workspace_lim[0][0]
        self.y_center = self.y_length/2. + self.workspace_lim[1][0]

        #define is_test
        self.is_test = is_test
        self.use_preset_test = use_preset_test

        #connect to simulator
        self.sim_client = RemoteAPIClient()
        self.sim = self.sim_client.require('sim')
        self.sim.startSimulation()

        #define the lift height after grasping successfully
        self.lift_z_after_grasp = lift_z_after_grasp

        #set reward threshold
        self.push_reward_thres        = push_reward_thres
        self.ground_collision_thres   = ground_collision_thres
        self.can_execute_action_thres = can_execute_action_thres

        #define gripper related velocity and force
        self.gripper_open_force  = gripper_open_force
        self.gripper_close_force = gripper_close_force
        self.gripper_velocity    = gripper_velocity
        self.gripper_joint_open  = gripper_joint_open
        self.gripper_joint_close = gripper_joint_close
        self.gripper_status      = None

        #define item directory
        self.item_dir = os.path.abspath(item_dir)

        #define items in the scene
        self.N_pickable_item = self.N_item = N_item

        #create 5-point at gripper tip for collision checking
        self.gripper_tip_box = np.array([[ -0.0175,        0,  -0.025], 
                                         [-0.00875,        0,  -0.025],
                                         [  0.0175,        0,  -0.025],
                                         [ 0.00875,        0,  -0.025],
                                         [       0,   0.0125,  -0.025],
                                         [       0,  0.00625,  -0.025],
                                         [       0,  -0.0125,  -0.025],
                                         [       0, -0.00625,  -0.025],
                                         [       0,        0,  -0.025]]).T
        
        #initialise N step for pushing and grasping
        self.N_step_grasp = N_step_grasp
        self.N_step_push  = N_step_push
        
        #set how items are generated
        self.cluttered_mode = cluttered_mode

        #setup rgbd cam
        self.setup_rgbd_cam()

        #initialise debug flag
        self.is_debug = is_debug

    def reset(self, reset_item = True):

        if reset_item:
            self.item_data_dict = dict()
            self.item_data_dict['color']    = []
            self.item_data_dict['path_ind'] = []
            self.item_data_dict['picked']   = []
            self.item_data_dict['handle']   = []
            self.item_data_dict['p_pose']   = []
            self.item_data_dict['c_pose']   = []
            self.item_data_dict['min_d']    = []
        else:
            self.item_data_dict['c_pose']  = self.update_item_pose()
            self.item_data_dict['p_pose']  = copy.copy(self.item_data_dict['c_pose'])

        #start setting
        self.start_env()

        #setup virtual RGB-D camera
        self.setup_rgbd_cam()
        try:
            if self.is_debug:
                print("[SUCCESS] setup rgbd camera")
        except:
            if self.is_debug:
                print("[FAIL] setup rgbd camera")

        if reset_item:
            self.N_pickable_item = self.N_item

        try:
            #get item paths
            self.item_paths  = os.listdir(self.item_dir)
            print(self.item_paths)
            if self.is_debug:
                print("[SUCCESS] load item paths")

            #assign item type
            if reset_item:
                self.item_data_dict['path_ind'] = np.random.choice(np.arange(len(self.item_paths)), self.N_item, replace = True).tolist()
                # self.item_data_dict['path_ind'] = np.array([1, 1])
                if self.is_debug:
                    print("[SUCCESS] randomly choose items")

                self.item_data_dict['color'] = constants.COLOR_SPACE[np.arange(self.N_item) % constants.COLOR_SPACE.shape[0]].tolist()
                if self.is_debug:
                    print("[SUCCESS] randomly choose item colors")
            
            #add items randomly to the simulation
            self.add_items(reset_item)
            if self.is_debug:
                print("[SUCCESS] add items to simulation")            
        except:
            if self.is_debug:
                print("[FAIL] add items to simulation")

        #reset to the default gripper open position
        self.open_close_gripper(is_open_gripper = True, target = self.gripper_joint_open)

    def start_env(self):

        # get UR5 goal handle
        self.UR5_goal_handle = self.sim.getObject('/UR5_goal')

        # get gripper handle
        self.gripper_tip_handle = self.sim.getObject('/UR5_tip')        

        # stop the simulation to ensure successful reset
        self.sim.stopSimulation()
        time.sleep(1)

        while True:

            self.sim.startSimulation()
            time.sleep(1)

            gripper_tip_pos = self.sim.getObjectPosition(self.gripper_tip_handle, 
                                                         self.sim.handle_world)
            
            UR5_goal_pos    = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                                         self.sim.handle_world)
            
            d_gripper_goal = np.linalg.norm(np.array(gripper_tip_pos) - np.array(UR5_goal_pos))

            if abs(gripper_tip_pos[2] - constants.HOME_POSE[2]) <= 1e-2 and d_gripper_goal <= 1e-2:
                print("[SUCCESS] restart environment")
                break
            else:
                #reset UR5 goal to home pose
                self.set_obj_pose(constants.HOME_POSE[0:3], constants.HOME_POSE[3:6], self.UR5_goal_handle, self.sim.handle_world)

    def setup_rgbd_cam(self):

        # get camera handle
        self.cam_handle = self.sim.getObject('/Vision_sensor')

        # get depth sensor related parameter
        self.near_clip_plane = self.sim.getObjectFloatParam(self.cam_handle, 
                                                            self.sim.visionfloatparam_near_clipping)
        
        self.far_clip_plane = self.sim.getObjectFloatParam(self.cam_handle, 
                                                           self.sim.visionfloatparam_far_clipping)
        
        # get camera pose
        cam_pos, cam_ori = self.get_obj_pose(self.cam_handle, self.sim.handle_world)

        # get rotation matrix relative to world frame
        rotmat    = utils.euler2rotm(cam_ori)

        # construct transformation matrix (from camera frame to world frame)
        self.cam_TM          = np.eye(4)
        self.cam_TM[0:3,3]   = np.array(cam_pos)
        self.cam_TM[0:3,0:3] = copy.copy(rotmat)
        self.K               = np.asarray([[618.62,      0, 320], 
                                           [     0, 618.62, 240], 
                                           [     0,      0,   1]])
        
        # print(f"[setup_rgbd_cam] \n {self.cam_TM}")

        # get RGB-D data
        self.depth_scale = 1.
        self.bg_color_img, self.bg_depth_img = self.get_rgbd_data()
        self.bg_depth_img = self.bg_depth_img * self.depth_scale

    def get_rgbd_data(self):

        #get color_img 
        raw_color_img_byte, self.color_resol = self.sim.getVisionSensorImg(self.cam_handle)
        raw_color_img = self.sim.unpackUInt8Table(raw_color_img_byte)

        color_img = np.array(raw_color_img)
        color_img.shape = (self.color_resol[1], self.color_resol[0], 3)
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(float)/255.
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = color_img.astype(np.uint8)

        #get depth_img
        raw_depth_img_byte, self.depth_resol = self.sim.getVisionSensorDepth(self.cam_handle)
        raw_depth_img = self.sim.unpackFloatTable(raw_depth_img_byte)
        
        depth_img = np.array(raw_depth_img)
        depth_img.shape = (self.depth_resol[1], self.depth_resol[0])
        depth_img = np.fliplr(depth_img)
        depth_img = depth_img*(self.far_clip_plane - self.near_clip_plane) + self.near_clip_plane

        return color_img, depth_img

    def is_within_working_space(self, pos, margin = 0.0):

        #check if the gipper tip is within working position + margin area
        min_x, max_x = self.workspace_lim[0]
        min_y, max_y = self.workspace_lim[1]
        is_within_x = min_x - margin <= pos[0] <= max_x + margin
        is_within_y = min_y - margin <= pos[1] <= max_y + margin

        return is_within_x, is_within_y

    def check_is_sim_stable(self):

        #get gripper_tip_pos        
        gripper_tip_pos = self.sim.getObjectPosition(self.gripper_tip_handle, 
                                                        self.sim.handle_world)

        is_within_x, is_within_y = self.is_within_working_space(gripper_tip_pos, margin = 0.0)

        if  not (is_within_x and is_within_y):
            print("[WARNING] simulation unstable")
            self.start_env()
            self.add_items()

    def reset_item2workingspace(self, max_reset_time = 2):

        for _ in range(max_reset_time):

            continue_to_check = True
            has_reset         = False

            while continue_to_check:

                has_reset = False
                for i, obj_handle in enumerate(self.item_data_dict['handle']):
                    
                    obj_pos = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)

                    #check if the item is within the working space
                    #want to leave some margin such that items on the edge of working area can still be picked up
                    is_within_x, is_within_y = self.is_within_working_space(obj_pos, margin = 0.0)

                    #TODO [NOTE ON 16/09/2024]: obj_pos[2] >= 0.03 is only suitable for restricted grasping and pushing action
                    #obj_pos[2] >= 0.03 => this condition avoids items stack on each other or upright position
                    if (not(is_within_x and is_within_y) or obj_pos[2] >= 0.035) and (not self.item_data_dict['picked'][i]) :
                        #randomise item pose
                        obj_pos, obj_ori = self.randomise_obj_pose(xc=self.x_center,
                                                                   yc=self.y_center,
                                                                   xL=self.x_length,
                                                                   yL=self.y_length)

                        #set item pose
                        self.set_obj_pose(obj_pos, obj_ori, obj_handle, self.sim.handle_world)

                        time.sleep(0.5)
                        has_reset = True
                        print(f"[SUCCESS] reset item {i} to working space")

                if not has_reset:
                    continue_to_check = False

        time.sleep(0.1)

        #update current pose
        self.item_data_dict['c_pose'] = self.update_item_pose()

    def randomise_obj_pose(self, xc, yc, xL, yL, margin = 0.0):

        #initialise item x, y
        obj_x = xc + np.random.uniform(-1, 1)*(xL/2. - margin) 
        obj_y = yc + np.random.uniform(-1, 1)*(yL/2. - margin) 
        obj_z = 0.025

        #define item droping position
        obj_pos = [obj_x, obj_y, obj_z]

        #define item droping orientation
        obj_ori = [2*np.pi*np.random.uniform(0, 1),
                   2*np.pi*np.random.uniform(0, 1),
                   2*np.pi*np.random.uniform(0, 1)]
        
        return obj_pos, obj_ori

    def add_items(self, reset_item = False, clustered_mode = False):

        # if self.cluttered_mode :
        #     x_center = self.x_center + np.random.uniform(-1, 1)*(self.x_length/2. - 0.05)
        #     y_center = self.y_center + np.random.uniform(-1, 1)*(self.x_length/2. - 0.05)

        #initialise item handles
        for i, path_ind in enumerate(self.item_data_dict['path_ind']):
            
            if self.is_test and self.use_preset_test:
                #TODO: add test cases
                pass
            else:
                
                #get item file path
                c_item_file = os.path.join(self.item_dir, self.item_paths[path_ind])
                
                if reset_item:
                    if self.cluttered_mode :
                        #randomise item pose
                        while True:
                            item_pos, item_ori = self.randomise_obj_pose(xc=self.x_center,
                                                                         yc=self.y_center,
                                                                         xL=0.035,
                                                                         yL=0.035)
                            time.sleep(0.5)
                            if self.is_within_working_space(item_pos, margin = 0.):
                                break
                    else:
                        #randomise item pose
                        item_pos, item_ori = self.randomise_obj_pose(xc=self.x_center,
                                                                     yc=self.y_center,
                                                                     xL=self.x_length,
                                                                     yL=self.y_length)
                else:
                    item_pos, item_ori = self.item_data_dict['c_pose'][i][0:3], self.item_data_dict['c_pose'][i][3:6]
            
            #define the shape name
            c_shape_name = f'shape_{i}'

            #define item color
            obj_color = self.item_data_dict['color'][i]

            print(f"item {i}: {c_shape_name}, path index: {path_ind}, pose: {item_pos + item_ori}")

            fun_out = self.sim.callScriptFunction('importShape@remoteApiCommandServer',
                                                  self.sim.scripttype_childscript,
                                                  [0,0,255,0],
                                                  item_pos + item_ori + obj_color,
                                                  [c_item_file, c_shape_name],
                                                  bytearray())
            
            c_shape_handle = fun_out[0][0]

            #update item data
            if not reset_item:
                self.item_data_dict['handle'][i]  = c_shape_handle
            else:
                self.item_data_dict['handle'].append(c_shape_handle)
                self.item_data_dict['picked'].append(False)

            if not (self.is_test and self.use_preset_test):
                time.sleep(0.5)                                      

        #check anything is outside of working space
        #if yes, reset pose
        self.reset_item2workingspace()

        #update items poses
        self.item_data_dict['c_pose'] = self.update_item_pose()
        self.item_data_dict['p_pose'] = copy.copy(self.item_data_dict['c_pose'])

        #compute min. distance between the item and its closest neighbour
        bbox_items, size_items, center_items, face_centers_items, Ro2w_items = self.compute_item_bboxes()

        for i in range(self.N_item):
            neighbour_pos, neighbour_ind, target_corner, neighbour_corner, min_distance = self.get_closest_item_neighbour(i, 
                                                                                                                          bbox_items,
                                                                                                                          face_centers_items)
            self.item_data_dict['min_d'].append(min_distance)

    def track_target(self, cur_pos, cur_ori, delta_pos_step, delta_ori_step):

        self.sim.setObjectPosition(self.UR5_goal_handle,
                                   (cur_pos[0] + delta_pos_step[0], 
                                    cur_pos[1] + delta_pos_step[1], 
                                    cur_pos[2] + delta_pos_step[2]),
                                    self.sim.handle_world)

        self.sim.setObjectOrientation(self.UR5_goal_handle, 
                                      (cur_ori[0] + delta_ori_step[0], 
                                       cur_ori[1] + delta_ori_step[1], 
                                       cur_ori[2] + delta_ori_step[2]), 
                                       self.sim.handle_world)

        fun_out = self.sim.callScriptFunction('ik@UR5',
                                              self.sim.scripttype_childscript,
                                              [0], 
                                              [0.], 
                                              ['0'], 
                                              bytearray())

        #get current goal position        
        cur_pos = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                             self.sim.handle_world)

        #get current goal orientation            
        cur_ori = self.sim.getObjectOrientation(self.UR5_goal_handle, 
                                                self.sim.handle_world)

        return cur_pos, cur_ori
            
    def compute_move_step(self, cur, next, N_steps):

        #get move vector
        move_vector = np.asarray([next[0] - cur[0], 
                                  next[1] - cur[1], 
                                  next[2] - cur[2]])
        
        #get move vector magnitude
        move_norm = np.linalg.norm(move_vector)

        if move_norm > 0.:
            #get unit direction
            unit_dir  = move_vector/move_norm
            #get move step
            delta_step = unit_dir*(move_norm/N_steps)
        else:
            delta_step = np.array([0., 0., 0.])

        return delta_step

    def move(self, delta_pos, delta_ori, N_steps = None):

        #get gripper tip pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #get current goal pose (ensure gripper tip and goal pose are the same)
        self.set_obj_pose(gripper_tip_pos, gripper_tip_ori, self.UR5_goal_handle, self.sim.handle_world)
        
        #get position
        UR5_cur_goal_pos, UR5_cur_goal_ori = self.get_obj_pose(self.UR5_goal_handle, self.sim.handle_world)

        #compute delta move step
        #N_steps cannot be too large, the motion will be very slow
        if N_steps is None:
            N_steps = 3 
        next_pos = np.array(UR5_cur_goal_pos) + np.array(delta_pos)
        next_ori = np.array(UR5_cur_goal_ori) + np.array(delta_ori)

        delta_pos_step = self.compute_move_step(UR5_cur_goal_pos, next_pos, N_steps)
        delta_ori_step = self.compute_move_step(UR5_cur_goal_ori, next_ori, N_steps)

        for step_iter in range(N_steps):
            UR5_cur_goal_pos, UR5_cur_goal_ori = self.track_target(UR5_cur_goal_pos, 
                                                                   UR5_cur_goal_ori,
                                                                   delta_pos_step, 
                                                                   delta_ori_step)
            
        UR5_cur_goal_pos, UR5_cur_goal_ori = self.track_target(next_pos, 
                                                               next_ori,
                                                               [0,0,0],
                                                               [0,0,0])
        
    def open_close_gripper(self, is_open_gripper, target):

        #get gripper open close joint handle        
        RG2_gripper_handle = self.sim.getObject('/openCloseJoint')

        #get joint position
        gripper_joint_position = self.sim.getJointPosition(RG2_gripper_handle)

        delta_position = target - gripper_joint_position

        if abs(delta_position) < 0.005:
            if is_open_gripper:
                self.gripper_status = constants.GRIPPER_FULL_OPEN
                print("[GRIPPER STATUS] FULL OPEN")
            else:
                self.gripper_status = constants.GRIPPER_FULL_CLOSE
                print("[GRIPPER STATUS] FULL CLOSE")

            return

        #set gripper force
        gripper_force = self.gripper_open_force if delta_position >= 0 else self.gripper_close_force

        self.sim.setJointTargetForce(RG2_gripper_handle, gripper_force)

        #set gripper velocity
        gripper_vel   = self.gripper_velocity if delta_position >= 0 else -self.gripper_velocity

        self.sim.setJointTargetVelocity(RG2_gripper_handle, gripper_vel)

        #initialise grasp counter
        grasp_cnt          = 0

        #initialise flags
        is_finish_action   = False
        is_grasp_something = False
        gripper_motion_cnt = 0

        self.gripper_cannot_operate = False
        while True:
            
            gripper_motion_cnt += 1
            new_gripper_joint_position = self.sim.getJointPosition(RG2_gripper_handle)
            
            if delta_position >= 0:
                if (new_gripper_joint_position >= target):
                    
                    #stop gripper movement
                    self.sim.setJointTargetVelocity(RG2_gripper_handle, 0.)
                    is_finish_action = True
            else:
                if new_gripper_joint_position <= target:
                    
                    #stop gripper movement
                    self.sim.setJointTargetVelocity(RG2_gripper_handle, 0.)
                    is_finish_action = True 

                elif new_gripper_joint_position >= gripper_joint_position:
                    grasp_cnt += 1

                    #add time.sleep(0.005) and lower the counter threshold to resolve long waiting problem
                    if grasp_cnt >= 2:
                        print("[GRASP] grasp something")
                        is_grasp_something = True
                        is_finish_action   = True 
                else:
                    grasp_cnt = 0

            if is_finish_action:
                if is_grasp_something and not is_open_gripper:
                    self.gripper_status = constants.GRIPPER_NON_CLOSE_NON_OPEN
                    print("[GRIPPER STATUS] NON_CLOSE_NON_OPEN")
                elif is_open_gripper:
                    self.gripper_status = constants.GRIPPER_FULL_OPEN
                    print("[GRIPPER STATUS] FULL OPEN")
                else: 
                    self.gripper_status = constants.GRIPPER_FULL_CLOSE
                    print("[GRIPPER STATUS] FULL ClOSE")

                return
            elif gripper_motion_cnt >= 100:
                self.gripper_status = constants.GRIPPER_CANNOT_OPERATE
                self.gripper_cannot_operate = True
                print("[WARNING] adnomral gripper motion")

                return

            gripper_motion_cnt += 1

            gripper_joint_position = new_gripper_joint_position
            time.sleep(0.005)

    def update_item_pose(self):

        poses = []
        for item_handle in self.item_data_dict['handle']:
            item_pos, item_ori = self.get_obj_pose(item_handle, self.sim.handle_world)
            poses.append(item_pos + item_ori)

        return poses

    def get_obj_pose(self, handle, ref_frame):

        pos = self.sim.getObjectPosition(handle, ref_frame)
        ori = self.sim.getObjectOrientation(handle, ref_frame)

        return pos, ori
    
    def set_obj_pose(self, pos, ori, handle, ref_frame):

        self.sim.setObjectPosition(handle, pos,ref_frame)
        self.sim.setObjectOrientation(handle, ori, ref_frame)
    
    def grasp_reward(self):

        #check if the item is grasped firmly
        is_success_grasp = False

        reward = 0.

        print(f"[GRASP DISTANCE] {reward}")

        if self.gripper_status == constants.GRIPPER_NON_CLOSE_NON_OPEN:
            #lift up the item for testing if the grasping is successful
            self.move(delta_pos = [0, 0, self.lift_z_after_grasp],
                      delta_ori = [0, 0, 0])

            #update current item position
            self.item_data_dict['c_pose'] = self.update_item_pose()

            for i, item_handle in enumerate(self.item_data_dict['handle']):
                
                #check if the item is picked already
                if self.item_data_dict['picked'][i]:
                    continue

                #compute change in z
                change_z = self.item_data_dict['c_pose'][i][2] - self.item_data_dict['p_pose'][i][2]

                #ensure the change in z is significant and is in upward direction
                if abs(change_z) >= self.lift_z_after_grasp*0.5 and change_z > 0:

                    #record the grasping is successful
                    is_success_grasp = True

                    #update pickable items counter
                    self.N_pickable_item -= 1

                    #reset item pose to out of working space
                    item_pos = [self.x_center - 1., self.y_center, 0.15]
                    item_ori = [                 0,             0,    0]
                    self.set_obj_pose(item_pos, item_ori, item_handle, self.sim.handle_world)
                    
                    #label the item as "picked"
                    self.item_data_dict['picked'][i] = True

                    print("[SUCESS] picked an item!")
                    break
            
            if is_success_grasp:
                reward += 1.0
            else:
                reward += 0.0

        print(f"[GRASP REWARD] R: {reward}")

        return reward, is_success_grasp

    def push_reward(self, gripper_pos):

        #[NOTE 25 Aug 2024]: push reward should encourage the robot push items that are cluttered together

        reward = 0.

        #update current item position after action
        self.item_data_dict['c_pose'] = self.update_item_pose()

        push_clustered_item     = False
        push_non_clustered_item = False
        for i, obj_handle in enumerate(self.item_data_dict['handle']):

            d_gripper2item =  np.linalg.norm(np.array(self.item_data_dict['c_pose'][i][0:3]) - np.array(gripper_pos))

            #only interested in the items closed to gripper tip
            if d_gripper2item > 0.075 or self.item_data_dict['picked'][i]:
                continue 
            
            #compute any change in position
            delta_distance = np.linalg.norm(np.array(self.item_data_dict['c_pose'][i][0:3]) - np.array(self.item_data_dict['p_pose'][i][0:3]))

            #ensure the item has been pushed significantly and the item is close to another item
            if delta_distance > self.push_reward_thres:
                if self.is_debug:
                    print(f'delta_distance {i}: {delta_distance}')
                if self.item_data_dict['min_d'][i] <= constants.MIN_DIST_PUSH:
                    push_clustered_item = True
                else:
                    push_non_clustered_item = True

        if push_clustered_item:
            reward += 1.
        elif push_non_clustered_item:
            reward += 0.1
        
        print(f"[PUSH REWARD] R: {reward}")

        return reward, push_clustered_item

    def check_is_action_executable(self, gripper_tip_pos, gripper_tip_ori, UR5_cur_goal_pos):

        d = np.linalg.norm(np.array(UR5_cur_goal_pos) - np.array(gripper_tip_pos))

        #check if the action is executable
        self.can_execute_action = True if d <= self.can_execute_action_thres else False

        #ensure goal pose and gripper pose are the same
        self.set_obj_pose(gripper_tip_pos, gripper_tip_ori, self.UR5_goal_handle, self.sim.handle_world)

    def check_is_collision_to_ground(self, gripper_tip_pos):

        #check if the tipper is in collision with ground
        self.is_collision_to_ground = (gripper_tip_pos[2] <= self.ground_collision_thres)
    
    def check_is_out_of_workingspace(self, gripper_tip_pos, margin = 0.1):
        
        if (gripper_tip_pos[0] < self.workspace_lim[0][0] - margin) or \
           (gripper_tip_pos[0] > self.workspace_lim[0][1] + margin):
            self.is_out_of_working_space = True
        elif (gripper_tip_pos[1] < self.workspace_lim[1][0] - margin) or \
             (gripper_tip_pos[1] > self.workspace_lim[1][1] + margin):
            self.is_out_of_working_space = True
        elif (gripper_tip_pos[2] < self.workspace_lim[2][0]) or \
             (gripper_tip_pos[2] > self.workspace_lim[2][1]):
            self.is_out_of_working_space = True
        else:
            self.is_out_of_working_space = False

    def check_is_within_sight(self, target_item_pos):

        # Get the aspect ratio
        ratio = self.depth_resol[1] / self.depth_resol[0]  # width/height

        res, fov = self.sim.getObjectFloatParameter(self.cam_handle, self.sim.visionfloatparam_perspective_angle)

        # Calculate the horizontal FOV from the vertical FOV and aspect ratio
        if ratio > 1:
            fov_x = fov
            fov_y = 2 * np.arctan(np.tan(fov_x / 2) * ratio)
        elif ratio < 1:
            fov_y = fov
            fov_x = 2 * np.arctan(np.tan(fov_y / 2) * ratio)
        else:
            fov_x = fov_y = fov

        # print(f"fov_x: {fov_x}, fov_y: {fov_y}")

        #get camera pose
        cam_pos, cam_ori = self.get_obj_pose(self.cam_handle, self.sim.handle_world)
        cam_yaw = cam_ori[2]
        cam_H   = cam_pos[2]

        #compute sensing area

        Sx = 2*abs(cam_H)*np.tan(fov_x/2.)
        Sy = 2*abs(cam_H)*np.tan(fov_y/2.)

        Wx, Wy = Sx/2., Sy/2.

        #check if the item is within the sensing area             
        offset_pos = np.array([target_item_pos[0] - cam_pos[0],
                               target_item_pos[1] - cam_pos[1]])
        
        local_x = offset_pos[0]*np.cos(-cam_yaw) - offset_pos[1]*np.sin(-cam_yaw)
        local_y = offset_pos[0]*np.sin(-cam_yaw) + offset_pos[1]*np.cos(-cam_yaw)

        print(f"Wx: {Wx :.4f}, Wy: {Wy :.4f}, cam_yaw: {np.rad2deg(cam_yaw) :.4f}, cam_H: {cam_H :.4f}")

        if np.abs(local_x) <= Wx and np.abs(local_y) <= Wy:
            print(f"IS_WITHIN_SIGHT")
            self.is_within_sight = True
        else:
            print(f"IS_OUT_OF_SIGHT")
            self.is_within_sight = False

    def reward(self, action_type, gripper_tip_pos, gripper_tip_ori):

        #compute how close between gripper tip and the nearest item
        is_success_grasp = False
        is_success_push  = False

        #get UR5 goal 
        UR5_cur_goal_pos, _ = self.get_obj_pose(self.UR5_goal_handle, self.sim.handle_world)    

        #compute reward
        if action_type == constants.GRASP:
            reward, is_success_grasp = self.grasp_reward()
        elif action_type == constants.PUSH:
            reward, is_success_push  = self.push_reward(gripper_tip_pos)

        #check if the action is executable
        self.check_is_action_executable(gripper_tip_pos, gripper_tip_ori, UR5_cur_goal_pos)

        #check if the tipper is in collision with ground
        self.check_is_collision_to_ground(gripper_tip_pos)

        #check if the tipper is out of working space
        self.check_is_out_of_workingspace(gripper_tip_pos)

        if self.is_out_of_working_space or self.is_collision_to_ground or self.gripper_cannot_operate or not self.can_execute_action:
            reward -= 0.5

        print(f"[OVERALL REWARD] {reward}")

        return reward, is_success_grasp, is_success_push

    def reward_hld(self, action_type, rewards_low_level, delta_moves_grasp, delta_moves_push):
        
        if action_type == constants.GRASP:
            if np.max(rewards_low_level) > 0:
                hld_reward =  1.0 
            else:
                hld_reward = -1.0 
        else:
            if len(delta_moves_grasp) > 0 and len(delta_moves_push) > 0:
                hld_reward =  0.0 #for neutral push decision
            elif len(delta_moves_grasp) > 0 and len(delta_moves_push) == 0:
                hld_reward = -1.0 #for incorrect push decision                               
            elif len(delta_moves_grasp) == 0 and len(delta_moves_push) > 0:
                hld_reward =  0.5 #for correct push decision

        if self.env.N_pickable_item <= 0:
            hld_reward += 5.0

        return hld_reward

    def get_raw_data(self, action_type):

        _, depth_img         = self.get_rgbd_data()
        _, gripper_tip_ori   = self.get_obj_pose(self.gripper_tip_handle, 
                                                 self.sim.handle_world)
        if action_type == constants.PUSH:
            yaw_ang        = gripper_tip_ori[2]
            gripper_status = None
        else:
            yaw_ang        = gripper_tip_ori[2]
            # gripper_status = self.env.gripper_status
            gripper_status = None

        return depth_img, gripper_status, yaw_ang

    def step(self, action_type, delta_pos, delta_ori, is_open_gripper):

        is_success_grasp = False

        delta_ori = [0, 0, delta_ori]

        self.gripper_cannot_operate = False

        # if action_type == constants.GRASP:
            # self.move(delta_pos = delta_pos, delta_ori = delta_ori)
            # target = self.gripper_joint_open if is_open_gripper else self.gripper_joint_close
            # self.open_close_gripper(is_open_gripper, target)
        # elif action_type == constants.PUSH:
            # self.move(delta_pos = delta_pos, delta_ori = delta_ori)
            # target = self.gripper_joint_close
            # self.open_close_gripper(is_open_gripper, target)

        #get gripper_tip_pos        
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        if action_type == constants.GRASP:
            target = self.gripper_joint_open
            is_open_gripper = True
            self.open_close_gripper(is_open_gripper, target)
        elif action_type == constants.PUSH:
            if gripper_tip_pos[2] >= 0.1:
                is_open_gripper = True
                target = self.gripper_joint_open
            else:
                is_open_gripper = False
                target = self.gripper_joint_close
            self.open_close_gripper(is_open_gripper, target)


        self.move(delta_pos = delta_pos, delta_ori = delta_ori)

        #get gripper_tip_pos        
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        is_push = False
        if action_type ==constants.GRASP:
            #TODO [NOTE ON 15/09/2024]: change it to tunable parameters later if workable
            if gripper_tip_pos[2] <= 0.05: 
                target = self.gripper_joint_close
                is_open_gripper = False
            else:
                target = self.gripper_joint_open
                is_open_gripper = True

            self.open_close_gripper(is_open_gripper, target)
        elif action_type == constants.PUSH:

            if gripper_tip_pos[2] >= 0.1:
                is_open_gripper = True
                target = self.gripper_joint_open
            else:
                is_open_gripper = False
                target = self.gripper_joint_close
            self.open_close_gripper(is_open_gripper, target)

            print(f"gripper tip height: {gripper_tip_pos[2]}")

            if gripper_tip_pos[2] <= 0.05:
                yaw = gripper_tip_ori[2]
                
                delta_push = [np.cos(yaw + np.pi/2.)*constants.PUSH_MAX_ACTION[0], 
                              np.sin(yaw + np.pi/2.)*constants.PUSH_MAX_ACTION[1], 
                              0.]
                
                self.move(delta_pos = delta_push, delta_ori = [0., 0., 0.])
                is_push = True

        #get gripper_tip_pos        
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #get next raw data
        next_depth_img, next_gripper_state, next_yaw_state = self.get_raw_data(action_type)

        #compute reward
        reward, is_success_grasp, is_success_push = self.reward(action_type, gripper_tip_pos, gripper_tip_ori)

        #update item poses
        self.item_data_dict['c_pose'] = self.update_item_pose()
        self.item_data_dict['p_pose'] = copy.copy(self.item_data_dict['c_pose'])

        #compute min. distance between the item and its closest neighbour
        bbox_items, size_items, center_items, face_centers_items, Ro2w_items = self.compute_item_bboxes()

        for i in range(self.N_item):
            _, _, _, _, min_distance = self.get_closest_item_neighbour(i, 
                                                                       bbox_items,
                                                                       face_centers_items)
            self.item_data_dict['min_d'][i] = min_distance

        return reward, is_success_grasp, is_push, next_depth_img, next_gripper_state, next_yaw_state, gripper_tip_pos[2]
    
    def return_home(self, is_env_reset, action_type = None):

        #get gripper tip pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #align the UR5 goal tip's pose to gripper tip pose
        self.set_obj_pose(gripper_tip_pos, gripper_tip_ori, self.UR5_goal_handle, self.sim.handle_world)

        #compute delta position and orientation
        delta_pos = np.array(constants.HOME_POSE[0:3]) - np.array(gripper_tip_pos)
        delta_ori = np.array(constants.HOME_POSE[3:6]) - np.array(gripper_tip_ori)

        is_lift = False

        return_home_counter = 0
        while True:


            if not is_env_reset and action_type == constants.PUSH and not is_lift:

                #lift up to prevent collision after pushing
                self.move(delta_pos = [0., 0., constants.MAX_ACTION[2]], delta_ori = [0., 0., 0.])

                #update gripper tip pose
                gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

                #update delta position and orientation
                delta_pos = np.array(constants.HOME_POSE[0:3]) - np.array(gripper_tip_pos)
                delta_ori = np.array(constants.HOME_POSE[3:6]) - np.array(gripper_tip_ori)

                is_lift = True

            self.move(delta_pos = delta_pos, delta_ori = delta_ori, N_steps = 10)

            #always open gripper
            self.open_close_gripper(is_open_gripper = True, target = self.gripper_joint_open)

            #get gripper tip pose
            gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

            #align the UR5 goal tip's pose to gripper tip pose
            self.set_obj_pose(gripper_tip_pos, gripper_tip_ori, self.UR5_goal_handle, self.sim.handle_world)

            #compute delta position and orientation
            delta_pos = np.array(constants.HOME_POSE[0:3]) - np.array(gripper_tip_pos)
            delta_ori = np.array(constants.HOME_POSE[3:6]) - np.array(gripper_tip_ori)

            #check if the robot arm return to home position successfully
            if np.linalg.norm(delta_pos) <= 0.01 and np.abs(delta_ori[2]) <= np.deg2rad(1.0):
                print("[SUCCESS] return home")
                break
            else:
                return_home_counter += 1
                print("[FAIL] return home")

            if return_home_counter >= 10:
                print("[FAIL] cannot return home")
                self.reset(reset_item = False)
                break

    def compute_item_bboxes(self):

        bbox_items          = []
        size_items          = []
        center_items        = []
        face_centers_items  = []
        Ro2w_items          = []

        for i, handle in enumerate(self.item_data_dict['handle']):

            path_ind = self.item_data_dict['path_ind'][i]

            #get min/max xyz based on body frame
            r, max_x = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_max_x)
            r, min_x = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_min_x)
            r, max_y = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_max_y)
            r, min_y = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_min_y)
            r, max_z = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_max_z)
            r, min_z = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_min_z)

            #get size
            size_items.append([max_x - min_x, max_y - min_y, max_z - min_z])

            #get item pose
            pose = self.sim.getObjectPose(handle, self.sim.handle_world)

            #get item center
            cxyz = pose[0:3]
            center_items.append(cxyz)

            #get item rotation matrix (from body frame to world frame)
            xyzw = pose[3:7]
            wxyz = [xyzw[-1], xyzw[0], xyzw[1], xyzw[2]]
            R_b2w  = utils.quaternion_to_rotation_matrix(wxyz)

            #get item rotation matrix (from object frame to world frame)
            # if path_ind != 2:
            #     R_o2b = np.array([[0, 0, 1],
            #                       [0, 1, 0],
            #                       [1, 0, 0]])
            # else:
            #     R_o2b = np.eye(3)

            R_o2b = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]])

            R = np.matmul(R_b2w, R_o2b)
            Ro2w_items.append(R)

            #construct 8 points bounding box 
            #x => z, y => y z => x
            p_obj = np.array([[max_x, max_y, min_z], #1a,3a,5a,0
                              [max_x, max_y, max_z], #2a,3b,5b,1
                              [max_x, min_y, min_z], #1b,3d,6a,2
                              [max_x, min_y, max_z], #2b,3c,6b,3
                              [min_x, max_y, min_z], #1d,4a,5d,4
                              [min_x, max_y, max_z], #2d,4b,5c,5
                              [min_x, min_y, min_z], #1c,4d,6d,6
                              [min_x, min_y, max_z]])#2c,4c,6c,7

            #transform bounding box to world frame
            p = np.zeros((8,3))
            for j in range(p_obj.shape[0]):
                p[j,:] = np.matmul(R, p_obj[j,:]) + np.array(cxyz)

            bbox_items.append(p)

            #get bounding box faces
            face_pts_items = [[p[0,:], p[2,:], p[6,:], p[4,:]],
                              [p[1,:], p[3,:], p[7,:], p[5,:]],
                              [p[0,:], p[1,:], p[3,:], p[2,:]],
                              [p[4,:], p[5,:], p[7,:], p[6,:]],
                              [p[0,:], p[1,:], p[5,:], p[4,:]],
                              [p[2,:], p[3,:], p[7,:], p[6,:]]]
            
            #compute face centers
            centers = []
            for face in face_pts_items:
                center = np.zeros((3,))
                for pt in face:
                    center += pt
                center /= 4.
                centers.append(center)

            face_centers_items.append(centers)

        return bbox_items, size_items, center_items, face_centers_items, Ro2w_items
    
    def sort_item_from_nearest_to_farest(self):

        #get item poses
        items_pose = copy.copy(self.item_data_dict['c_pose'])

        #get gripper pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, 
                                                             self.sim.handle_world)

        #compute distance
        distances  = []
        delta_vecs = []

        for i in range(len(items_pose)):
            delta_vector = np.array(items_pose[i][0:3]) - np.array(gripper_tip_pos)

            delta_vecs.append(delta_vector)
            distances.append(np.linalg.norm(delta_vector))

        items_pose      = np.array(items_pose)
        distances       = np.array(distances)
        delta_vecs      = np.array(delta_vecs)

        sorted_item_ind   = np.argsort(distances)
        sorted_items_pose = items_pose[sorted_item_ind,:]
        sorted_distances  = distances[sorted_item_ind]
        sorted_delta_vecs = delta_vecs[sorted_item_ind,:]

        return sorted_item_ind, sorted_items_pose[:,0:3], gripper_tip_pos, sorted_delta_vecs

    def get_closest_item_neighbour(self, 
                                   target_ind, 
                                   bbox_items,
                                   face_centers_items):

        #no neighbour item anymore or the target item is picked already
        if self.N_pickable_item <= 1 or self.item_data_dict['picked'][target_ind]:
            return None, None, None, None, np.inf

        #get all item poses
        item_poses = copy.copy(self.item_data_dict['c_pose'])

        #get closest neighbour item based on center distance
        target_pose              = item_poses[target_ind]
        min_center_distance      = np.inf
        neighbour_ind = None

        for i, item_pose in enumerate(item_poses):

            if i == target_ind or self.item_data_dict['picked'][i]:
                continue

            distance = np.linalg.norm(np.array(item_pose[0:3]) - np.array(target_pose[0:3]))
            if min_center_distance > distance:
                min_center_distance = distance
                neighbour_ind = i

        #get the target bounding box points + face centers
        target_pts    = np.vstack((bbox_items[target_ind], face_centers_items[target_ind]))

        #get the neighbour bounding box points + face centers
        neighbour_pts = np.vstack((bbox_items[neighbour_ind], face_centers_items[neighbour_ind]))

        #compute cloest points between the target and its closest neighbour
        min_distance      = np.inf
        neighbour_corner  = None
        target_corner     = None

        for target_pt in target_pts:
            for neighbour_pt in neighbour_pts:

                    distance = np.linalg.norm(target_pt - neighbour_pt)

                    if min_distance > distance:
                        min_distance     = distance
                        target_corner    = copy.copy(target_pt)
                        neighbour_corner = copy.copy(neighbour_pt)

        return item_poses[neighbour_ind][0:3], neighbour_ind, target_corner, neighbour_corner, min_distance

    def compute_push_path(self, 
                          center_items,
                          size_items, 
                          Ro2w_items,
                          target_ind, 
                          neighbour_ind, 
                          target_pt, 
                          neighbour_pt, 
                          offset_z = 0.):

        #get all item poses
        item_poses = copy.copy(self.item_data_dict['c_pose'])

        #get gripper pose
        gripper_pos, gripper_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #get target box size, target orientation (from item frame to world frame) and target box center (relative to world frame)
        target_bbox_size    = size_items[target_ind]
        target_R            = Ro2w_items[target_ind]
        target_center       = center_items[target_ind]

        #get neighbour box size, orientation (from item frame to world frame) and neighbour box center (relative to world frame)
        neighbour_bbox_size = size_items[neighbour_ind]
        neighbour_R         = Ro2w_items[neighbour_ind]
        neighbour_center    = center_items[neighbour_ind]

        target_distance     = np.linalg.norm(np.array(gripper_pos) - np.array(target_center))
        neighbour_distance  = np.linalg.norm(np.array(gripper_pos) - np.array(neighbour_center))

        bbox_sizes          = np.vstack((target_bbox_size, neighbour_bbox_size))
        # print(f"[bbox_sizes] {bbox_sizes.shape}")
        centers             = np.vstack((target_center, neighbour_center))
        # print(f"[centers] {centers.shape}")
        min_distance_index  = np.argmin([target_distance, neighbour_distance])
        max_distance_index  = np.argmax([target_distance, neighbour_distance])
        
        #compute the pushing point that can help singulate items 
        # random_choice = np.random.choice(2, 1, p = [0.9, 0.1])[0]

        # if random_choice == 0:
        #     push_point = centers[min_distance_index] 
        # else:
        #     push_point = centers[max_distance_index] 

        push_point = centers[min_distance_index]

        #offset the z coordinate to avoid gripper tip in collision with the items 
        #some corners on on the ground => need to offset for safe
        push_point    = np.array([push_point[0], 
                                  push_point[1], 
                                  push_point[2] + offset_z]) 

        #compute unit vector from item a (being pushed) to neighbour item b
        # if random_choice == 0:
        #     unit_vec_a2b = centers[max_distance_index] - centers[min_distance_index]
        # else:
        #     unit_vec_a2b = centers[min_distance_index] - centers[max_distance_index]

        unit_vec_a2b = centers[max_distance_index] - centers[min_distance_index]

        unit_vec_a2b[2] = 0.
        unit_vec_a2b /= np.linalg.norm(unit_vec_a2b)

        #compute a list of orientation candidate
        ang_list = np.linspace(0, 2*np.pi, 36, endpoint = True)
        be4_push_point = None

        delta_cnt = 0
        while True:

            #initialise start point list of push path
            push_start_pt_list          = []
            #initialise the collision counter
            collision_free_counter_list = []
            shift_distance_list         = []

            #compute offset from push point 
            # if random_choice == 0:
            #     max_length = np.max(bbox_sizes[min_distance_index])
            #     # max_length = np.max(target_bbox_size) if target_distance <= neighbour_distance else np.max(neighbour_bbox_size)
            # else:
            #     max_length = np.max(bbox_sizes[max_distance_index])
            #     # max_length = np.max(target_bbox_size) if target_distance  > neighbour_distance else np.max(neighbour_bbox_size)

            max_length = np.max(bbox_sizes[min_distance_index])

            print(f"[compute_push_path]: {max_length}")
            if max_length > 0.05:
                offset = 0.065
            else:
                offset = 0.040

            for i, ang, in enumerate(ang_list): 

                #comput the rotation matrix for various angles
                R = np.array([[np.cos(ang), -np.sin(ang), 0],
                              [np.sin(ang),  np.cos(ang), 0],
                              [          0,            0, 1]])
                
                #compute the just be4 pushing candidate
                p  = np.matmul(R, np.array([offset+delta_cnt*0.005, 0, 0])) + push_point

                #deep copy to ensure correct storage
                push_start_pt_list.append(copy.copy(p))
                
                #create 5 - point check for collision check
                gripper_tip_box = copy.copy(self.gripper_tip_box)

                gripper_tip_box = np.matmul(R, gripper_tip_box).T + p

                #ensure the starting point of pushing is not in collision with other items
                is_collision_at_start_pt = False

                for j in range(len(center_items)):
                    is_collision_at_start_pt = self.is_within_bbox(gripper_tip_box, center_items[j], size_items[j], Ro2w_items[j], margin = 0.005)
                    if is_collision_at_start_pt:
                        break

                if not is_collision_at_start_pt:
                    
                    #generate trajectory points for collision checking
                    push_delta       = push_point - p
                    push_delta_norm  = np.linalg.norm(push_delta)
                    push_unit_vector = push_delta/push_delta_norm
                    cos_sim          = np.dot(push_unit_vector, unit_vec_a2b)

                    push_N_step      = np.ceil(push_delta_norm/0.005).astype(np.int32) + 1
                    push_step_mag    = np.linspace(0, push_delta_norm, push_N_step, endpoint = True)
                    push_step_mag    = push_step_mag[1:] - push_step_mag[0:-1]

                    collision_free_counter = 0
                    for step_mag in push_step_mag:
                        
                        #update the gripper box coordinates for collision checking
                        for j in range(gripper_tip_box.shape[0]):
                            gripper_tip_box[j] += push_unit_vector*step_mag

                        if (self.is_within_bbox(gripper_tip_box,    target_center,    target_bbox_size,    target_R) or 
                            self.is_within_bbox(gripper_tip_box, neighbour_center, neighbour_bbox_size, neighbour_R)):
                                break
                        
                        collision_free_counter += 1
                    
                    #compute the distance that really move the item
                    impact_distance = np.min(push_delta_norm - collision_free_counter*push_step_mag[0], 0)

                    if cos_sim > 0:
                        distance               = -np.inf
                        collision_free_counter = -1
                    else:
                        shifted_center  = centers[min_distance_index] + push_unit_vector*impact_distance
                        distance        = np.linalg.norm(shifted_center - neighbour_center)   

                    # elif random_choice == 0:
                    #     shifted_center  = centers[min_distance_index] + push_unit_vector*impact_distance
                    #     distance        = np.linalg.norm(shifted_center - neighbour_center)
                    # else:
                    #     shifted_center  = centers[max_distance_index] + push_unit_vector*impact_distance
                    #     distance        = np.linalg.norm(shifted_center - target_center)

                    #it means that this pushing cannot touch any item => useless push
                    #reset the cnt to -1
                    if collision_free_counter == len(push_step_mag) - 1:
                        collision_free_counter = -1
                else:
                    collision_free_counter = -1  
                    distance = -np.inf    
                
                #store how many trajectory points are not in collision with items 
                collision_free_counter_list.append(collision_free_counter)
                shift_distance_list.append(distance)

            #get the path with max points
            sorted_item_ind            = np.argsort(shift_distance_list)[::-1]
            sorted_shift_distance_list = np.array(shift_distance_list)[sorted_item_ind]
            sorted_push_start_pt_list  = np.array(push_start_pt_list)[sorted_item_ind]
            sorted_push_path_collision_free_counter = np.array(collision_free_counter_list)[sorted_item_ind]

            is_path_found = False
            for i, shift_dist in enumerate(sorted_shift_distance_list):

                if sorted_push_path_collision_free_counter[i] >= 2:
                    be4_push_point = sorted_push_start_pt_list[i]
                    is_path_found  = True
                    break

            if not is_path_found:
                if self.is_debug:
                    print("[FAIL] continue to search push path")
                delta_cnt += 1
            else:
                if self.is_debug:
                    print("[SUCCESS] search push path")
                break

        return push_point, be4_push_point

    def is_within_bbox(self, points, center, size, Ro2w, margin = 0.0):
        
        
        Rw2o = np.linalg.inv(Ro2w)
        ps_  = copy.copy(points)
        for p in ps_:
            p -= center
            p  = np.matmul(Rw2o, p.T)
            if (0 <= abs(p[0]) <= size[0]/2. + margin and 
                0 <= abs(p[1]) <= size[1]/2. + margin and
                0 <= abs(p[2]) <= size[2]/2. + margin):
                return True

        return False

    def compute_guidance_grasp_ang(self, item_ind, bbox_items):

        #extract bounding box corresponding to item index
        p         = bbox_items[item_ind]
        path_ind  = self.item_data_dict['path_ind'][item_ind]

        #initialise vectors, norms and unit vectors
        vecs      = np.zeros((3,3))
        vecs_norm = np.zeros(3)
        unit_vecs = np.zeros((3,3))

        vecs[0,:] = p[0,:] - p[1,:]
        vecs[1,:] = p[0,:] - p[2,:]
        vecs[2,:] = p[0,:] - p[4,:]

        for i in range(vecs.shape[0]):
            vecs_norm[i] = np.linalg.norm(vecs[i,:])
            unit_vecs[i,:] = vecs[i,:]/vecs_norm[i]

        #compute yaw angle orientation
        axis_list      = []
        axis_norm_list = []
        for i in range(unit_vecs.shape[0]):

            z_component = abs(unit_vecs[i,2])
            
            #store all axis list
            if z_component < 0.85:
                axis_list.append(unit_vecs[i,:])
                axis_norm_list.append(vecs_norm[i])

        target_yaw_angle_list = []
        delta_yaw_list        = []

        for i in range(len(axis_list)):
            #compute two angle options 
            #the gripper yaw angle is the x-axis of gripper frame relative to world frame
            #we want the y-axis of gripper align with the orientation of the longest dimension
            #this become problematic if the dimension is symmetric in x-y plane
            ang1 = utils.wrap_ang(np.arctan2(axis_list[i][1], axis_list[i][0]) - np.deg2rad(90.))
            ang2 = ang1 + np.pi if np.sign(ang1) <= 0 else ang1 - np.pi

            #compute delta orientation
            _, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)
            delta_ori1 = ang1 - gripper_tip_ori[2]
            delta_ori2 = ang2 - gripper_tip_ori[2]

            #choose the minimum delta yaw 
            if abs(delta_ori1) <= abs(delta_ori2):
                target_yaw_angle = ang1
                delta_yaw        = delta_ori1
            else:
                target_yaw_angle = ang2
                delta_yaw        = delta_ori2

            target_yaw_angle_list.append(target_yaw_angle)
            delta_yaw_list.append(delta_yaw)

        is_all_within_gripper_width = True
        for i in range(len(axis_norm_list)):
            if self.is_debug:
                print(f"axis_norm_list[i]: {axis_norm_list[i]}/{self.gripper_joint_open*2.}")
            if axis_norm_list[i] >= self.gripper_joint_open*2.:
                is_all_within_gripper_width = False
        
        if is_all_within_gripper_width:
            chosen_index = np.argmin(np.abs(delta_yaw_list))
        else:
            chosen_index = np.argmax(axis_norm_list)

        target_yaw_angle  = target_yaw_angle_list[chosen_index]
        delta_yaw         = delta_yaw_list[chosen_index]

        # if path_ind != 7: #non-cylinder
        #     target_yaw_angle  = target_yaw_angle_list[chosen_index]
        #     delta_yaw         = delta_yaw_list[chosen_index]
        # else: #cylinder
        #     if abs(np.min(axis_norm_list) - np.max(axis_norm_list[1])) <= 0.001:
        #         if self.is_debug:
        #             print("[GRASP_ANG] UPRIGHT CYLINDER")
        #         target_yaw_angle  = gripper_tip_ori[2]
        #         delta_yaw         = 0.
        #     else:
        #         if self.is_debug:
        #             print("[GRASP_ANG] LYING CYLINDER")
        #         target_yaw_angle  = target_yaw_angle_list[chosen_index]
        #         delta_yaw         = delta_yaw_list[chosen_index]
        current_yaw_angle = gripper_tip_ori[2]

        return target_yaw_angle, current_yaw_angle, delta_yaw
    
    def demo_guidance_generation(self,
                                 grasp_max_move = constants.MAX_ACTION[0], 
                                 grasp_max_ori  = constants.MAX_ACTION[3],
                                 push_max_move  = constants.PUSH_MAX_ACTION[0],
                                 push_max_ori   = constants.PUSH_MAX_ACTION[3], 
                                 min_distance_threshold = constants.MIN_DIST_PUSH,
                                 offset_z      = 0.050,
                                 offset_z_push = 0.050):

        #initialise move delta
        delta_move_grasp = []
        delta_move_push  = []

        is_get_push_target  = False
        is_get_grasp_target = False
        target_item_pos_grasp = None
        target_item_pos_push  = None

        #step1: check if all items are picked
        if self.N_pickable_item == 0:
            return delta_move_grasp, delta_move_push, target_item_pos_grasp, target_item_pos_push
        
        #update all item pose
        self.item_data_dict['c_pose'] = self.update_item_pose()

        #step2: get the cloest to the farest item relative to gripper tip
        sorted_item_ind, sorted_items_pos, gripper_tip_pos, sorted_delta_vecs = self.sort_item_from_nearest_to_farest()

        #step 3: [MOVE] move in a straight line to the top of the item and adjust the yaw orientation

        #get items bounding boxes and related properties 
        bbox_items, size_items, center_items, face_centers_items, Ro2w_items = self.compute_item_bboxes()

        #initialise target position
        for i in range(len(sorted_item_ind)):
            
            #check if this item is picked or not
            if self.item_data_dict['picked'][sorted_item_ind[i]]:
                continue
            
            #check if the item is within the working space
            is_within_x, is_within_y = self.is_within_working_space(sorted_items_pos[i], margin = 0.0)

            #do not pick anything out of working space
            if not (is_within_x and is_within_y):
                continue

            #compute min. distance if # of pickable items >= 2
            if self.N_pickable_item >= 2:
                neighbour_pos, neighbour_ind, target_corner, neighbour_corner, min_distance = self.get_closest_item_neighbour(sorted_item_ind[i], 
                                                                                                                              bbox_items,
                                                                                                                              face_centers_items)
            else:
                min_distance = np.inf

            print(f"[MIN_DISTANCE] {min_distance}")
            #check if the target item is graspable
            if not is_get_grasp_target and min_distance > min_distance_threshold:

                is_get_grasp_target = True

                item_ind        = sorted_item_ind[i]
                target_item_pos_grasp = sorted_items_pos[i]

                #compute linear movement N_step
                #offset the delta to ensure the gripper tip is not in collision with the ground/target item
                delta_lin       = sorted_delta_vecs[i] + np.array([0,0,offset_z]) 
                delta_lin_norm  = np.linalg.norm(delta_lin)
                lin_unit_vector = delta_lin/delta_lin_norm

                #compute angular movement N_step
                _, _, delta_ori = self.compute_guidance_grasp_ang(item_ind, bbox_items)

                #get a unified N_step
                N_step          = self.N_step_grasp

                #compute magnitude of each delta move
                step_mag_lin = np.linspace(0, delta_lin_norm, N_step, endpoint = True)
                step_mag_lin = step_mag_lin[1:] - step_mag_lin[0:-1]

                step_mag_ori = np.linspace(0, delta_ori, N_step, endpoint = True)
                step_mag_ori = step_mag_ori[1:] - step_mag_ori[0:-1]

                for j in range(len(step_mag_lin)):
                    delta_move_grasp.append([lin_unit_vector[0]*step_mag_lin[j], 
                                             lin_unit_vector[1]*step_mag_lin[j],
                                             lin_unit_vector[2]*step_mag_lin[j],
                                             step_mag_ori[j],
                                             1, 0]) #open gripper

                if step_mag_lin[0] > constants.MAX_ACTION[0]:
                    print("[ERROR] step_mag_lin[0] > constants.MAX_ACTION[0]")
                elif step_mag_ori[0] > constants.MAX_ACTION[-1]:
                    print("[ERROR] step_mag_ori[0] >constants.MAX_ACTION[-1]")

                print(f"[GRASP GUIDANCE] home2grasp- linear: {step_mag_lin[0]}, angular {np.rad2deg(step_mag_ori[0])}")

                #step4: [GRASP] open gripper and move vertically down by a constant height => close gripper
                # delta_move.append([0, 0, -offset_z*0.9, 0, 0, 1]) #close gripper
                delta_move_grasp.append([0, 0, -offset_z*0.7, 0., 0, 1]) #close gripper

                # return delta_move, target_item_pos, constants.GRASP

            if not is_get_push_target and min_distance <= min_distance_threshold:

                is_get_push_target = True

                #compute the closest point between two items and push start point
                target_ind      = sorted_item_ind[i]
                target_item_pos_push = sorted_items_pos[i]

                push_point, be4_push_point = self.compute_push_path(center_items,
                                                                    size_items, 
                                                                    Ro2w_items,
                                                                    target_ind, 
                                                                    neighbour_ind, 
                                                                    target_corner, 
                                                                    neighbour_corner)

                #move from the home position to the starting point of pushing
                push_home2start          = be4_push_point + np.array([0, 0, offset_z_push]) - np.array(gripper_tip_pos)
                push_home2start_norm     = np.linalg.norm(push_home2start)
                push_home2start_unit_vec = push_home2start/push_home2start_norm

                #move from just starting point of pushing to closest point between two items
                push_start2closest          = push_point - be4_push_point
                push_start2closest_norm     = np.linalg.norm(push_start2closest)
                push_start2closest_unit_vec = push_start2closest/push_start2closest_norm

                #compute target yaw angle 
                target_yaw_ang = utils.wrap_ang(np.arctan2(push_start2closest_unit_vec[1], push_start2closest_unit_vec[0]) - np.deg2rad(90.))

                #compute delta move from home position to the starting point of pushing

                #[compute angular movement]
                _, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)
                delta_yaw = target_yaw_ang - gripper_tip_ori[2]

                #unify N_step
                N_step = self.N_step_push

                #compute step magnitude of each step
                step_mag_lin_home2start    = np.linspace(0, push_home2start_norm, N_step, endpoint = True)
                step_mag_lin_home2start    = step_mag_lin_home2start[1:] - step_mag_lin_home2start[0:-1]
        
                step_mag_ori_home2start    = np.linspace(0, delta_yaw, N_step, endpoint = True)
                step_mag_ori_home2start    = step_mag_ori_home2start[1:] - step_mag_ori_home2start[0:-1]

                #move to the point on top of the push start point
                for j in range(len(step_mag_lin_home2start)):
                    delta_move_push.append([push_home2start_unit_vec[0]*step_mag_lin_home2start[j], 
                                            push_home2start_unit_vec[1]*step_mag_lin_home2start[j],
                                            push_home2start_unit_vec[2]*step_mag_lin_home2start[j],
                                            step_mag_ori_home2start[j],
                                            0, 1])
                
                #move downward to reach the push start point
                delta_move_push.append([0,0,-offset_z_push*0.5, 0, 0, 1]) #+1
                # delta_move_push.append([0,0,-offset_z_push*1.05, 0, 0, 1]) #+1

                # #push item
                # delta_move.append([push_start2closest_unit_vec[0]*push_max_move, 
                #                    push_start2closest_unit_vec[1]*push_max_move,
                #                    push_start2closest_unit_vec[2]*push_max_move,
                #                    0,
                #                    0, 1]) #close gripper #+1
                
                if step_mag_lin_home2start[0] > constants.PUSH_MAX_ACTION[0]:
                    print("[ERROR] step_mag_lin_home2start[0] > constants.PUSH_MAX_ACTION[0]")
                elif step_mag_ori_home2start[0] > constants.PUSH_MAX_ACTION[-1]:
                    print("[ERROR] step_mag_ori_home2start[0] > constants.PUSH_MAX_ACTION[-1]")

                print(f"[PUSH GUIDANCE] home2start- linear: {step_mag_lin_home2start[0]}, angular: {np.rad2deg(step_mag_ori_home2start[0])}")

                # return delta_move, target_item_pos, constants.PUSH

        return delta_move_grasp, delta_move_push, target_item_pos_grasp, target_item_pos_push
