import numpy as np

#TODO [NOTE 21 AUG 2024] temporarily hard code here

#initialise color space
COLOR_SPACE = np.asarray([[78.0, 121.0, 167.0],   # blue
                          [89.0, 161.0, 79.0],    # green
                          [156, 117, 95],         # brown
                          [242, 142, 43],         # orange
                          [237.0, 201.0, 72.0],   # yellow
                          [186, 176, 172],        # gray
                          [255.0, 87.0, 89.0],    # red
                          [176, 122, 161],        # purple
                          [118, 183, 178],        # cyan
                          [255, 157, 167]])/255.0 # pink

#initialise gripper status
GRIPPER_FULL_CLOSE = 0
GRIPPER_FULL_OPEN  = 1
GRIPPER_NON_CLOSE_NON_OPEN = 2

#initialise gripper action
OPEN_GRIPPER  = 0
CLOSE_GRIPPER = 1

#initialise action type
GRASP = 0
PUSH  = 1

#initialise home pose 
HOME_POSE = [-0.1112, 
             0.48541, 
             0.26883, 
             0, 0, 0]

#max actions for the networks
MAX_ACTION = [0.05, 0.05, 0.05, np.deg2rad(30.)]