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
GRIPPER_FULL_CLOSE         = 0
GRIPPER_FULL_OPEN          = 1
GRIPPER_NON_CLOSE_NON_OPEN = 2
GRIPPER_CANNOT_OPERATE     = 3
 
#initialise gripper action
OPEN_GRIPPER  = 0
CLOSE_GRIPPER = 1

#initialise action type
GRASP = 0
PUSH  = 1

#define the min. distance needs to be pushed
MIN_DIST_PUSH = 0.035

#initialise number of step for grasping/pushing
N_STEP_GRASP_DEMO = 7
N_STEP_PUSH_DEMO  = 7
N_STEP_GRASP = 10
N_STEP_PUSH  = 10

#initialise home pose 
HOME_POSE = [-0.11120, 
              0.48541, 
              0.26883, 
              0, 0, 0]

#initialise working space center
WORK_SPACE_CENTER = [-0.110, 0.560, 0.001]
WORK_SPACE_DIM    = [0.30, 0.25, 0.]

#max actions for the networks
MAX_ACTION      = [0.07, 0.07, 0.07, np.deg2rad(30.)]
PUSH_MAX_ACTION = [0.07, 0.07, 0.07, np.deg2rad(45.)]

#define push height
PUSH_HEIGHT  = 0.05
GRASP_HEIGHT = 0.05

#define the maximum possible distance
MAX_POSSIBLE_DIST = 0.
for i in range(2):
    DELTA = (WORK_SPACE_CENTER[i] + WORK_SPACE_DIM[i] - HOME_POSE[i])**2
    MAX_POSSIBLE_DIST += DELTA

MAX_POSSIBLE_DIST = MAX_POSSIBLE_DIST**0.5

print(f"[MAX_POSSIBLE_DIST]: {MAX_POSSIBLE_DIST}")

#define HLD mode

HLD_MODE       = 0
GRASP_ONLY     = 1
SEQ_GRASP_PUSH = 2