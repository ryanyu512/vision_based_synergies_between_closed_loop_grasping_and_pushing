# vision_based_synergies_between_closed_loop_grasping_and_pushing

**Abstract** 
---------------
<p align="justify">
Even though recent robotics research has made great progress, coordination between closed-loop operations remains a relatively underexplored topic. In this work, a depth vision sensor is mounted on the robot arm to enable closed-loop operations. A hierarchical structure is proposed: The high-level decision (HLD) network determines high-level decisions (either pushing or grasping) and low-level action (LLA) networks (grasping and pushing network) execute the selected decisions. Reinforcement learning updates the HLD network. Both LLA networks learn from behaviour cloning and reinforcement learning. The demonstration program is designed to provide correct examples for behaviour cloning based on oriented bounding boxes provided by the CoppeliaSim simulator. 5 squared cubes are randomly placed and close to each other on the 0.3m × 0.25m working space to impose challenges for grasping. In evaluation, the proposed system achieved a Completion Rate (CR) of 100%, an Average Grasping Success (AGS) rate per completion is 93.1% and an average number of Decisions Taken for Completion (DTC) is 7.8. Three baselines are introduced for comprehensive comparisons: The grasping-only baseline performs graspings only, while the sequential grasping-pushing baseline follows a fixed grasp-push sequence. The demonstration system made high-level decisions and performed actions based on current situations. The evaluation showed that the grasping-only baseline's AGS (86.6%) is the lowest, demonstrating the necessity of pushing in a cluttered environment. The proposed system's DTC (7.8) is lower than the sequential grasping-pushing baseline (10.6) and the designed demonstration system (11.4), indicating the importance of the content-aware decision and the effectiveness of the designed architecture and training approach.	
</p>

**Demo**
=============

The demo has been sped up 4 times to reduce its size and fit within the repository's storage limit.

| Proposed System | Grasping-only Baseline | 
|-----------------|------------------------|
| ![hld_mode](https://github.com/ryanyu512/vision_based_synergies_between_closed_loop_grasping_and_pushing/blob/main/research2.0/hld_mode.gif)  |  ![grasp_only](https://github.com/ryanyu512/vision_based_synergies_between_closed_loop_grasping_and_pushing/blob/main/research2.0/grasp_only.gif) |

| Sequential Grasp-Push Baseline | Demo Baseline | 
|-----------------|------------------------|
| ![seq_grasp_push](https://github.com/ryanyu512/vision_based_synergies_between_closed_loop_grasping_and_pushing/blob/main/research2.0/seq_grasp_push.gif) |  ![demo_mode](https://github.com/ryanyu512/vision_based_synergies_between_closed_loop_grasping_and_pushing/blob/main/research2.0/demo_mode.gif) |

**Code Overview**
---------------

1. research2.0_train.ipynb: Used for training the proposed system
2. research2.0_eval.ipynb: Used for evaluating the proposed system and baselines
3. env.py: define the functions of the environment, such as raw data extraction and reward
4. agent.py: define the functions of agent, such as interaction with the environment and networks' update
5. network.py: define the proposed networks
6. buffer.py: define the buffer for demonstration and self-exploration experience
7. buffer_hld.py: define the buffer for the high-level system
8. utils.py: define some frequently used functions, such as angle range limitation
9. constants.py: define the constants for the proposed system and simulation setup
</p>
