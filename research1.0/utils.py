import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

def wrap_ang(ang):

    if ang > np.pi:
        ang -= 2*np.pi
    elif ang < -np.pi:
        ang += 2*np.pi

    return ang

def euler2rotm(theta):

    R_x = np.array([[ 1,                     0,                     0 ],
                    [ 0, np.math.cos(theta[0]), -np.math.sin(theta[0])],
                    [ 0, np.math.sin(theta[0]),  np.math.cos(theta[0])]
                    ])
    R_y = np.array([[ np.math.cos(theta[1]), 0,  np.math.sin(theta[1])],
                    [                     0, 1,                      0],
                    [-np.math.sin(theta[1]), 0,  np.math.cos(theta[1])]
                    ])         
    R_z = np.array([[ np.math.cos(theta[2]), -np.math.sin(theta[2]), 0],
                    [ np.math.sin(theta[2]),  np.math.cos(theta[2]), 0],
                    [                     0,                      0, 1]
                    ])     
           
    R = np.dot(np.dot(R_x, R_y), R_z)

    return R

def quaternion_to_rotation_matrix(q):

    w, x, y, z = q

    # Ensure the quaternion is normalized

    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Compute the rotation matrix elements
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
    
    return R

def convert_depth_img_to_robot_frame(depth_img, K, cam_pos, cam_ori):

    im_h, im_w = depth_img.shape[0], depth_img.shape[1] 
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), 
                              np.linspace(0,im_h-1,im_h))

    x = np.multiply(pix_x-K[0][2], depth_img/K[0][0])
    y = np.multiply(pix_y-K[1][2], depth_img/K[1][1])
    z = copy.copy(depth_img)

    x.shape = (im_h*im_w,1)
    y.shape = (im_h*im_w,1)
    z.shape = (im_h*im_w,1)
    ones    = np.ones_like(z)
    xyz_pts = np.concatenate((x, y, z, ones), axis=1)

    # get rotation matrix relative to world frame
    cam_ori = [-cam_ori[0], -cam_ori[1], -cam_ori[2]]
    rotmat    = np.linalg.inv(euler2rotm(cam_ori))

    # construct transformation matrix (from camera frame to world frame)
    cam_TM          = np.eye(4)
    cam_TM[0:3,3]   = np.array(cam_pos)
    cam_TM[0:3,0:3] = copy.copy(rotmat)

    #trasnform to world frame (coninside with robot frame)
    T_xyz_pts = np.matmul(cam_TM, xyz_pts.T)
    T_z       = T_xyz_pts[2,:]
    T_z.shape = (im_h, im_w)

    return T_z