#!/usr/bin/env/ python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: kevin

Southeast University
"""


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip


#-----------------------------------------------------
import threshold
import getpic
import perpstransform
import findline
import display
import cal_cur_pos
import line
import calibration





#------------calibration test---------------------

cal_imgs = utils.get_images_by_dir('camera_cal')

object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))

test_imgs = utils.get_images_by_dir('test_images')

undistorted = []
for img in test_imgs:
    img = utils.cal_undistort(img,object_points,img_points)
    undistorted.append(img)






