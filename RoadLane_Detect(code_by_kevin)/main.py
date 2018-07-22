#!/usr/bin/env/ python
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






def thresholding(img):
#    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=55, thresh_max=100)
#    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(70, 255))
#    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
#    s_thresh = utils.hls_select(img,channel='s',thresh=(160, 255))
#    s_thresh_2 = utils.hls_select(img,channel='s',thresh=(200, 240))
#    
#    white_mask = utils.select_white(img)
#    yellow_mask = utils.select_yellow(img)

    x_thresh = threshold.abs_sobel_thresh(img, orient='x', thresh_min=25 ,thresh_max=255)
    mag_thresh = threshold.mag_thresh(img, sobel_kernel=3, mag_thresh=(40, 250))
    dir_thresh = threshold.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.7))
    hls_thresh = threshold.hls_select(img, thresh=(90, 255))
    hls_thresh_white = threshold.hls_select(img,channel='l', thresh=(180, 255))
    hls_thresh_yellow = threshold.hls_select(img,channel='h', thresh=(90, 120))
    lab_thresh = threshold.lab_select(img, thresh=(155, 200))
    luv_thresh = threshold.luv_select(img, thresh=(225, 255))
    #Thresholding combination
    #threshholded = np.zeros_like(x_thresh)
    #threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1


#-----------------mine test -------------------------
    threshholded = np.zeros_like(x_thresh)
    threshholded[((hls_thresh_white == 1) | (hls_thresh_yellow == 1)) & ((dir_thresh == 1)| (x_thresh == 1) | (mag_thresh == 1)) ]=1

    return threshholded




def processing(img,M,Minv,left_line,right_line):
    
    thresholded = thresholding(img)
    
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
   
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = findline.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
        
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = findline.find_line(thresholded_wraped)
	
    left_line.update(left_fit)
    right_line.update(right_fit)
    
    area_img = display.draw_area(img,thresholded_wraped,Minv,left_fit, right_fit)
    
    curvature,pos_from_center = cal_cur_pos.calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    
    result = display.draw_values(area_img,curvature,pos_from_center)
    
    return result




left_line = line.Line()
right_line = line.Line()

M,Minv = perpstransform.get_M_Minv()
#print('jjjjjjjjjjjjj')
project_outpath = 'video_out/yellow03.mp4'
project_video_clip = VideoFileClip("yellow03.mp4")
#print('jjjjjjjjjjjjj')
project_video_out_clip = project_video_clip.fl_image(lambda clip: processing(clip,M,Minv,left_line,right_line))
#print('jjjjjjjjjjjjj')
project_video_out_clip.write_videofile(project_outpath, audio=False)







