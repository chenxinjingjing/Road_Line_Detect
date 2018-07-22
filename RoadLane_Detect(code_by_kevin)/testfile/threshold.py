#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
"""
Created on 2018 05 23

@author: kevin
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip






# 全局变量



def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
	#print('ggggg')
	#print(abs_sobel)
	#print(np.max(abs_sobel))
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
	return binary_output


def magi_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output









def main():


	img=cv2.imread('yellow02.jpg')

	
	#print(img)
	
	x_thresh = 255*abs_sobel_thresh(img, orient='x', thresh_min=25 ,thresh_max=255)
	
	mag_thresh = 255*magi_thresh(img, sobel_kernel=3, mag_thresh=(40, 250))
	dir_thresh = 255*dir_threshold(img, sobel_kernel=3, thresh=(0.8, 1.7))
	hls_thresh_white = 255*hls_select(img,channel='l', thresh=(180, 255))
	hls_thresh_yellow = 255*hls_select(img,channel='h', thresh=(90, 120))
	lab_thresh = 255*lab_select(img, thresh=(100, 250))
	luv_thresh = 255*luv_select(img, thresh=(90, 150))
	
	#Thresholding combination
	#threshholded = np.zeros_like(hls_thresh)
	#threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1


	 

	#cv2.nameWindow('Threashold')


	'''
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	x=cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(x)
	y=cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	abs_sobely = np.absolute(y)
	dst = cv2.addWeighted(abs_sobelx,0,abs_sobely,1,0)
	print(dst)
	'''


	


	cv2.imshow('Threashold',hls_thresh_yellow)
	#print(threshholded)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

