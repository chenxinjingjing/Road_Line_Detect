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





def get_M_Minv():
    src = np.float32([[(125, 707), (545, 345), (705, 343), (1177, 707)]])
    dst = np.float32([[(125, 707), (125, 100), (1177, 100), (1177, 707)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv




def main():


	img=cv2.imread('white01.jpg')

	M,Minv = get_M_Minv()

	thresholded_wraped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

	


	cv2.imshow('Threashold',thresholded_wraped)
	#print(threshholded)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

