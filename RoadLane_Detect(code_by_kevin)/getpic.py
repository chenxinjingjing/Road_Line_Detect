# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: kevin

Southeast University
"""


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def get_images_by_dir(dirname):
	img_names = os.listdir(dirname)
	img_paths = [dirname+'/'+img_name for img_name in imgnames]
	imgs = [cv2.imread(path) for path in img_paths]
        return imgs




