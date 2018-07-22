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




def get_M_Minv():
    src = np.float32([[(125, 707), (545, 345), (705, 343), (1177, 707)]])
    dst = np.float32([[(125, 707), (125, 100), (1177, 100), (1177, 707)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
