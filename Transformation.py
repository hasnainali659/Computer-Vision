# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:53:27 2020

@author: DELL
"""

################### SCALING ##########################

import cv2
import numpy as np

image = cv2.imread('1.jpg')

image = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

################### TRANSLATION ######################

image = cv2.imread('1.jpg')
image = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
rows,columns,RGB = image.shape

point = np.float32([[1.5,0.5,101],[0.5,1.2,51]])
dist = cv2.warpAffine(image,point,(columns,rows))

cv2.imshow('image',dist)
cv2.waitKey(0)
cv2.destroyAllWindows()

################### ROTATION #########################

image = cv2.imread('1.jpg')
image = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
rows,columns,RGB = image.shape

point = cv2.getRotationMatrix2D((columns/1.8,rows/1.8),91,1.01)
dist = cv2.warpAffine(image,point,(columns,rows))

cv2.imshow('image',dist)
cv2.waitKey(0)
cv2.destroyAllWindows()

################### AFFINE TRANSFORMATION #############
import numpy as np

image = cv2.imread('1.jpg')
image = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
rows,columns,RGB = image.shape

point1 = np.float32([[51,51],[201,51],[51,201]])
point2 = np.float32([[11,101],[201,51],[101,251]])

point = cv2.getAffineTransform(point1,point2)

dist = cv2.warpAffine(image,point,(columns,rows))
cv2.imshow('image',dist)
cv2.waitKey(0)
cv2.destroyAllWindows()

################### PERSPECTIVE TRANSFORMATION ##########

image = cv2.imread('1.jpg')
rows,columns,RGB = image.shape

pts1 = np.float32([[57,66],[367,53],[27,388],[390,391]])
pts2 = np.float32([[0,0],[301,0],[0,301],[301,301]])

point = cv2.getPerspectiveTransform(pts1,pts2)

dist = cv2.warpPerspective(image,point,(600,1280))

cv2.imshow('image',dist)
cv2.waitKey(0)
cv2.destroyAllWindows()


