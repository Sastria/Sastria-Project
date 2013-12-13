


import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    print "gray"
    print gray
    print" --------------------------"
    detected_edges = cv2.GaussianBlur(gray,(1,1),5)
    print "detected_edges"
    print detected_edges
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  
    print dst
    cv2.imshow('canny demo',dst)

lowThreshold = 30
max_lowThreshold = 100
ratio = 3
kernel_size = 3

#img = cv2.imread('/home/felix/Pictures/uff.jpg')
img = cv2.imread('/home/felix/Desktop/myopencv/FrameMindPositive/att_faces/s8/5.pgm')
print "img ",  img
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(lowThreshold)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
