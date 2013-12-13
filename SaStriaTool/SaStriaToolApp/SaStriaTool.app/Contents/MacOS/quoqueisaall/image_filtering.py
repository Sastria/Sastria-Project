import numpy as np
import sys, math, Image
import cv2,  os,  cv
import cv
import time

cv.NamedWindow("camera", 1)
capture = cv.CreateCameraCapture(0)

width = None #leave None for auto-detection
height = None #leave None for auto-detection

if width is None:
    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
else:
	cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,width)    

if height is None:
	height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
else:
	cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height) 

result = cv.CreateImage((width,height),cv.IPL_DEPTH_8U,3)

while True:
    img = cv.QueryFrame(capture)
    cv.Smooth(img,result,cv.CV_GAUSSIAN,9,9)
    cv.ShowImage("smooth", result)
    
    cv.Dilate(img,result,None,5) #uncommet to apply affect
    cv.ShowImage("dilate", result)
    
    cv.Erode(img,result,None,1) #uncommet to apply affect
    cv.ShowImage("erode", result)
#    
#    cv.MorphologyEx()
#    cv.ShowImage("MorphologyEx", result)
    
    threshold=10
    colour=255
    cv.Threshold(img,result, threshold,colour,cv.CV_THRESH_BINARY)
    cv.ShowImage("Threshold", result)
    
    dst_16s2 = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_16S, 1)
    cv.Laplace(img, dst_16s2,3)
    cv.Convert(dst_16s2,img)
    cv.ShowImage("Laplace", result)
    
#    image = cv2.imread('faces_tmp/2.pgm"')
#    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    ret,thresh = cv2.threshold(imgray,127,255,0)
#    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    k = cv.WaitKey(10);
    if k == 'f':
        break
