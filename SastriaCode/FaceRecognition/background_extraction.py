import cv2
import cv
import numpy as np
 
#c = cv2.VideoCapture(0)
c= cv2.VideoCapture("/home/felix/Videos/CC_piubella.mp4")
_,f = c.read()


avg1 = np.float32(f)
avg2 = np.float32(f)
avg3 = avg1
res3=0 
while(1):
    _,f = c.read()

    cv2.accumulateWeighted(f,avg1,0.1)
    cv2.accumulateWeighted(f,avg2,0.01)
    avg3= avg3+avg1
    cv2.accumulate(f,avg3)
    
#    cv2.accumulateWeighted(img,avg1,0.1)
#    cv2.accumulateWeighted(img,avg2,0.01)
     
    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)
    res3 = cv2.convertScaleAbs(avg3)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    cv2.imshow('avg3',res3)
    k = cv2.waitKey(20)
 
    if k == 27:
        break
 
cv2.destroyAllWindows()
c.release()
