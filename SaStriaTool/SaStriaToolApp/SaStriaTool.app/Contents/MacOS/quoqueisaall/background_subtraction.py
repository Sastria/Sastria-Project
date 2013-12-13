import cv
capture=cv.CaptureFromCAM(0)
cv.WaitKey(200)
frame=cv.QueryFrame(capture)
temp=cv.CloneImage(frame)
cv.Smooth(temp,temp,cv.CV_BLUR,5,5)
i =0
while True:
#    i=i+1
#    print i
#    if i> 50:
#        i=0
#        frame=cv.QueryFrame(capture)
#        temp=cv.CloneImage(frame)
#        cv.Smooth(temp,temp,cv.CV_BLUR,5,5)
    frame2=cv.CloneImage(frame)
    frame=cv.QueryFrame(capture)
    cv.AbsDiff(frame,frame2,frame2)
    print frame2
    cv.ShowImage("Windo2w",temp)
    cv.ShowImage("Windo3",frame2)
    cv.ShowImage("Window",frame)
    c=cv.WaitKey(2)
    if c==27: #Break if user enters 'Esc'.
        break
