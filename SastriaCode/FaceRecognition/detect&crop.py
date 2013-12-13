
import sys, math
import getopt

import cv
import cv2
import time
import Image
import os
import numpy as np

import isa_resource as isar

def read_images(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    labels[c]=str(os.path.join(subject_path, filename))+"//"+str(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

def DetectFace(image, faceCascade):
 
    min_size = (20,20)
    image_scale = 2
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    find=0

    # Allocate the temporary images
    grayscale = cv.CreateImage((image.width, image.height), 8, 1)
    smallImage = cv.CreateImage(
            (
                cv.Round(image.width / image_scale),
                cv.Round(image.height / image_scale)
            ), 8 ,1)
 
    # Convert color input image to grayscale
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
 
    # Scale input image for faster processing
    cv.Resize(grayscale, smallImage, cv.CV_INTER_LINEAR)
 
    # Equalize the histogram
    cv.EqualizeHist(smallImage, smallImage)
 
    # Detect the faces
    faces = cv.HaarDetectObjects(
            smallImage, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )
    #print "faces ",  faces
    # If faces are found
    if faces:
        find = 1
        for ((x, y, w, h), n) in faces:
            X_np,y_np = [], []
            try :
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                sub_face = image[ pt1[1]:pt2[1],  pt1[0] : pt2[0]]                
                face_file_name = "faces_tmp/face_a.pgm"
                cv.SaveImage(face_file_name, sub_face)
                im = cv2.imread(face_file_name, cv2.IMREAD_GRAYSCALE)     
                im = cv2.resize(im,  (92, 112))
                cv.SaveImage("att_faces/face_a.pgm", im)
                
            except IOError, (errno, strerror):
                            print "I/O error({0}): {1}".format(errno, strerror)
            except:
                            print "Unexpected error:", sys.exc_info()[0]                                   
        print    labels[p_label2], " confidence ",p_confidence2                
                

    return image,  find

def CropFromFile(video_path,  (x, y, w, h),  face_file_name,  pos_msec=None,  sz=(92, 112)):
    capture = cv.CaptureFromFile(video_path)
    if pos_msec :
        cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_POS_MSEC,  pos_msec)
    image = cv.QueryFrame(capture)    
    #cp = CropFromVideo(image,  (x, y, w, h),  face_file_name):
        
def CropFromVideo(image,  (x, y, w, h),  face_file_name,  sz=(92, 112)):

    min_size = (20,20)
    image_scale = 1

    # Allocate the temporary images
    grayscale = cv.CreateImage((image.width, image.height), 8, 1)
    smallImage = cv.CreateImage(
            (
                cv.Round(image.width / image_scale),
                cv.Round(image.height / image_scale)
            ), 8 ,1)
 
    # Convert color input image to grayscale
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
 
    # Scale input image for faster processing
    cv.Resize(grayscale, smallImage, cv.CV_INTER_LINEAR)
 
    # Equalize the histogram
    cv.EqualizeHist(smallImage, smallImage)
 



    X_np,y_np = [], []
    try :
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                sub_face = image[ pt1[1]:pt2[1],  pt1[0] : pt2[0]]                
                cv.SaveImage(face_file_name, sub_face)
#                im = cv2.imread(face_file_name, cv2.IMREAD_GRAYSCALE)     
#                im = cv2.resize(im,  (92, 112))
#                cv.SaveImage(face_file_name, im)
                
    except IOError, (errno, strerror):
                            print "I/O error({0}): {1}".format(errno, strerror)
#    except:
#                            print "Unexpected error:", sys.exc_info()[0]                                                   
    

def CropFromImage(image,  (x, y, w, h), face_file_name,  sz=(92, 112)):
 
    min_size = (20,20)
    print " face_file_name ", face_file_name
#    image_scale = 2
#    haar_scale = 1.1
#    min_neighbors = 3
#    haar_flags = 0
#    find=0
#
#    # Allocate the temporary images
#    grayscale = cv.CreateImage((image.width, image.height), 8, 1)
#    smallImage = cv.CreateImage(
#            (
#                cv.Round(image.width / image_scale),
#                cv.Round(image.height / image_scale)
#            ), 8 ,1)
 
    # Convert color input image to grayscale
#    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
# 
#    # Scale input image for faster processing
#    cv.Resize(grayscale, smallImage, cv.CV_INTER_LINEAR)
 
    # Equalize the histogram
#    cv.EqualizeHist(smallImage, smallImage)
 
    # Detect the faces
#    faces = cv.HaarDetectObjects(
#            smallImage, faceCascade, cv.CreateMemStorage(0),
#            haar_scale, min_neighbors, haar_flags, min_size
#        )
    #print "faces ",  faces
    # If faces are found
#    if faces:
#        find = 1
#        for ((x, y, w, h), n) in faces:
#            X_np,y_np = [], []
    try :
                pt1 = (int(x ), int(y))
                pt2 = ( int(x + w) , int(y + h) )
                sub_face = image[ pt1[1]:pt2[1],  pt1[0] : pt2[0] ]                
#                face_file_name = "faces_tmp/face_a.pgm"
                #print  pt1, pt2
                #cv.SaveImage(face_file_name, sub_face)
                cv2.imwrite(face_file_name, sub_face)
                
                print "saved"
#                im = cv2.imread(face_file_name, cv2.IMREAD_GRAYSCALE)     
#                im = cv2.resize(im,  sz)
#                cv.SaveImage(face_file_name, im)
                
    except IOError, (errno, strerror):
                            print "I/O error({0}): {1}".format(errno, strerror)
#    except:
#                            print "Unexpected error:", sys.exc_info()[0]                                   
                

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan(float(eye_direction[1])/float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image

def SizeSuggest(rect=(0, 0, 92, 112)):
    w=float(92)
    h=float(112)
    a =rect[2]
    b=rect[3]
#    if a>w and b>h and float(a)/b > w/h:
#        incr_h =(h/w)*float(a) - float(b)

    if float(rect[2])/rect[3] ==w/h:
        return rect
    else:
        incr_w=(w/h)*float(b) - float(a)
        incr_h =(h/w)*float(a) - float(b)
        if incr_w > incr_h :
            return (rect[0], rect[1],  rect[2]+incr_w, rect[3])
        else:
            return (rect[0], rect[1],  rect[2], rect[3]+incr_h)
    return rect   



def test1():
        print  "test1"
        video_path = "/home/felix/Videos/test1.mpeg"
        capture = cv.CaptureFromFile(video_path)
        index =0
        try:
            video = isar.Video()
            print " tracce del video ",  len(video._track_list )
            bb_list =isar.get_bb("/home/felix/Desktop/facerec-master/py/apps/videofacerec/test1.xml" )
            print len(bb_list)
            for b in bb_list :
                        cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_POS_FRAMES, b._frame)
                        image = cv.QueryFrame(capture)   
                        cr= CropFromVideo(image, b._position,"/home/felix/Pictures/att_test/" +str(index) + ".jpg" )
                        index = index +1
            print "finish"           
        except getopt.error, msg:
            print msg
            print " usage: file_conf.xml"    
            
    
def test2():
        print  "test2"
        video_path = "/home/felix/Videos/test1.mpeg"
        capture = cv.CaptureFromFile(video_path)

        try:
            video = isar.Video()
            print " tracce del video ",  len(video._track_list )
            bb_list =isar.get_bb("/home/felix/Desktop/facerec-master/py/apps/videofacerec/test1.xml" )
            print len(bb_list)
            video.preprocessing(bb_list)
            print " tracce del video ",  len(video._track_list )
            value_list=[]
            for i in range (len(video._track_list ) ) :
                index =0
                if not os.path.exists("/home/felix/Pictures/att_test/"+str(i+1)):
                        os.makedirs("/home/felix/Pictures/att_test/"+str(i+1))
                        os.makedirs("/home/felix/Pictures/att_test/"+str(i+1) +"/s1"  )

                tr=video._track_list [i]
                tr_bb_list =tr._bbList
                for b in tr_bb_list :
                        cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_POS_FRAMES, b._frame)
                        image = cv.QueryFrame(capture)   
                        cr= CropFromVideo(image, b._position,"/home/felix/Pictures/att_test/"+str(i+1)+"/s1/" +str(index) + ".pgm" )
                        index = index +1
                        if index>9 :
                            break
            print "finish"           
        except getopt.error, msg:
            print msg
            print " usage: file_conf.xml"    
                    

        while (cv.WaitKey(15)==-1):     
            print  "---"    


            break
        
if __name__ == "__main__":
    test2()
#    
#        video_path = "/home/felix/Videos/test2.mpeg"
#        capture = cv.CaptureFromFile(video_path)
#        cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_POS_FRAMES, 50)
#        while (cv.WaitKey(15)==-1):     
#            print  "---"    
#            image = cv.QueryFrame(capture)   
#            print image
#            cv.ShowImage("face recognition test", image)
#            cr= CropFromVideo(image, (200, 200,100, 100 ),"/home/felix/Pictures/att_test/da_video.jpg" )
#
#            break
#    
#    
#    
#        print "cropping..."
#        image = cv2.imread(os.path.join("/home/felix/Pictures/Angelina3.jpg"), cv2.IMREAD_GRAYSCALE)
#        print str(type(image) )
#        if not str(type(image) ).find("numpy.ndarray") ==-1: 
#            print " meglio convertire"
#            #image =cv.fromarray(image) 
#        crop_image = CropFromImage(image, (370, 490,740, 740 ),"/home/felix/Pictures/att_test/Angelina3.jpg" )
#        #cv.SaveImage("/home/felix/Pictures/att_test/Angelina3.jpg", crop_image)
#        print "end"
#if __name__ == "__main__":
#        base ="/home/felix/Desktop/OpenCV-2.4.3/data/"
#         
#        fc_list=[]
#        fc_list.append(cv.Load(base+"haarcascades/haarcascade_frontalface_default.xml"))
#        fc_list.append(cv.Load(base+"haarcascades/haarcascade_frontalface_alt2.xml"))
#        fc_list.append(cv.Load(base+"haarcascades/haarcascade_frontalface_alt.xml"))
#        fc_list.append(cv.Load(base+"haarcascades/haarcascade_frontalface_alt_tree.xml"))
#        print fc_list
#        image = cv2.imread(os.path.join("/home/felix/Pictures/Angelina.jpg"), cv2.IMREAD_GRAYSCALE)
#        #image =  cv.imread("/home/felix/Pictures/Angelina.jpg")   
#        print image
##        xc=805
##        yc=143
##        num=8
##
##        image11= image.crop((xc,yc,xc+59,yc+118))
##        image11 = image11.resize((92,112), Image.ANTIALIAS)
##        image11.save("att_test/sa/"+str(num)+".pgm")
##
##        labels={}
##        image_dir= "att_faces"
##        read_images(image_dir)
#        for fc in fc_list:
#                find = DetectFace(image, fc)
#                if find:
#                    break
#         
