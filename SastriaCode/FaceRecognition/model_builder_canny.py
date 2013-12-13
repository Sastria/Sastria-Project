

import os
import sys
import cv2
import cv
import numpy as np

def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def read_images(path, sz=None):

    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    gray = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
#                    ## canny
                    detected_edges = cv2.GaussianBlur(gray,(1,1),5)
                    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
                    im = cv2.bitwise_and(img,img,mask = detected_edges)
                    
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))    
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    labels[c]=str(os.path.join(subject_path, filename))
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

def read_1_image(path, sz=None):
    X,y = [], []
    try:
                    gray = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    img = cv2.imread(os.path.join(subject_path, filename))
                    ## canny
                    detected_edges = cv2.GaussianBlur(gray,(1,1),5)
                    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
                    im = cv2.bitwise_and(img,img,mask = detected_edges)
                    
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    #cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))    
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    labels[c]=str(os.path.join(subject_path, filename))+"//"+str(c)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))    
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(777)
    except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
    except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    return [X,y]

if __name__ == "__main__":
    
    lowThreshold = 30
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    labels={}
    # This is where we write the images, if an output_dir is given
    out_dir = None
    image_dir= "/home/felix/Desktop/facerec-master/py/apps/videofacerec/att_faces"
#    if len(sys.argv) >= 2:
#           image_dir= sys.argv[1]
    [X,y] = read_images(image_dir)
#    if len(sys.argv) == 3:
#        out_dir = sys.argv[2]
    #model = cv2.createEigenFaceRecognizer()
    model = cv2.createLBPHFaceRecognizer()
    model.train(np.asarray(X), np.asarray(y))
    model.save("/home/felix/Desktop/facerec-master/py/apps/videofacerec/model-train-SIequaliz-SICANNY_GIACOMO_LBF_1img")
    print "build model"
