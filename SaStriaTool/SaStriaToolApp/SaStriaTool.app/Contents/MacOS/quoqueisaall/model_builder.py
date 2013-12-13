

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
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
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
                    #cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))    
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

def read_1_image(path, sz=None):
    X,y = [], []
    try:
                    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    #cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))    
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(777)
    except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
    except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    return [X,y]

if __name__ == "__main__":
    labels={}
    # This is where we write the images, if an output_dir is given
    out_dir = None
    image_dir= "att_faces"
    if len(sys.argv) >= 2:
           image_dir= sys.argv[1]
    [X,y] = read_images(image_dir)
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    #model = cv2.createEigenFaceRecognizer()
    model = cv2.createLBPHFaceRecognizer()
    model.train(np.asarray(X), np.asarray(y))
    model.save("model-train-equaliz-_ln_pb_em_fc_gp_lbph")
    print "build model"
