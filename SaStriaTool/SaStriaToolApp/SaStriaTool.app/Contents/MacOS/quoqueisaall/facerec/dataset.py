import os as os
import numpy as np
import PIL.Image as Image
import random
import csv
import cv2
import sys

class DataSet(object):
    def __init__(self, filename=None, sz=None):
        self.labels = []
        self.groups = []
        self.names = {}
        self.data = []
        self.sz = sz
        if filename is not None:
            self.load(filename)

    def shuffle(self):
        idx = np.argsort([random.random() for i in xrange(len(self.labels))])
        self.data = [self.data[i] for i in idx]
        self.labels = self.labels[idx]
        if len(self.groups) == len(self.labels):
            self.groups = self.groups[idx]

    def load(self, path):
        c = 0
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        #im =cv2.imread(os.path.join(subject_path, filename))
#                        ##canny
#                        gray = cv2.imread(os.path.join(subject_path, filename), cv2.COLOR_BGR2GRAY)
#                        detected_edges = cv2.GaussianBlur(gray,(1,1),5)
#                        detected_edges = cv2.Canny(detected_edges,30,30*3,apertureSize = 3)
#                        print"..."
#                        im = cv2.bitwise_and(im,im,mask = detected_edges)            
#                        ##end canny            
                        im = im.convert("L")
                        # resize to given size (if given)
                                               
                        if (self.sz is not None) and isinstance(self.sz, tuple) and (len(self.sz) == 2):
                            im = im.resize(self.sz, Image.ANTIALIAS)
                        self.data.append(np.asarray(im, dtype=np.uint8))
                        self.labels.append(c)
                    except :
                            print "Unexpected error:" , sys.exc_info()[0]
                self.names[c] = subdirname
                c = c+1
        self.labels = np.array(self.labels, dtype=np.int)
        
    def readFromCSV(self, filename):
        # <filename>;<classId>;<groupId>
        data = [ [str(line[0]), int(line[1]),int(line[2])] for line in csv.reader(open(filename, 'rb'), delimiter=";")]
        self.labels = np.array([item[1] for item in data])
        self.groups = np.array([item[2] for item in data])
        print self.labels
        print self.groups
        for item in data:
            im_filename = item[0]
            print im_filename
            im = Image.open(os.path.join(im_filename))
            im = im.convert("L")
            # resize to given size (if given)
            if (self.sz is not None) and isinstance(self.sz, tuple) and (len(self.sz) == 2):
                im = im.resize(self.sz, Image.ANTIALIAS)
            self.data.append(np.asarray(im, dtype=np.uint8))
