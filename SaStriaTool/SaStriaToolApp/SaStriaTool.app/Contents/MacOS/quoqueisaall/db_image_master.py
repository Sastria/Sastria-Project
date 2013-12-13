

import os
import sys
import cv2
import cv
import numpy as np
import shutil
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
def read_images_canny(path, sz=None):

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

def read_1_image_canny(path, sz=None):
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
                    cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))        
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
                    cv.EqualizeHist( cv.fromarray(im) , cv.fromarray(im))    
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(777)
    except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
    except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    return [X,y]

def resplit():
    print "start resplit"
    base_directory_rs="/home/felix/Pictures/multi_image/"
    labels={}
    out_dir = None
    all_image = os.listdir(base_directory+"db/s1")

    for img_name in all_image:
#        try:
              print "img_name ",  img_name  
              min_val=1000
              if not os.path.exists(base_directory+"/resplit/s1"):
                        os.makedirs(base_directory+"/resplit/s1") 
                        #print " creo directory" 
              #sposto questa immagine e la uso come db
              shutil.move(base_directory+"db/s1/"+img_name, base_directory+"/resplit/s1")
        
              for i in range(10):

                [X,y] = read_images(   base_directory+"/resplit/", (92,112))
                model = cv2.createLBPHFaceRecognizer()
                model.train(np.asarray(X), np.asarray(y))        
                #model.load(base_directory+"model")      
                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1,1,0,3,8)

                other_image = os.listdir(base_directory+"db/s1")
                #print "other_image ",  other_image
                if i==0:
                  val_min=[1000]
                for other_i in  other_image :
                    #print "read_1_image ",  base_directory+"/mix_image/"+other_i
                    [X2,y2] = read_1_image(base_directory+"/db/s1/"+other_i, (92,112))        
                    [p_label2, p_confidence2] = model.predict(np.asarray(X2[0]))
                    #print  "image ",  labels[p_label2], " confidence ",str(p_confidence2) 
                    #print "------------------",   i
                    if i==0:
                        if val_min[0]>p_confidence2:
                            val_min.remove(val_min[0])
                            val_min.append(p_confidence2)
                            val_min.sort()
                            min_val =val_min[0]
                            print " minore quindi appendo ",  p_confidence2
                        print val_min                                                                                                            

                    if i >0:
                      #print base_directory+"/mix_image/"+other_i,  "  ",  p_confidence2      
#                      try:
                          if  p_confidence2 < min_val+1 :
                            #shutil.copy(base_directory+"/mix_image/"+other_i, base_directory+"/db/s1")
                            shutil.move(base_directory+"/db/s1/"+other_i, base_directory+"/resplit/s1")
                            #print " move image ",base_directory+"/mix_image/"+other_i  
                            try:
                                all_image.remove(other_i)
                            except:
                                    pass
#                      except:
#                        print " shutil move ",base_directory+"/mix_image/"+other_i    
              
              rec_image = os.listdir(base_directory+"/resplit/s1")
              print "-----------------rec_image ",  rec_image
              if len(rec_image)>=2 :
                    print " COPIOOOOOOOO ",base_directory+"/db/s1/"+rec_image[0]  
                    shutil.copy(base_directory+"/resplit/s1/"+rec_image[0], base_directory+"/rec_image/")
              for img in rec_image:
                    shutil.move(base_directory+"/resplit/s1/"+img,base_directory+"/tmp" )
#              else:
#                  for f in rec_image :
#                                          shutil.copy(base_directory+"/db/s1/"+f, base_directory+"/mix_image/")

              shutil.rmtree(base_directory+"resplit/s1")    
    print "---------------------------------"
    print "========FINE   RESPLIT ======================"





if __name__ == "__main__":
    
    risultato_atteso=[1, 2,5, 8, 13, 14, 15, 19, 23,  25, 28, 32]
    
                                #1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 21,23, 25,28, 31, 32
    
    print "start multi image recognition"
    base_directory="/home/felix/Pictures/att_faces3/"
    training_directory= "/home/felix/Pictures/att_faces2/"
    labels={}
    out_dir = None
    all_dir= os.listdir(base_directory)
    #print all_dir
    all_prediction={}
#    ## CERCO DELLE SOGlie
#    for img_dir in all_dir:
#                 other_dir=all_dir
#                 [X,y] = read_images(   training_directory+img_dir, (92,112))
#                 #[X,y] = read_images(   training_directory+"/1/", (92,112))
#                 model = cv2.createLBPHFaceRecognizer()
#                 model.train(np.asarray(X), np.asarray(y)) 
#                 o_dir=img_dir   
#   
#                 all_image = os.listdir(base_directory+o_dir+"/s"+o_dir)
#                 media=0
#                 vec_media=[]
#                 for img_name in all_image:
#                                [X2,y2] = read_1_image(base_directory+o_dir+"/s"+o_dir+"/"+img_name, (92,112)) 
#                                [p_label2, p_confidence2] = model.predict(np.asarray(X2[0]))
#                                #print  "image ",  labels[p_label2], " confidence ",str(p_confidence2) 
#                                #print base_directory+o_dir+"/s"+o_dir+"/"+img_name,  "   === confidece ==>>",  p_confidence2
#                                if p_confidence2 > 50:
#                                        print base_directory+o_dir+"/s"+o_dir+"/"+img_name,  "   === confidece ==>>",  p_confidence2
#                                        os.remove(base_directory+o_dir+"/s"+o_dir+"/"+img_name)
#                                #media=media+int(p_confidence2)
#                                vec_media.append(int(p_confidence2))
#                                
#                 vec_media=vec_media[int( 0.1*len(vec_media)) : int( 0.9*len(vec_media))]               
#                 for i  in range(len(vec_media)):
#                   media =media+vec_media[i] 
#                 media =media /len(vec_media)
#                 trovato =0
#                 step=0.0001
#                 while not trovato <0:
#                    if all_prediction.has_key(media+trovato*step):
#                        trovato=trovato+1
#                    else:
#                       all_prediction[media+trovato*step]=[img_dir, o_dir]    
#                       trovato=-1
#                 trovato=0       
#    _key= all_prediction.keys()
#    _key.sort()
#    for k in _key:
#        print k,  "   ",  all_prediction[k]
#    
#    exit()
#    
#    
    
    faces={}
    for i in range(len(all_dir)):
        faces[i+1]=i+1
    print faces

    soglia=60
    while soglia<61:
            print "====================================================="
            print "============",  soglia ,   "===========================" 
            for img_dir in all_dir:
                 #print "img_dir ",  img_dir
                 other_dir=all_dir
                 #other_dir.remove(img_dir)   
                 [X,y] = read_images(   training_directory+img_dir, (92,112))
                 #[X,y] = read_images(   training_directory+"/1/", (92,112))
                 model = cv2.createLBPHFaceRecognizer()
                 model.train(np.asarray(X), np.asarray(y))            
                 for o_dir in other_dir:     
                                all_image = os.listdir(base_directory+o_dir+"/s"+o_dir)
                                media=0
                                vec_media=[]
                                for img_name in all_image:
                                                [X2,y2] = read_1_image(base_directory+o_dir+"/s"+o_dir+"/"+img_name, (92,112)) 
                                                [p_label2, p_confidence2] = model.predict(np.asarray(X2[0]))
                                                #print  "image ",  labels[p_label2], " confidence ",str(p_confidence2) 
                                                #media=media+int(p_confidence2)
                                                vec_media.append(int(p_confidence2))
                                                
                                vec_media=vec_media[int( 0.1*len(vec_media)) : int( 0.9*len(vec_media))]               
                                for i  in range(len(vec_media)):
                                   media =media+vec_media[i] 
                                media =media /len(vec_media)
                                trovato =0
                                step=0.0001
                                while not trovato <0:
                                    if all_prediction.has_key(media+trovato*step):
                                        trovato=trovato+1
                                    else:
                                       all_prediction[media+trovato*step]=[img_dir, o_dir]    
                                       trovato=-1
                                trovato=0       
#                                if media< soglia and media >0:
#                                        print " distanza media tra il segmento ",  img_dir ,  " e il segmento ",  o_dir,  " corrisponde a ",  media
#                                        if int(img_dir)>int(o_dir):
#                                           if int(img_dir)-int(o_dir)>1 and faces[int(img_dir)]==int(img_dir):
#                                               faces[int(img_dir)]=int(o_dir)
#                                        else:
#                                           if int(o_dir)-int(img_dir)>1 and faces[int(o_dir)]==int(o_dir):
#                                               faces[int(o_dir)]=int(img_dir)                                            
#
#            print faces
            soglia =soglia+10
     
    result={}   
    ks=all_prediction.keys()
    print "all_prediction ",  all_prediction
    ks.sort()
    for media in ks:
            dr=all_prediction[media]
            img_dir=dr[0]
            o_dir=dr[1]
            if media< soglia and media >0:
                    #print " distanza media tra il segmento ",  img_dir ,  " e il segmento ",  o_dir,  " corrisponde a ",  media
                    if int(img_dir)>int(o_dir):
                       if int(img_dir)-int(o_dir)>1 and faces[int(img_dir)]==int(img_dir):
                           faces[int(img_dir)]=int(o_dir)
                           print img_dir,  "---->",  o_dir
                    else:
                       if int(o_dir)-int(img_dir)>1 and faces[int(o_dir)]==int(o_dir):
                           faces[int(o_dir)]=int(img_dir) 
                           print o_dir,  "====>",  img_dir     
                                               

    for f in faces.itervalues(): 
        result[f]=1
    print "---------------------------------"
    print "---------------------------------"
    print "---------------------------------"
    for f in result:
      print f   
            
    print "---------------------------------"
    print "========FINE======================"
    
    
