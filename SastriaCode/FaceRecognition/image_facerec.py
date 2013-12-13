#    Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
#    Released to public domain under terms of the BSD Simplified license.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are met:
#        * Redistributions of source code must retain the above copyright
#          notice, this list of conditions and the following disclaimer.
#        * Redistributions in binary form must reproduce the above copyright
#          notice, this list of conditions and the following disclaimer in the
#          documentation and/or other materials provided with the distribution.
#        * Neither the name of the organization nor the names of its contributors 
#          may be used to endorse or promote products derived from this software 
#          without specific prior written permission.
#
#    See <http://www.opensource.org/licenses/bsd-license>
import cv2
import cv
import os
# cv2 helper
from helper.common import *
from helper.video import *
# add facerec to system path
import sys
sys.path.append("../..")
sys.path.append("../../../..")
sys.path.append("../../..")

# facerec imports
from facerec.dataset import DataSet
from facedet.detector import CascadedDetector
from facerec.preprocessing import TanTriggsPreprocessing,  HistogramEqualization,  Canny
from facerec.feature import LBP
from facerec.classifier import NearestNeighbor
from facerec.operators import ChainOperator ,  CombineOperator,  ChainOperator3
from facerec.model import PredictableModel
from facerec.distance import ChiSquareDistance

from isa_resource import *

help_message = '''USAGE: videofacerec.py [<video source>] [<face database>]

Keys:
  ESC   - exit
'''
class Values(object):
    def __init__ (self, nframes_start, nframes_stop,    name_tr, interv=[] ):
            self.nframes_start =nframes_start
            self.nframes_stop=nframes_stop
            self.interv=interv #una forma del tipo [[frame iniziale, frame finale], [frame iniziale, frame finale]]
            self.val_atteso=[0, 0] # primo valore somma, secondo valore numero addendi
            self.val_int=[]
            self.name_tr=name_tr
    def run(self,  nfra,  val):
        #print ".",  self.nframes_start,  " ",  self.nframes_stop,  " ",  nfra,  val
        if (nfra>self.nframes_start) and (self.nframes_stop>nfra) :
            self.val_atteso[0]=self.val_atteso[0]+val
            self.val_atteso[1]=self.val_atteso[1]+1
            print " valore medio ",self.name_tr, " ",   str(self.val_atteso[0]/self.val_atteso[1])
#        else:
#            print "extra ",  self.name_tr
    def print_value(self):
        try:
                    print " valore medio ",self.name_tr, " ",   str(self.val_atteso[0]/self.val_atteso[1])
        except:
            print "error in values print_value"

class Breackframe(object):       
    def __init__ (self):
            self.buffer=[]
    def add_buffer(self,  nframe,  value):
        self.buffer.append([nframe,  value])
    def run(self):
        if len(self.buffer)<20 :
            return 
        sin= 0
        for s in range(11) :
                sin =sin+self.buffer[s][1]
        des=0
        for d in range(11,  21)   :
            des=des+self.buffer[s][1]
        self.buffer.pop(0)   
        print " scalino ",  str(sin-des)   
        if abs(sin-des)> 80 :
            print "------------------------------------------------------------------------------"
class App(object):
    def __init__(self, video_src, dataset_fn, face_sz=(130,130), cascade_fn="haarcascade_frontalface_alt2.xml"):
        self.face_sz = face_sz
        self.cam = create_capture(video_src)
        ret, self.frame = self.cam.read()
        #print " rectangle and frame ",  ret, self.frame
        self.detector = CascadedDetector(cascade_fn=cascade_fn, minNeighbors=5, scaleFactor=1.1)
        # define feature extraction chain & and classifier) 
        #feature = ChainOperator( TanTriggsPreprocessing(), LBP())
        feature = ChainOperator(HistogramEqualization(), LBP())
        ##feature = ChainOperator(Canny(), LBP())
        classifier = NearestNeighbor(dist_metric=ChiSquareDistance())
        # build the predictable model
        self.predictor = PredictableModel(feature, classifier)
        # read the data & compute the predictor
        self.dataSet = DataSet(filename=dataset_fn,sz=self.face_sz)
        self.predictor.compute(self.dataSet.data,self.dataSet.labels)
        #self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  460)
        self.vm =[] # [avra la forma somma, iterazioni ] vm est il valore medio o valore atteso, il primo valore est il valore medio della traccia di riferimento     

    #def run_prediction(self,  isa_video,  dir_image,  info_file_prediction):
    def run_prediction(self,  isa_video,  info_file_prediction):
            #while True:
            #while True:
            index_tr=0
            for tr in isa_video._track_list :
                index_tr= index_tr +1
                bbs = tr._bbList
                #step = len(tr._bbList)/10
                print "len(bbs) ",   len(bbs)
                index_img=1
                for i in range(len(bbs)):
                        print i
                        bb=bbs[i]
                        time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
                        frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
#                        print "time_vid ",  time_vid,  " ----  ",  "frame_vid  ", frame_vid,  " type(frame_vid) ",  type(frame_vid)
#                        print bb._frame
#                        print type(bb._frame)
#                        print type(self.cam)
                        #self.cam.set(cv.CV_CAP_PROP_POS_MSEC,  10000  )
                        self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  bb._frame  )
                        #print bb._frame
                        ret, frame = self.cam.read()
                        img =  cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
                        imgout = None
                        imgout = img.copy()
                        x0,y0,x1,y1 = int(bb._position[0]), int(bb._position[1]), int(bb._position[2]), int(bb._position[3])
                        #print x0,y0,x1,y1
                        
                        #face = img[y0:y1, x0:x1]
                        face =img[int(y0):int(y1), int(x0):int(x1)] 
                        #print " face ok"
                        #print dir_image+"/"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm"
                        #cv2.imwrite(dir_image+"/../"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm" , face )
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        #index_img =index_img+1
                        print " self.face_sz ",  self.face_sz
                        face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                        prediction = self.predictor.predict(face)[0]
                        bad_prediction = self.predictor.predict(face)[1]
                        print  "prediction ",  prediction,  " distance ",  str(self.predictor.predict(face)[1]['distances'])

                        cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                        draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
                        info_xml=open(info_file_prediction, "a+b")
                        info_xml.write(     "<BBox><label>" +str(self.dataSet.names[prediction])+ "</label><confidence>"+str(self.predictor.predict(face)[1]['distances'][0])+"</confidence><frame>"+str(bb._frame) +"</frame> <time>"+str(time_vid) +"</time><id>"+str(time_vid)+"</id> <x>"+str(x0)+"</x><y>"+str(y0)+"</y><w>"+str(x1)+"</w><h>"+str(y1)+"</h><text>222</text></BBox>")
                        info_xml.close()                        
                        cv2.imshow('videofacerec', imgout)
                        ch = cv2.waitKey(10)
                        if ch == 27:
                            break



    def run(self,  info_file):
#        value_tr1=Values(10,  460,  "traccia1")
#        value_tr2=Values(480,  680,  "------traccia2")
#        value_tr3=Values(1600,  1780,  "-----------traccia3")
#        breackframe =Breackframe()
        while True:
            ret, frame = self.cam.read()
            # resize the frame to half the original size
            time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
            frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
#            if frame_vid > 1781 :
#                print value_tr1.print_value()
#                print value_tr2.print_value()
#                print value_tr3.print_value()
                
            #print "frame_vid ",  frame_vid
#            info_xml= open("test_info.xml", "a+b")
#            info_xml.write(str(time_vid) )
#            info_xml.close()
            img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
            for i,r in enumerate(self.detector.detect(img)):
                x0,y0,x1,y1 = r
                #print " coordinate ",  x0,y0,x1,y1

                #print "frame ",  frame
                # get face, convert to grayscale & resize to face_sz
                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
#                gray = cv2.cvtColor(img[y0:y1, x0:x1],cv2.COLOR_BGR2GRAY)
#                
#                ### canny
#                detected_edges = cv2.GaussianBlur(gray,(1,1),5)
#                #detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*int(ratioTh),apertureSize = int(kernel_size))
#                detected_edges = cv2.Canny(detected_edges,30,30*3,apertureSize = 3)
#                face = cv2.bitwise_and(face,face,mask = detected_edges)
#                ### end canny
#                
                face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                # get a prediction
                prediction = self.predictor.predict(face)[0]
                bad_prediction = self.predictor.predict(face)[1]
                
#                print "predictor ",  self.predictor.predict(face)
#                print "predict ", prediction
#                print "bad_predict ", bad_prediction 
#                    
                # draw the face area
                cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                # draw the predicted name (folder name...)
                draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
                
#                value_tr1.run(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                value_tr2.run(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                breackframe.add_buffer(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                breackframe.run()
                print self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances'])
                info_xml=open(info_file, "a+b")
                info_xml.write(     "<BBox><label>" +str(self.dataSet.names[prediction])+ "</label><confidence>"+str(self.predictor.predict(face)[1]['distances'][0])+"</confidence><frame>"+str(frame_vid) +"</frame> <time>"+str(time_vid) +"</time><id>"+str(time_vid)+"</id> <x>"+str(x0)+"</x><y>"+str(y0)+"</y><w>"+str(x1)+"</w><h>"+str(y1)+"</h><text>222</text></BBox>")
                info_xml.close()                
            cv2.imshow('videofacerec', imgout)
            # get pressed key
            ch = cv2.waitKey(10)
            if ch == 27:
                break
                
                
                
    def run_prediction_stream(self,  info_file):
#        value_tr1=Values(10,  460,  "traccia1")
#        value_tr2=Values(480,  680,  "------traccia2")
#        value_tr3=Values(1600,  1780,  "-----------traccia3")
#        breackframe =Breackframe()
        while True:
            ret, frame = self.cam.read()
            # resize the frame to half the original size
            time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
            frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
#            if frame_vid > 1781 :
#                print value_tr1.print_value()
#                print value_tr2.print_value()
#                print value_tr3.print_value()
                
            #print "frame_vid ",  frame_vid
#            info_xml= open("test_info.xml", "a+b")
#            info_xml.write(str(time_vid) )
#            info_xml.close()
            img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
            for i,r in enumerate(self.detector.detect(img)):
                x0,y0,x1,y1 = r
                #print " coordinate ",  x0,y0,x1,y1

                #print "frame ",  frame
                # get face, convert to grayscale & resize to face_sz
                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
#                gray = cv2.cvtColor(img[y0:y1, x0:x1],cv2.COLOR_BGR2GRAY)
#                
#                ### canny
#                detected_edges = cv2.GaussianBlur(gray,(1,1),5)
#                #detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*int(ratioTh),apertureSize = int(kernel_size))
#                detected_edges = cv2.Canny(detected_edges,30,30*3,apertureSize = 3)
#                face = cv2.bitwise_and(face,face,mask = detected_edges)
#                ### end canny
#                
                face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                # get a prediction
                prediction = self.predictor.predict(face)[0]
                bad_prediction = self.predictor.predict(face)[1]
                
#                print "predictor ",  self.predictor.predict(face)
#                print "predict ", prediction
#                print "bad_predict ", bad_prediction 
#                    
                # draw the face area
                cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                # draw the predicted name (folder name...)
                draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
                
#                value_tr1.run(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                value_tr2.run(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                breackframe.add_buffer(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                breackframe.run()
                print self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances'])
                info_xml=open(info_file, "a+b")
                info_xml.write(     "<BBox><label>" +str(self.dataSet.names[prediction])+ "</label><confidence>"+str(self.predictor.predict(face)[1]['distances'][0])+"</confidence><frame>"+str(frame_vid) +"</frame> <time>"+str(time_vid) +"</time><id>"+str(time_vid)+"</id> <x>"+str(x0)+"</x><y>"+str(y0)+"</y><w>"+str(x1)+"</w><h>"+str(y1)+"</h><text>222</text></BBox>")
                info_xml.close()                
            cv2.imshow('videofacerec', imgout)
            # get pressed key
            ch = cv2.waitKey(10)
            if ch == 27:
                break
    def run_crop(self,  dir_image):
        index=0
        while True:
            
            ret, frame = self.cam.read()
            time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
            frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)

            img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
            for i,r in enumerate(self.detector.detect(img)):
                index =index +1
                x0,y0,x1,y1 = r
                face = img[y0:y1, x0:x1]
                cv2.imwrite(dir_image+str(index)+".pgm" , face )
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
           
                face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                prediction = self.predictor.predict(face)[0]
                bad_prediction = self.predictor.predict(face)[1]
                

                cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
          
            cv2.imshow('videofacerec', imgout)
            ch = cv2.waitKey(10)
            if ch == 27:
                break

    def run_crop_bb(self,  isa_video,  dir_image):
            #while True:
            index_tr=0
            for tr in isa_video._track_list :
                index_tr= index_tr +1
                
                " Scelgo  dieci bb"
                bbs = tr._bbList
                step = len(tr._bbList)/10
                index_img=1
                for i in range(10) :
                        bb=bbs[i*step]
#                        time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
#                        frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
#                        print "time_vid ",  time_vid,  " ----  ",  "frame_vid  ", frame_vid,  " type(frame_vid) ",  type(frame_vid)
#                        print bb._frame
#                        print type(bb._frame)
#                        print type(self.cam)
                        #self.cam.set(cv.CV_CAP_PROP_POS_MSEC,  10000  )
                        self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  bb._frame  )
                        
                        ret, frame = self.cam.read()
                        img =  cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
                        imgout = img.copy()
                        x0,y0,x1,y1 = bb._position[0], bb._position[1], bb._position[2], bb._position[3]
                        #print x0,y0,x1,y1
                        
                        #face = img[y0:y1, x0:x1]
                        face =img[int(y0):int(y1), int(x0):int(x1)] #da cambiare, bisogna fare il crop
                        #print " face ok"
                        print dir_image+"/"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm"
                        if not os.path.exists(  dir_image+"/"+str (index_tr) +"/s"+str (index_tr) ):
                                os.makedirs( dir_image+"/"+str (index_tr) )
                                os.makedirs( dir_image+"/"+str (index_tr) +"/s"+str (index_tr) )      
                                
                        cv2.imwrite(dir_image+"/"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm" , face )
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        print " ok croppata"
                        #index_img =index_img+1
#                       
#                        face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
#                        prediction = self.predictor.predict(face)[0]
#                        bad_prediction = self.predictor.predict(face)[1]
#                        
#
#                        cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
#                        draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
          
                cv2.imshow('videofacerec', imgout)
                ch = cv2.waitKey(10)
                if ch == 27:
                    break

#    def run_crop_bb_stream(self,  isa_video,  dir_image):
#    
#            #while True:
#            for tr in isa_video._track_list :
#                
#                " Scelgo  dieci bb"
#                bbs = tr._bbList
#                step = len(tr._bbList)/10
#                for i in range(10) :
#                        bb=bbs[i*step]
#                        #time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
#                        #frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
##                        print bb._frame
##                        print type(int(bb._frame) )
##                        print type(self.cam)
#                        self.cam.SetCaptureProperty( cv.CV_CAP_PROP_POS_MSEC,  1000)
#                        #self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  bb._frame  )
#                        ret, frame = self.cam.read()
#                        img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
#                        imgout = img.copy()
#                        x0,y0,x1,y1 = bb._position[0], bb._position[1], bb._position[2], bb._position[3]
#                        face = img[y0:y1, x0:x1]
#                        cv2.imwrite(dir_image+str(index)+".pgm" , face )
#                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
##                   
##                        face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
##                        prediction = self.predictor.predict(face)[0]
##                        bad_prediction = self.predictor.predict(face)[1]
##                        
##
##                        cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
##                        draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
#          
#                cv2.imshow('videofacerec', imgout)
#                ch = cv2.waitKey(10)
#                if ch == 27:
#                    break
#
#

def test1():
    print "test 1"
    import sys
    user = str(os.environ["USER"])
    print help_message
    info_file=""
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 
    video_src = "/home/"+user+"/Videos/test2.mpeg"            
    list_s= os.listdir(base_dataset_fn )
    
    for i in range( len(list_s)-2):
            info_file=base_dataset_fn + "/info_file/"+str(i+1) +".xml"
            
            #dataset_fn ="/home/felix/Desktop/myopencv/FrameMindPositive/att_faces/" #sys.argv[2]
            dataset_fn =base_dataset_fn+"/"+str(i+1) 
            print "dataset_fn ",  dataset_fn

           # start facerec app
            print  "preparo il file di info"
            info_xml=open(info_file, "a+b")
            info_xml.write(     "<BBoxs>")
            info_xml.close()    
            print "start app"
            try:
                App(video_src, dataset_fn).run(info_file)
            except  Exception as er:
                    print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 
            info_xml=open(info_file, "a+b")
            info_xml.write(     "</BBoxs>")
            info_xml.close()    
def test2():
    print "test 12"
    import sys
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/att_faces"
    dir_image ="/home/"+user+"/Pictures/att_test/crop/" 
    video_src = "/home/"+user+"/Videos/test2.mpeg"
    try:
                App(video_src, dataset_fn).run_crop(dir_image)
    except  Exception as er:
                    print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 

def test3():
    print "test 3"
    sys.path.append("home/felix/Desktop/quoqueisa")
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/eva_mendes"
    dir_image ="/home/"+user+"/Pictures/att_test/crop/" 
    video_src = "/home/"+user+"/Videos/test2.mpeg"
    
    ## ottengo una istanza di video
    video_info =None
    try:
            #print dir(isa_resource)
            video_info = Video()
            #print " tracce del video ",  len(video._track_list )
            bb_list =get_bb("/home/felix/Pictures/att_test/info_file/1.xml" )
            print len(bb_list)
            video_info.preprocessing(bb_list)
            print " tracce del video ",  len(video_info._track_list )
    except getopt.error, msg:
            print msg
            print " usage: file_conf.xml"   

    #### croppo immagin
    try:
                App(video_src, dataset_fn).run_crop_bb(video_info,  dir_image)
    except  Exception as er:
                    print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 
    
def test4():
    
    
    print "test 4"
    
    
    sys.path.append("home/felix/Desktop/quoqueisa")
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/att_faces"
    dir_image ="/home/"+user+"/Pictures/att_test" 
    video_src = "/home/"+user+"/Videos/test1.mpeg"
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 

    ## creo il file dei bbox.

    info_xml=open("/home/"+user+"/Videos/test1.xml", "a+b")
    info_xml.write(     "<BBoxs>")
    info_xml.close()    
    print "start app"
    try:
        App(video_src, dataset_fn).run("/home/"+user+"/Videos/test1.xml")
    except  Exception as er:
            print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 
    info_xml=open("/home/"+user+"/Videos/test1.xml", "a+b")
    info_xml.write(     "</BBoxs>")
    info_xml.close()
    
    ## ottengo una istanza di video
    video_info =None
    try:
            #print dir(isa_resource)
            video_info = Video()
            #print " tracce del video ",  len(video._track_list )
            bb_list =get_bb("/home/"+user+"/Videos/test1.xml")
            print len(bb_list)
            video_info.preprocessing(bb_list)
            print " tracce del video ",  len(video_info._track_list )
            for i in range (len(video_info._track_list ) ) :
                        tr=video_info._track_list [i]
                        print " taccia ",  i+1,  "  frame iniziale=",  tr._frame_start,  " frame finale",  tr._frame_stop ,  "  numero bbox ",  len(tr._bbList) 
#                        for b in tr._bbList:
#                            print b._frame
    except getopt.error, msg:
            print msg
            print " usage: file_conf.xml"   

    #### croppo immagin
    try:
                print "start run_crop_bb "
                App(video_src, dataset_fn).run_crop_bb(video_info,  dir_image)
    except  Exception as er:
                    print " error - run crop",  er


    list_s= os.listdir(base_dataset_fn )   
    for i in range( len(list_s)-2):
            info_file_prediction=base_dataset_fn + "/info_file/prediction_"+str(i+1) +".xml"
            print info_file_prediction
            #dataset_fn ="/home/felix/Desktop/myopencv/FrameMindPositive/att_faces/" #sys.argv[2]
            dataset_fn =base_dataset_fn+"/"+str(i+1) 
            print "dataset_fn ",  dataset_fn

           # start facerec app
            print  "preparo il file di info"
            info_xml=open(info_file_prediction, "a+b")
            info_xml.write(     "<BBoxs>")
            info_xml.close()    
            print "start app"
            try:
                App(video_src, dataset_fn).run_prediction(video_info, info_file_prediction)
            except  Exception as er:
                    print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 
            info_xml=open(info_file_prediction, "a+b")
            info_xml.write(     "</BBoxs>")
            info_xml.close()    

def test5():
    print "test 5"
    sys.path.append("home/felix/Desktop/quoqueisa")
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/att_faces"
    dir_image ="/home/"+user+"/Pictures/att_test" 
    video_src = "/home/"+user+"/Videos/test1.mpeg"
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 

    ## ottengo una istanza di video
    video_info =None
    try:
            #print dir(isa_resource)
            video_info = Video()
            #print " tracce del video ",  len(video._track_list )
            bb_list =get_bb("/home/felix/Pictures/att_test_1/info_file/1.xml" )
            print len(bb_list)
            video_info.preprocessing(bb_list)
            print " tracce del video ",  len(video_info._track_list )
            for i in range (len(video_info._track_list ) ) :
                        tr=video_info._track_list [i]
                        print " taccia ",  i+1,  "  frame iniziale=",  tr._frame_start,  " frame finale",  tr._frame_stop ,  "  numero bbox ",  len(tr._bbList) 
#                        for b in tr._bbList:
#                            print b._frame
    except getopt.error, msg:
            print msg
            print " usage: file_conf.xml"   


    list_s= os.listdir(base_dataset_fn )   
    for i in range( len(list_s)-2):
            info_file_prediction=base_dataset_fn + "/info_file/"+str(i+1) +".xml"
            print info_file_prediction
            #dataset_fn ="/home/felix/Desktop/myopencv/FrameMindPositive/att_faces/" #sys.argv[2]
            dataset_fn =base_dataset_fn+"/"+str(i+1) 
            print "dataset_fn ",  dataset_fn

           # start facerec app
            print  "preparo il file di info"
            info_xml=open(info_file_prediction, "a+b")
            info_xml.write(     "<BBoxs>")
            info_xml.close()    
            print "start app"
            try:
                App(video_src, dataset_fn).run_prediction(video_info, info_file_prediction)
            except  Exception as er:
                    print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 
            info_xml=open(info_file_prediction, "a+b")
            info_xml.write(     "</BBoxs>")
            info_xml.close()    



if __name__ == '__main__':
    #test2()
    #test1()
    test4()
    #test5()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    
#    print "--"
#    import sys
#    print help_message
#    info_file="test2_mpeg.xml"
#    if len(sys.argv) < 3:
#        #/home/felix/Videos/videodemo.mov               /home/felix/Desktop/myopencv/FrameMindPositive/att_faces/
#        #sys.exit()
#        print ""
#   # get params
#   #video_src = "/home/felix/Videos/videodemo.mov"  #sys.argv[1]
#    #video_src = "/home/felix/Videos/facere-big-face.mov"  #sys.argv[1]
#    video_src = "/home/felix/Videos/test2.mpeg"  #sys.argv[1]
#    
#    #dataset_fn ="/home/felix/Desktop/myopencv/FrameMindPositive/att_faces/" #sys.argv[2]
#    dataset_fn ="./att_faces/" #sys.argv[2]
#   # start facerec app
#    info_xml=open(info_file, "a+b")
#    info_xml.write(     "<BBoxs>")
#    info_xml.close()    
#    print "start app"
#    try:
#        App(video_src, dataset_fn).run()
#    except :
#            print " error"
#            info_xml=open(info_file, "a+b")
#            info_xml.write(     "</BBoxs>")
#            info_xml.close() 
#    info_xml=open(info_file, "a+b")
#    info_xml.write(     "</BBoxs>")
#    info_xml.close()    
