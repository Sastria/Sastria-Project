print " VIDEOFACEREC ON QUOQUEISAALL"

import os
import sys
dir_base_fr=str(os.getcwd())+"/../"
print "ROOT DIRECTORY", dir_base_fr	
dataset_fn=dir_base_fr+"/Parameters/resource/att_faces/"



print "dataset_fn ", dataset_fn


sys.path.insert(0,'./lib/scipy-0.13.0.dev_c31f167_20130617-py2.7-macosx-10.8-intel.egg')
sys.path.insert(0,'./lib/numpy-1.8.0.dev_dff8c94_20130602-py2.7-macosx-10.8-intel.egg')

#print str(os.getcwd())
from ctypes import cdll
mydll = cdll.LoadLibrary('./lib/cv2.so')
cdll.LoadLibrary('./lib/_imaging.so')  #/opt/local/lib/libjpeg.9.dylib

cdll.LoadLibrary('./lib/_imagingft.so')  # /opt/local/lib/libfreetype.6.dylib
cdll.LoadLibrary('./lib/_imagingmath.so')
cdll.LoadLibrary('./lib/_imagingtk.so')

sys.path.append('./lib')
#print sys.path

import PIL
import PIL.Image as Image
import scipy
import numpy

import cv2
import cv
print '/n'
print PIL
print scipy
import numpy
print numpy

print cv2


sys.path.append('./lib')
sys.path.append('./quoqueisaall')
# cv2 helper
from helper.common import *
from helper.video import *
# add facerec to system path
import socket

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
import shelve
class WhoIs():
    def __init__(self, nrFrames):
	print "whois she lava ", str(os.getcwd())+'/Data/db_shelve.db'
	self.shelve_db = shelve.open( str(os.getcwd())+'/Data/db_shelve.db')
        self.nrFrames=nrFrames
        self.bb={}
        self.whois=None
    def append(self, val):
        if self.bb.has_key(val):
            t=self.bb[val]
            t=t+1
            self.bb[val]=t
        else: 
            self.bb[val]=1
    def run(self):
        ks = self.bb.keys()
	    
        for k in ks:
            val = self.bb[k]
            if val > self.nrFrames*0.6:
                self.whois=k
                return k
        return None
    def save_as_db(self, track_id):
	if self.whois==None:
	    self.whois="UNDEF"  
        if not self.whois==None:
            self.shelve_db[str(track_id)]=[self.whois]
            print " track ", track_id, " is ", self.whois
	    if self.shelve_db.has_key(str(self.whois)):
		tmp = self.shelve_db[str(self.whois)]
		tmp.append(str(track_id))
		self.shelve_db[str(self.whois)]=tmp
		
    def save4statistic(self, video_info, yml_out):
		"""
	        tmp=""
		pers_dif=0
		label_face=shelve.open(dir_base_fr+'/quoqueisaall/Data/db_shelve.db')
		file_yml=open(yml_out, "a+")
		file_yml.write("______________")
		file_yml.write("Numero di persone : "+str(len(label_face.keys()) ) +"\n")
		file_yml.write("Nome persone : ")
		for k in label_face.keys():
		    if tmp.find("#"+str(k)+"#")>-1:
			tmp=tmp+"#"+str(k)+"#"
			pers_dif=pers_dif+1
			file_yml.write( k )
			file_yml.write( "," )
		file_yml.write("\nNumero di persone differenti: "+pers_dif+"\n")				    
		file_yml.close()    
		"""
		
		print " salva info per metriche"
		label_face=shelve.open(dir_base_fr+'/quoqueisaall/Data/db_shelve.db')
		file_yml=open(yml_out, "a+")
		file_yml.write("______________\n")
		num_p=0
		pers_dif=0
		tmp=""
		file_yml.write("Nome persone : ")
		for k in label_face.items():
		    if not str(k[1][0]).isdigit() :
		    	    print " nome personaggio", k[1][0]
			    num_p = num_p +1	
			    if tmp.find("#"+str(k[1][0])+"#")==-1:
				tmp=tmp+"#"+str(k[1][0])+"#"			    
				file_yml.write( k[1][0] )
				file_yml.write( "," )
				pers_dif=pers_dif+1
		file_yml.write("\nNumero di persone : "+str( num_p) +"\n")
		file_yml.write("\nNumero di persone differenti: "+str( pers_dif) +"\n")
		file_yml.write("\nRecognition Execution time (in sec) : "+str( END_TIME-START_TIME) +"\n")
	
		file_yml.close()    		
		
    def save_shelva_xml(self, video_info, xml_out):
	#label_face = shelve.open('/Users/labcontenuti/Documents/FaceRecognition/resource/db_shelve_ex')
	label_face=self.shelve_db
	print "label_face", label_face
	label_face=shelve.open(dir_base_fr+'/quoqueisaall/Data/db_shelve.db')
	print " dir del db", dir_base_fr+'/quoqueisaall/Data/db_shelve.db'
	print "DB in uso ", label_face
	xml_file="<root><nrFrames>"+str(video_info._frame_number)+"</nrFrames><Faces>"
	file_xml=open(xml_out, "w+")
	file_xml.write(xml_file)
	ks=label_face.keys()
	ks.sort()
	print ks
	db_name=label_face  #{}
	for id_tr in ks:
	    print id_tr
	    u=unicode(id_tr)
	    if u.isnumeric():	    
		nome=label_face[id_tr][0]
		#if label_face.has_key(nome):
		if db_name.has_key(nome):
		    tmp=db_name[nome]
		    if not tmp.count(id_tr)>0:
			tmp.append(id_tr)
			db_name[nome]=tmp
		else:
		    db_name[nome]=[id_tr]
	print "db name ",db_name
	ks=db_name.keys()
	ks.sort()	
	for key in ks :
	    u=unicode(key)
	    if not u.isnumeric():
		print "key ",key
		#xml_file=xml_file+"<Face>"
		file_xml.writelines("<Face>\n")
		
		#xml_file=xml_file+"<label>"+str(key)+"</label>"
		file_xml.writelines("<label>"+str(key)+"</label>\n")
		
		#xml_file=xml_file+"<r>"+str(random.randint(0,255))+"</r>"+"<g>"+str(random.randint(0,255))+"</g>"+"<b>"+str(random.randint(0,255))+"</b>"
		file_xml.writelines("<r>"+str(random.randint(0,255))+"</r>"+"<g>"+str(random.randint(0,255))+"</g>"+"<b>"+str(random.randint(0,255))+"</b>\n")
		
		
		for val in db_name[key]:
			print "val ",  val
			k=val

			bbs=video_info._track_list[int(val)-1]._bbList
			for b in bbs:
				#xml_file=xml_file+"<BBox>"
				file_xml.writelines("<BBox>\n")
				
				#xml_file=xml_file+"<frameId>"+str(int(b._frame)).split(".")[0]+"</frameId><x>"+b._position[0]+"</x><y>"+b._position[1]+"</y><w>"+b._position[2]+"</w><h>"+b._position[0]+"</h>"
				file_xml.writelines("<frameId>"+str(int(b._frame)).split(".")[0]+"</frameId>\n<x>"+b._position[0]+"</x>\n<y>"+b._position[1]+"</y>\n<w>"+b._position[2]+"</w>\n<h>"+b._position[0]+"</h>\n")
				
				
				#xml_file=xml_file+"<confidence>100</confidence>"
				file_xml.writelines("<confidence>100</confidence>\n")
				
				
				
				"""
				xml_file=xml_file+"<confidence>"+str(val[1]).split(".")[0] +"</confidence>"
				if float(val[1])<90:
					print "cerco altre persone ", val[1]
					for o_key in ks:
						if not o_key==key:
							print label_face[o_key]
							for tmp_val in label_face[o_key]:
								if int(tmp_val[0])==int(k):
									#print "<person><label>"+str(o_key)+"</label><confidence>"+str(tmp_val[1]) +"</confidence> </person>"
									xml_file=xml_file+"<person><label>"+str(o_key)+"</label><confidence>"+str(tmp_val[1]).split(".")[0] +"</confidence> </person>"
				"""
				#xml_file=xml_file+"</BBox>"
				file_xml.writelines("</BBox>\n")
		#xml_file=xml_file+"</Face>"
		file_xml.writelines("</Face>\n")
	#xml_file=xml_file+"</Faces>"
	file_xml.writelines("</Faces></root>")
	#print xml_file
	file_xml.close()
	
	
	file_xml.close()
	
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
        #print " creo il predictor "
        self.predictor.compute(self.dataSet.data,self.dataSet.labels)
        #print "predictor ", self.predictor
        #self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  460)
        self.vm =[] # [avra la forma somma, iterazioni ] vm est il valore medio o valore atteso, il primo valore est il valore medio della traccia di riferimento     
        #print " finito init della classe app"
    #def run_prediction(self,  isa_video,  dir_image,  info_file_prediction):
    def run_prediction(self,  isa_video,  info_file_prediction):
            #while True:
            #while True:
            index_tr=0
            for tr in isa_video._track_list :
                index_tr= index_tr +1
                bbs = tr._bbList
                #step = len(tr._bbList)/10
                #print "len(bbs) ",   len(bbs)
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
                        #img =  cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
                        img=frame
                        imgout = None
                        imgout = img.copy()
                        x0,y0,x1,y1 = int(bb._position[0]), int(bb._position[1]), int(bb._position[2]), int(bb._position[3])
                        #print x0,y0,x1,y1
                        
                        #face = img[y0:y1, x0:x1]
                        face =img[int(y0):int(y1)+int(y0), int(x0):int(x1)+int(x0)] 
                        #print " face ok"
                        #print dir_image+"/"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm"
                        #cv2.imwrite(dir_image+"/../"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm" , face )
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        #index_img =index_img+1
                        #print " self.face_sz ",  self.face_sz
                        face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                        prediction = self.predictor.predict(face)[0]
                        bad_prediction = self.predictor.predict(face)[1]
                        print  "prediction ",  prediction,  " distance ",  str(self.predictor.predict(face)[1]['distances'])

                        cv2.rectangle(imgout, (x0,y0),(x0+x1,y0+y1),(0,255,0),2)
                        draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
                        #info_xml=open(info_file_prediction, "a+b")
                        #info_xml.write(     "<BBox><label>" +str(self.dataSet.names[prediction])+ "</label><confidence>"+str(self.predictor.predict(face)[1]['distances'][0])+"</confidence><frame>"+str(bb._frame) +"</frame> <time>"+str(time_vid) +"</time><id>"+str(time_vid)+"</id> <x>"+str(x0)+"</x><y>"+str(y0)+"</y><w>"+str(x1)+"</w><h>"+str(y1)+"</h><text>222</text></BBox>")
                        #info_xml.close()                        
                        cv2.imshow('videofacerec', imgout)
                        ch = cv2.waitKey(10)
                        if ch == 27:
                            break

########################
####MAKE DB
    def make_db(self,  isa_video,  dir_image):

            index_tr=0
            print "start method make db"
            for tr in isa_video._track_list :
                index_tr= index_tr +1
                bbs = tr._bbList
                index_img=1                
                for i in xrange(len(bbs)):
		    print i
                    try:
                        bb=bbs[i]
                        frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
                        self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  bb._frame  )
                        ret, frame = self.cam.read()
                        img=frame
                        imgout = None
                        imgout = img.copy()

                        x0,y0,x1,y1 = int(bb._position[0]), int(bb._position[1]), int(bb._position[2]), int(bb._position[3])
                        #print x0,y0,x1,y1
                        
                        #face = img[y0:y1, x0:x1]
                        face =img[int(y0):int(y1)+int(y0), int(x0):int(x1)+int(x0)] 
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                        print "cerco directory ",  dir_image+"/"+str (index_tr)
                        
                        if not os.path.exists(  dir_image+"/"+str (index_tr)  ):
                                print "creo directory ",  dir_image+"/"+str (index_tr) 
                                os.makedirs( dir_image+"/"+str (index_tr) )
                                              
                        if not os.path.exists(  dir_image+"/"+str (index_tr) +"/s"+str (index_tr) ):
                                print " creo sub directory ",  dir_image+"/"+str (index_tr) +"/s"+str (index_tr) 
                                os.makedirs( dir_image+"/"+str (index_tr) +"/s"+str (index_tr) )      
                        print " salvo immagine ", dir_image+"/"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm"        
                        cv2.imwrite(dir_image+"/"+str (index_tr) +"/s"+str (index_tr) +"/"+str(i)+".pgm" , face )
		    except Exception, e :
			print e
    
		
		
#############
############
### END RUN DB
    def run_db(self,  isa_video, xml_out):
            #while True:
            #while True:
            HOST = ''                 # local host
            PORT = 8889         
            try:    
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)     
                sock.connect((HOST, PORT))
            except:
                            print "socket error "
            index_tr=0
            print "start  run db"
            num_tr=len(isa_video._track_list)
            perc=0
            perc_step=100.0/num_tr-1
	    		
	    
            for tr in isa_video._track_list :
             try:
                index_tr= index_tr +1
                bbs = tr._bbList
                #step = len(tr._bbList)/10
                #print "len(bbs) ",   len(bbs)
                index_img=1
                who=WhoIs(0)
#                 perc=perc+perc_step
                print ">>>>>>>>>>>>>>>>>>>> perc solo tracce ", perc
                for i in range(len(bbs)):
                        #perc=perc+1.0/(len(bbs))
                        perc=perc+perc_step/(len(bbs))
                        #perc=perc+0.1
                        #print ">>>>>>>>>>>>> perc definitivo ", perc
                        try:
                            #print "sending data ", sock, "  perc", str(int(perc)) 
                            sock.send( str(int(perc)) )
                        except:
                            #print "socket error"
			    pass
                        bb=bbs[i]
                        frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
                        self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  bb._frame  )
                        ret, frame = self.cam.read()
                        #img =  cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
                        img=frame			
			try:
			    imgout = None
			    imgout = img.copy()
			    x0,y0,x1,y1 = int(bb._position[0]), int(bb._position[1]), int(bb._position[2]), int(bb._position[3])
			    #print x0,y0,x1,y1
			    
			    #face = img[y0:y1, x0:x1]
			    face =img[int(y0):int(y1)+int(y0), int(x0):int(x1)+int(x0)] 
			    face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
			    face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
			    prediction = self.predictor.predict(face)[0]
			    bad_prediction = self.predictor.predict(face)[1]
			    dst =int (self.predictor.predict(face)[1]['distances'])                        
			    cv2.rectangle(imgout, (x0,y0),(x0+x1,y0+y1),(0,255,0),2)
			    if dst<40:                            
				draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
				who.append(str(self.dataSet.names[prediction]))
				who.nrFrames=who.nrFrames+1
				who.shelve_db.sync()
    #                        info_xml=open(info_file_prediction, "a+b")
    #                        info_xml.write(     "<BBox><label>" +str(self.dataSet.names[prediction])+ "</label><confidence>"+str(self.predictor.predict(face)[1]['distances'][0])+"</confidence><frame>"+str(bb._frame) +"</frame> <time>"+str(time_vid) +"</time><id>"+str(time_vid)+"</id> <x>"+str(x0)+"</x><y>"+str(y0)+"</y><w>"+str(x1)+"</w><h>"+str(y1)+"</h><text>222</text></BBox>")
    #                        info_xml.close()                        
			    cv2.imshow('videofacerec', imgout)
			    ch = cv2.waitKey(10)
			    if ch == 27:
				break

			except Exception, e:
			    print "RUN DB ERROR B", e
             
                try:
                            print "sending data ", sock, "  perc", str(int(100)) 
                            sock.send( str(int(100)) )
                except:
                            print "socket error"
                sock.close()
                who.run()
                who.save_as_db(index_tr)
		who.shelve_db.sync()
             except Exception, e:
			    print "RUN DB ERROR A ", e

#############
############
### END RUN DB
#############
############
### END RUN VIDEOWRITER
    def run_VideoWriter(self,  isa_video, video_name):
		index_tr=0
		print "start  run VideoWriter"
		width = 800 #np.size(frame, 1) 
		height = 600# np.size(frame, 0)
		#writer = cv2.VideoWriter(filename, fourcc=cv.CV_FOURCC('i','Y', 'U', 'V'),fps=15,  frameSize=(width, height))	
		writer=None
		for tr in isa_video._track_list :
		    index_tr= index_tr +1
		    bbs = tr._bbList
		    index_img=1
		    who=WhoIs(0)
		    
		    for i in xrange(len(bbs)):
			try:
			    bb=bbs[i]
			    frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
			    self.cam.set(cv.CV_CAP_PROP_POS_FRAMES,  bb._frame  )
			    ret, frame = self.cam.read()
			    #img =  cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
			    img=frame
			    if writer ==None: 
				writer = cv2.VideoWriter(filename=video_name, fourcc=cv.CV_FOURCC('X','V','I','D'),fps=15, frameSize=(frame.shape[1] , frame.shape[0] ))    			    
			    imgout = None
			    imgout = img.copy()
			    x0,y0,x1,y1 = int(bb._position[0]), int(bb._position[1]), int(bb._position[2]), int(bb._position[3])
			    #print x0,y0,x1,y1
			    
			    #face = img[y0:y1, x0:x1]
			    face =img[int(y0):int(y1)+int(y0), int(x0):int(x1)+int(x0)] 
			    face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
			    face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)

			    cv2.rectangle(imgout, (x0,y0),(x0+x1,y0+y1),(0,255,0),2)
			    name=who.shelve_db[str(index_tr)][0]
			    draw_str(imgout, (x0-20,y0-20), name )
			    writer.write(imgout)
			    cv2.imshow('videofacerec', imgout)
			    ch = cv2.waitKey(10)
			    if ch == 27:
				break
			except Exception, e:
			    print " RUN VIDEO WRITER ERROR ", e

    #############
    ############
    ### END RUN DB

#############
############
### END RUN4image
    def run4image(self,  face):
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)          
                face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
                prediction = self.predictor.predict(face)[0]
                bad_prediction = self.predictor.predict(face)[1]
                
                print "predictor ",  self.predictor.predict(face)
                print "predict ", prediction
                print "bad_predict ", bad_prediction 
                cv2.imshow('videofacerec', imgout)
                # get pressed key
                ch = cv2.waitKey(10)

###############
    
    ### END RUN info file
    def run_1(self, video_info, frame, video_name):
            shelva_db=WhoIs(0)
	    width = 800 #np.size(frame, 1) 
	    height = 600# np.size(frame, 0)
	    w =None
	    writer=None
	    if not frame==None:
		self.cam.set(cv.CV_CAP_PROP_POS_FRAMES, int(frame))
            while True:
                ret, frame = self.cam.read()
		#print frame.shape[1] , frame.shape[0] 
		if writer==None:
		    writer = cv2.VideoWriter(filename=video_name, fourcc=cv.CV_FOURCC('X','V','I','D'),fps=15, frameSize=(frame.shape[1] , frame.shape[0] ))	
		    
		if w==None:
		    w = cv.CreateVideoWriter(video_name+".1",cv.CV_FOURCC('X','V','I','D'),25,(frame.shape[1] , frame.shape[0]))
            
                # resize the frame to half the original size
                time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
                frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
		bb_of_fraame=[]
		for tr in video_info._track_list :
		    for bb in tr._bbList:
			if int(bb._frame)==frame_vid:
			    bb_of_fraame.append(bb)
			
                img = cv2.resize(frame, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_CUBIC)
                imgout = img.copy()
		try:
		    for bb in bb_of_fraame:
			x0,y0,x1,y1 = (int(bb._position[0])/10)*10, (int(bb._position[1])/10)*10, (int(bb._position[2])/10)*10, (int(bb._position[3])/10)*10
			#cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
			draw_str(imgout, (x0-20,y0-20),  shelva_db.shelve_db[str(bb._track._id)][0])
		except :    
		    print " tr ", bb._track._id
		writer.write(imgout) 
		#cv.WriteFrame(w,imgout)
                cv2.imshow(video_name, imgout)
		
                ch = cv2.waitKey(10)
                if ch == 27:
                    break

    def run_writevideo(self, video_info, frame, video_name):
        shelva_db=WhoIs(0)
        width = 800 #np.size(frame, 1) 
        height = 600# np.size(frame, 0)
        w =None
        writer=None
        if not frame==None:
            self.cam.set(cv.CV_CAP_PROP_POS_FRAMES, int(frame))
        while True:
            ret, frame = self.cam.read()
	    if frame ==None:
		break		
            if writer==None:
                writer = cv2.VideoWriter(filename=video_name, fourcc=cv.CV_FOURCC('X','V','I','D'),fps=15, frameSize=(frame.shape[1] , frame.shape[0] ))    
                time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
            frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
            bb_of_fraame=[]
            for tr in video_info._track_list :
                for bb in tr._bbList:
                    if int(bb._frame)==frame_vid:
                        bb_of_fraame.append(bb)
            
                        img = cv2.resize(frame, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_CUBIC)
                        imgout = img.copy()
                        try:
                            for bb in bb_of_fraame:
                                x0,y0,x1,y1 = (int(bb._position[0])/10)*10, (int(bb._position[1])/10)*10, (int(bb._position[2])/10)*10, (int(bb._position[3])/10)*10
                                draw_str(imgout, (x0-20,y0-20),  shelva_db.shelve_db[str(bb._track._id)][0])
                        except :    
                            print " tr ", bb._track._id
                        writer.write(imgout) 
                        cv2.imshow(video_name, imgout)
        
                ch = cv2.waitKey(10)
                if ch == 27:
                    break
		break

#############
############
### END RUN info file
    def run(self):

        while True:
            ret, frame = self.cam.read()
            time_vid= self.cam.get(cv.CV_CAP_PROP_POS_MSEC)
            frame_vid= self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)

            img = cv2.resize(frame, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_CUBIC)
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
		if prediction<40:                
		    cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
		    #draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" "+ str(self.predictor.predict(face)[1]['distances']) )
		    draw_str(imgout, (x0-20,y0-20), self.dataSet.names[prediction]+" dist:"+ str(prediction) +" next:"+str(self.predictor.predict(face)[1]['distances']) )
		    
#                value_tr1.run(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                value_tr2.run(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                breackframe.add_buffer(frame_vid,  self.predictor.predict(face)[1]['distances'])
#                breackframe.run()
               
            cv2.imshow('...', imgout)
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
                        #print self.cam.get(cv.CV_CAP_PROP_POS_FRAMES)
                        ret, frame = self.cam.read()
                        #img =  cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
                        img =frame
                        imgout = img.copy()
                        x0,y0,x1,y1 = bb._position[0], bb._position[1], bb._position[2], bb._position[3]
                        print x0,y0,x1,y1
                        #print int(y0),  " ",  int(y0)+int(y1)," ",   int(x0),  "  ",  int(x0)+int(x1)
                        face =img[int(y0):int(y0)+int(y1), int(x0):int(x0)+int(x1)]
                        #face=img[int(42):int(185+42), int(385):int(385+185)]
                        print " face ok"
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


def test4_bis():
    
    
    print "test 4 bis"
    sys.path.append("home/felix/Desktop/quoqueisa")
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/att_faces1"
    dir_image ="/home/"+user+"/Pictures/att_test" 
    video_src = "/home/"+user+"/Videos/videodemo.mov"
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 
#
#    ## creo il file dei bbox.
#
#    info_xml=open("/home/"+user+"/Videos/test2.xml", "a+b")
#    info_xml.write(     "<BBoxs>")
#    info_xml.close()    
#    print "start app"
#    try:
#        App(video_src, dataset_fn).run("/home/"+user+"/Videos/test2.xml")
#    except  Exception as er:
#            print " error",  er
##                    info_xml=open(info_file, "a+b")
##                    info_xml.write(     "</BBoxs>")
##                    info_xml.close() 
#    info_xml=open("/home/"+user+"/Videos/test2.xml", "a+b")
#    info_xml.write(     "</BBoxs>")
#    info_xml.close()
    
    ## ottengo una istanza di video
    video_info =None
    try:
            #print dir(isa_resource)
            video_info = Video()
            #print " tracce del video ",  len(video._track_list )
            video_info  =get_bb2("/home/"+user+"/Videos/FacesList.xml")
            #print len(bb_list)
            #video_info.preprocessing(bb_list)
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


def test_make_db():
    
    
    print "test make db"
    sys.path.append("home/felix/Desktop/quoqueisa")
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/att_faces1"
    dir_image ="/home/"+user+"/Pictures/att_test" 
    video_src = "/home/"+user+"/Videos/videodemo.mov"
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 
    db_image ="/home/"+user+"/Pictures/att_faces3" 

    ## ottengo una istanza di video
    video_info =None
    try:
            video_info = Video()
            video_info  =get_bb2("/home/"+user+"/Videos/FacesList.xml")
            print " tracce del video ",  len(video_info._track_list )
    except getopt.error, msg:
            print msg
            print " usage: file_conf.xml"   

    #### croppo immagin
    
    try:
                print "start make db "
                App(video_src, dataset_fn).make_db(video_info,  db_image)
    except  Exception as er:
                    print " error - run crop",  er





def test4():
    
    
    print "test 4"
    
    
    sys.path.append("home/felix/Desktop/quoqueisa")
    user = str(os.environ["USER"])
    dataset_fn = "/home/"+user+"/Pictures/att_faces"
    dir_image ="/home/"+user+"/Pictures/att_test" 
    video_src = "/home/"+user+"/Videos/test2.mpeg"
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 

    ## creo il file dei bbox.

    info_xml=open("/home/"+user+"/Videos/test2.xml", "a+b")
    info_xml.write(     "<BBoxs>")
    info_xml.close()    
    print "start app"
    try:
        App(video_src, dataset_fn).run("/home/"+user+"/Videos/test2.xml")
    except  Exception as er:
            print " error",  er
#                    info_xml=open(info_file, "a+b")
#                    info_xml.write(     "</BBoxs>")
#                    info_xml.close() 
    info_xml=open("/home/"+user+"/Videos/test2.xml", "a+b")
    info_xml.write(     "</BBoxs>")
    info_xml.close()
    
    ## ottengo una istanza di video
    video_info =None
    try:
            #print dir(isa_resource)
            video_info = Video()
            #print " tracce del video ",  len(video._track_list )
            bb_list =get_bb("/home/"+user+"/Videos/test2.xml")
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
def test4image():
    print "test 4image"
    import sys
    user = str(os.environ["USER"])
    print help_message
    info_file=""
    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 
    
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


if __name__ == '__main__':
    START_TIME=time.time()	
    print " VIDEOFACEREC ON QUOQUEISAALL"
    try:
        print "Verifico presenza DB ", str(os.getcwd())+'/Data/db_shelve.db'
	if os.path.exists(str(os.getcwd())+'/Data/db_shelve.db.db') :
	    print " db presente "	
	    os.remove(str(os.getcwd())+'/Data/db_shelve.db.db')
	    print "ELIMINATO DB ", str(os.getcwd())+'/Data/db_shelve.db.db'
	    print os.path.exists(str(os.getcwd())+'/Data/db_shelve.db.db') 	
        else: 
		print " DB non presente. Esiste? ", os.path.exists(str(os.getcwd())+'/Data/db_shelve.db.db') 	

    except Exception, e:
	print e
    #exit()
    dir_base_fr=str(os.getcwd())+"/../"
    print "ROOT DIRECTORY", dir_base_fr	
    dataset_fn=dir_base_fr+"/Parameters/resource/att_faces/"
    nome_video="test2.mpg"
    if len(sys.argv)==2:
    		nome_video=str(sys.argv[1])    
    #video_src=dir_base_fr+"/Video/"+nome_video
    if os.path.exists(dir_base_fr+"/Video/"+nome_video):
	video_src=dir_base_fr+"/Video/"+nome_video
    else :
	if os.path.exists(dir_base_fr+"/"+nome_video):
	    video_src=dir_base_fr+"/"+nome_video	
	else:
	    print "  NO SOURCE FILE"
	    exit()
	
	    
    xml_of_video=dir_base_fr+"/Data/faceList"+nome_video+".xml"
    xml_out=dir_base_fr+"/Data/RecfaceList"+nome_video+".xml"
    yml_out=dir_base_fr+"/Data/"+nome_video+".yml"    
    print "video ", video_src
    print "xml video ", xml_of_video
    print "yml video ", yml_out
    
    #App(video_src, dataset_fn).run()    
    #exit()
    try:
        video_info = Video()

        who=WhoIs(-1)
	
        #who.shelve_db = shelve.open(dir_base_fr+'/Data/db_shelve.db')
        print " xml of video ", xml_of_video
        video_info=get_bb2(xml_of_video)
	
	#App(video_src, dataset_fn).make_db(video_info, "./db_image")
	#exit()	
	
        print " tracce del video ",  len(video_info._track_list ), " numero di frame ", video_info._frame_number
	print "video src", video_src, "dataset_fn ", dataset_fn
        App(video_src, dataset_fn).run_db(video_info, xml_out)
	
        ####App(video_src, dataset_fn).run_1()
        cv2.destroyAllWindows()
        who.run()
        print " who.shelve_db ",   who.shelve_db
        who.save_shelva_xml(video_info, xml_out)
	
        ####print " frame ", video_info._track_list[34]._bbList[0]._frame
        ####App(video_src, dataset_fn).run_1(video_info, video_info._track_list[38]._bbList[0]._frame)
        ####App(video_src, dataset_fn).run_1(video_info, None, dir_base_fr+"/VideoOut/"+nome_video)
        #App(video_src, dataset_fn).run_writevideo(video_info, None, dir_base_fr+"/VideoOut/"+nome_video)
	
	END_TIME=time.time()

	App(video_src, dataset_fn).run_VideoWriter(video_info, dir_base_fr+"/VideoOut/"+nome_video)	
        cv2.destroyAllWindows()
	
	who.save4statistic(video_info, yml_out)
    except getopt.error, msg:
        print "&&&ERROR&&&"
        print msg
        cv2.destroyAllWindows()	
    print "the end"

