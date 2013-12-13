
import getopt
import random
import shelve
import time
import os
import sys
import xml.etree.cElementTree as et
import xml.etree.ElementTree as ET

#def _write_xml():
#    print "mettodo utile soltanto per generare un file xml finto di prova"
#    user = str(os.environ["USER"])
#    print help_message
#    info_test_file=""
#    base_dataset_fn ="/home/"+user+"/Pictures/att_test" 
#    video_src = "/home/"+user+"/Videos/test2.mpeg"            
#    list_s= os.listdir(base_dataset_fn )
#    
#    for i in range( len(list_s)-2):
#            info_test_file  =base_dataset_fn + "/info_file/"+str(i+1) +".xml"
#
def get_bb2(_path):    
	#tree = ET.parse('FacesList.xml')
	tree = ET.parse(_path)
	root = tree.getroot()
	#print root.tag
	id_face=1
	video= Video() 
	id_bb=1
	for child_faces in root:
		#print(child_faces.tag, child_faces.attrib)
		if child_faces.tag == "nrFrames":
			print child_faces.text
			video._frame_number=str(child_faces.text)
		for child_ in child_faces:
			#print(child_.tag, child_.attrib)
			for child_face in child_:
				#print(child_face.tag, child_face.attrib)
				#print "------- creo una nuova  ------------------------------------- Face"
				#print "face id ", id_face
				tr =Track()
				tr._id=id_face
				video._track_list.append(tr)
				id_face=id_face+1
				id_bb=1
				for child_bboxes in child_face:
					for child__ in child_bboxes:
						for child_bb in child__:
							#print "------- creo una nuova  ------------------------------------- BBox"
							bb=Bbox()
							bb._id=id_bb
							bb._track=tr
							id_bb=id_bb+1
							for c in child_bb:
								#print(c.tag, c.text)
								if  c.tag=="frameId":
									bb._frame=float(c.text)
									if tr._frame_start==None:
										tr._frame_start=float(c.text)
									tr._frame_stop=float(c.text)
									#print c.text
								if  c.tag=="x":
									bb._position[0]=c.text
									#print c.text
								if  c.tag=="y":
									bb._position[1]=c.text
									#print c.text
								if  c.tag=="h":
									bb._position[3]=c.text
									#print c.text
								if  c.tag=="w":
									bb._position[2]=c.text
									#print c.text                                            
							tr._bbList.append(bb)

	#print "numero facce ", len(root[0].findall('_'))
	return video


def get_bb(_path ):
	fl=open(_path, "r+")
	s2xml=fl.read()
	tree=et.fromstring(s2xml)
	bb_list=[]

	for el in tree.findall('BBox'):
		bb= Bbox()
		for ch in el.getchildren():
			#print '{:>15}: {:<30}'.format(ch.tag, ch.text) 
			if ch.tag == "time" :
				bb._time=ch.text
			if ch.tag == "x" :
				bb._position[0]=ch.text
			if ch.tag == "y" :
				bb._position[1]=ch.text    
			if ch.tag == "w" :
				bb._position[2]=ch.text             
			if ch.tag == "h" :
				bb._position[3]=ch.text             
			if ch.tag == "frame" :
				bb._frame=float(ch.text)
			if ch.tag == "confidence" :
				bb.confidence=float(ch.text)
			if ch.tag == "label" :
				bb._label=ch.text                                
				#print ch.text
		bb_list.append(bb)
	return bb_list

class Video():
	def __init__(self):
		self._track_list = []
		self._video_id=str(time.time())
		#eventuali bbox senza traccia
		self._untrack =None
		self._bb_by_frame=None
		self._bb_list=[]
		self._frame_number=None
	def get_bb_by_frame(self):
		if self._bb_by_frame ==None:
			self._bb_by_frame={}
			for b in self._bb_list :
				if self._bb_by_frame.has_key(b._frame) :
					self._bb_by_frame[b._frame].append(b)
				else:
					self._bb_by_frame[b._frame]=[b]
		return self._bb_by_frame               
	def preprocessing(self,  bblist):
		bbl =bblist
		for b in bbl:
			if self._track_list :
				tr =self._track_list[-1]
				#print int(b._frame)
				#print tr._frame_stop , "<",   int( b._frame )
				if tr._frame_stop+5> b._frame :
					#print "Aggiungo alla traccia ",  tr
					tr._bbList.append(b)
					tr._frame_stop=b._frame
				else:    
					#print "Creo una nuova traccia"
					tr =Track()
					tr._frame_start=b._frame                        
					tr._frame_stop=b._frame
					tr._bbList.append(b)
					self._track_list.append(tr)     
					#print " traccia numero ", len(self._track_list)
					#print "fine traccia ",      tr._frame_stop
			else:
				#print "Creo prima traccia"
				tr =Track()
				tr._frame_start=b._frame                        
				tr._bbList.append(b)
				tr._frame_stop=b._frame
				self._track_list.append(tr)
				#print self._track_list[0]._frame_stop
	def compute(self):    
		print "compute"
	def serialize(self):
		s = shelve.open('video.db')
		try:
			s[self._video_id] =self 
		finally:
			s.close()

class Track:
	def __init__(self):
		self._id=None
		self._person=None
		self._start=None
		self._stop=None
		self._frame_start=None
		self._frame_stop=None        
		self._bbList=[]

		# _status o --
		# _status 1
		# _status 2
		self._status=0
	def  statistic(self):
		plist={}
		for bb in self._bbList :
			if bb._person:
				if plist.has_key(bb._person) :
					plist[bb._person] =[ plist[bb._person][0]+1,  min(bb._confidence, plist[bb._person][1] ),  max(bb._confidence, plist[bb._person][2] ),  (plist[bb._person][3] + bb._confidence) /(plist[bb._person][0]+1)  ]
				else:
					plist[bb._person] =[ 1,  bb._confidence,  bb._confidence, bb._confidence*1.0 ]
				#print  "name " + bb._person +"  statistic "  
				#print plist[bb._person]  
		return plist

	def compute(self,  plist=None):
		if plist==None:
			plist = statistic()
		p_tot = len(plist.keys())
		#print "p_tot ",  p_tot
		for p in plist.iteritems():
			#print int(p[1][0]) 
			if int(p[1][0])>p_tot*0.85 :
				print ".85 detect person ",  p[0]
				self._person=p[0]
				return
			if int(p[1][0])>p_tot*0.5 :
				#print ".5 detect person %s ",  p[0]
				self._person=p[0][0]
				return





class Bbox:
	def __init__(self):
		self._id=None
		self._person=None
		self.confidence=100000
		self._position=[-1, -1, -1, -1]
		self._time=None
		self._track=None
		self._frame=None
		self._label=None
		# _status o --
		# _status 1
		# _status 2
		self._status=0
		self._auto_id=str(random.choice("aasdfghjklpoiuytrew")) +"--"+str(random.random())+"--"+str(random.choice("aasdfghjklpoiuytrew"))
	def __repr__(self):
		return "Bbox : " +self._auto_id +"-"+ str(self._id)+"-"+  str(self._person)+"-"+ str(self._position)+"-"+  str(self._time)+"-"+  str(self._track)
#    def __repr__(self):
#        return "Bbox:" +self._auto_id 
class Person():
	def __init__(self):
		self._id=None
		self._bbList=[]
		self._trackList=[]    
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
			#print " valore medio ",self.name_tr, " ",   str(self.val_atteso[0]/self.val_atteso[1])
#        else:
#            print "extra ",  self.name_tr
	def print_value(self):
		try:
			print " valore medio ",self.name_tr, " ",   str(self.val_atteso[0]/self.val_atteso[1])
		except:
			print "error in values print_value"        
	def get_value(self):
		try:
			return  self.val_atteso[0]/self.val_atteso[1]
		except:
			print "error in values print_value"
			return -1

def Test7():
	print "test7"
	try:
		video = Video()
		#print " tracce del video ",  len(video._track_list )
		bb_list =get_bb("/home/felix/Documents/test.xml" )
		print len(bb_list)
		video._bb_list=bb_list
		print video.get_bb_by_frame()

		print len(video._bb_by_frame)
		for bf in video._bb_by_frame:
			print bf,  
	except getopt.error, msg:
		print msg
		print " usage: file_conf.xml"   


#class Test1:
#    bb=Bbox()
#    bb._person="/s51"
#    bb._position=[0, 0, 10, 10]
#    bb._time=100
#    print bb
#
#
#class Test2():
#    print "un video, una traccia, molti bbox. Quasi tutti riferiscono alla stessa persona"
#    tr=Track()
#    for i in range(20):
#            bb=Bbox()
#            bb._person="/s"+str(i)
#            bb._position=[i, i, i+10, i+10]
#            bb._confidence=1
#            bb._time=100+i
#            tr._bbList.append(bb)
#            
#         # sempre la stessa persona   
#    for i in range(20):
#            bb=Bbox()
#            bb._person="/s1"
#            bb._position=[i, i, i+10, i+10]
#            bb._confidence=i+1
#            bb._time=100+i
#            tr._bbList.append(bb)       
#    st = tr.statistic()
#    tr.compute(st)
#    print "The person is: ", tr._person   

def Test3():
	print "test3"
	try:
		video = Video()
		#print " tracce del video ",  len(video._track_list )
		bb_list =get_bb("/home/felix/Desktop/facerec-master/py/apps/videofacerec/test1.xml" )
		print len(bb_list)
		video.preprocessing(bb_list)
		print " tracce del video ",  len(video._track_list )
		#print video.serialize()
		value_list=[]
		for i in range (len(video._track_list ) ) :
			tr=video._track_list [i]
			v_tmp=Values(tr._frame_start,  tr._frame_stop,  "traccia"+str(i))
			for b in tr._bbList:
				v_tmp.run(b._frame,  b.confidence)
			value_list.append(v_tmp)

		ref_val = value_list[0].get_value()
		for v in value_list:
			val_tmp=v.get_value()
			#print v.name_tr,  "  media ", str(val_tmp)  

		#print "\n \n \n"    


		print "Valore di riferimento " +str(ref_val)
		for v in value_list:
			val_tmp=v.get_value()
			print v.name_tr,  "  media ", str(val_tmp)  

			if ref_val +ref_val/10.0 > val_tmp:
				print " valori simili, stesse persone "
			if ref_val > val_tmp:
				print " valori simili, stesse persone "
			print "\n \n"    
	except getopt.error, msg:
		print msg
		print " usage: file_conf.xml"   

def Test6():
	print "test6"
	try:
		base_dir="/home/felix/Pictures/att_test/info_file/"
		list_file_info= os.listdir(base_dir)
		print "list_file_info ",  list_file_info
		tr_index=-1
		for f in list_file_info:
			print "----------------------------------------"
			result=[]            
			video = Video()
			print f
			tr_index = int(f.split(".")[0])-1
			#print " tracce del video ",  len(video._track_list )
			bb_list =get_bb(base_dir+f)
			#print len(bb_list)
			video.preprocessing(bb_list)
			print " tracce del video ",  len(video._track_list )
			#print video.serialize()
			for i in range (len(video._track_list ) ) :
				tr=video._track_list [i]
				#print " taccia ",  i+1,  "  frame iniziale=",  tr._frame_start,  " frame finale",  tr._frame_stop
			value_list=[]
			for i in range (len(video._track_list ) ) :
				tr=video._track_list [i]
				v_tmp=Values(tr._frame_start,  tr._frame_stop,  "traccia"+str(i+1 ))
				for b in tr._bbList:
					v_tmp.run(b._frame,  b.confidence)
				value_list.append(v_tmp)
			#print " valori in lista ",  len(value_list)   
			ref_val = value_list[ tr_index].get_value()
#                    for v in value_list:
#                        print v.get_value()
			#print "Valore di riferimento " +str(ref_val) 
			tmp=0
			for v in value_list:
				val_tmp=v.get_value()
				print tmp+1,  "  media ", str(val_tmp)  ,  "   == ",ref_val / val_tmp 
				if ref_val / val_tmp > 0.5:
					#print " ----traccia " +str(tr_index+1) +" e traccia "+str(tmp+1) +"  appartengono alla stessa persona"
					result.append(1)
				if ref_val / val_tmp < 0.5:
					#print " ++++traccia " +str(tr_index+1) +" e traccia "+str(tmp+1) +" NON appartengono alla stessa persona"
					result.append(0)
				tmp=tmp+1
			print result   
	except getopt.error, msg:
		print msg
		print " usage: file_conf.xml"               
#class Test4():
#        try:
#            bb_list =get_bb("/home/felix/Desktop/facerec-master/py/apps/videofacerec/test_info.xml" )
#            print len(bb_list)
#            tr= Track()
#            tr._id=1
#            for bb in bb_list:
#                bb._person="s1"
#                bb._confidence = random.uniform(1, 100)
#                bb._track=tr
#                tr._bbList.append(bb)
#            st = tr.statistic()
#            tr.compute(st)
#            
#            print "The person is: ", tr._person   
#        except getopt.error, msg:
#            #read_conf("fr_cc.xml")
#            print " usage: file_conf.xml"    
#            
#class Test5():
#        s = shelve.open('test_shelf.db')    
#        try:
#
#            bb_list =get_bb("/home/felix/Desktop/facerec-master/py/apps/videofacerec/test_info.xml" )
#            print len(bb_list)
#            tr= Track()
#            tr._id=1
#            for bb in bb_list:
#                bb._person="s1"
#                bb._confidence = random.uniform(1, 100)
#                bb._track=tr
#                tr._bbList.append(bb)
#            st = tr.statistic()
#            tr.compute(st)
#            s["id"+str(tr._id)]=tr
#            print "The person is: ", tr._person   
#        except getopt.error, msg:
#            #read_conf("fr_cc.xml")
#            print " usage: file_conf.xml"    
#
#        finally:
#            s.close()
#            
#class Test5_bis():
#        s = shelve.open('test_shelf.db')    
#        try:
#            print "______________Test5_bis___________"
#            print len(s.keys())
#            print s["id1"]
#            print "id ",  s["id1"]._id            
#            print "bbList ",  len(s["id1"]._bbList)
#            print "bbList ",  s["id1"]._bbList[23]._position
#
#            
#        except getopt.error, msg:
#            #read_conf("fr_cc.xml")
#            print " usage: file_conf.xml"    
#
#        finally:
#            s.close()
"""
class Test8():
	
	print "start test 6"
	pt="/home/felix/Videos/FacesList.xml"
	video= get_bb2(pt)
	print " numero tracce",  len(video._track_list)
	for tr in video._track_list:
		print "-----------------------------"
		print "numero bb della traccia ",tr._id,   len(tr._bbList),  " frame iniziale ",  tr._frame_start ,  " frame finale ",  tr._frame_stop
#        for bb in tr._bbList:
#            print "--"
#            print bb._frame
#            print bb._position

def Test9():
	print "test 9"
	try:
		base_dir="/home/felix/Pictures/att_test/info_file/"
		list_file_info= os.listdir(base_dir)
		print "list_file_info ",  list_file_info
		tr_index=-1
		video = get_bb2("/home/felix/Videos/FacesList.xml")
		for f in list_file_info:
			print "----------------------------------------"
			result=[]            
			video = Video()
			print f
			tr_index = int(f.split(".")[0])-1
			print "tr_index ",  tr_index
			#print " tracce del video ",  len(video._track_list )
			bb_list =get_bb(base_dir+f)
			#print len(bb_list)
			video.preprocessing(bb_list)
			#print " tracce del video ",  len(video._track_list )
			#print video.serialize()
			for i in range (len(video._track_list ) ) :
				tr=video._track_list [i]
				#print " taccia ",  i+1,  "  frame iniziale=",  tr._frame_start,  " frame finale",  tr._frame_stop
			value_list=[]
			for i in range (len(video._track_list ) ) :
				tr=video._track_list [i]
				v_tmp=Values(tr._frame_start,  tr._frame_stop,  "traccia"+str(i+1 ))
				for b in tr._bbList:
					v_tmp.run(b._frame,  b.confidence)
				value_list.append(v_tmp)
			#print " valori in lista ",  len(value_list)   
			ref_val = value_list[ tr_index].get_value()
#                    for v in value_list:
#                        print v.get_value()
			#print "Valore di riferimento " +str(ref_val) 
			tmp=0
			for v in value_list:
				val_tmp=v.get_value()
				print tmp+1,  "  media ", str(val_tmp)  ,  "   == ",ref_val / val_tmp 
				if ref_val / val_tmp > 0.5:
					#print " ----traccia " +str(tr_index+1) +" e traccia "+str(tmp+1) +"  appartengono alla stessa persona"
					result.append(1)
				if ref_val / val_tmp < 0.5:
					#print " ++++traccia " +str(tr_index+1) +" e traccia "+str(tmp+1) +" NON appartengono alla stessa persona"
					result.append(0)
				tmp=tmp+1
				print "================================================="
			print result   
	except getopt.error, msg:
		print msg
		print " usage: file_conf.xml"               

def Test_save_as_xml():
	try:
		file_xml=open("/home/felix/Pictures/RecFaceList.xml", "w+")
		video_as_xml='<?xml version="1.0"?><Faces>'
		base_dir="/home/felix/Pictures/att_test/info_file/"
		list_file_info= os.listdir(base_dir)
		print "list_file_info ",  list_file_info
		tr_index=-1
		video = get_bb2("/home/felix/Videos/FacesList.xml")
		tr_list= video._track_list
		face={"1":[["s1", 1]], "2":[["s2", 1]],"3":[["s3", 0.5], ["s1", 0.5]] , "4":[["s2", 1]], "5":[["s5", 1]], "6":[["s2", 0.8],["s6", 0.2] ] , "7":[["s1", 1]], "8":[["s8", 1]], "9":[["s8", 1]],"10":[["s2", 1]],"11":[["s2", 0.8],["s11", 0.2] ], "12":[["s12", 1]], "13":[["s13", 1]], "14":[["s14", 1]], "15":[["s15", 1]], "16":[["s15", 1]], "17":[["s15", 1]], "18":[["s18", 1]], "19":[["s19", 1]], "20":[["s20", 1]], "22":[["s2", 1]], "23":[["s21", 1]], "13":[["s24", 1]]}
		for tr in tr_list:
			print " taccia ",  tr._id,  "  frame iniziale=",  tr._frame_start,  " frame finale",  tr._frame_stop
			video_as_xml=video_as_xml+"<Face><face_id>"+str(tr._id)+"</face_id><start_frame>"+str(tr._frame_start)+"</start_frame><stop_frame>"+str(tr._frame_stop)+"</stop_frame>"+"<person><name></name><confidence></confidence></person><person><name></name><confidence></confidence></person></Face>"
		video_as_xml=video_as_xml+"</Faces>"    
		file_xml.write(video_as_xml)
		file_xml.close()    
	except getopt.error, msg:
		print msg
		print " usage: file_conf.xml" 
"""
if __name__ == '__main__':    
		#Test1()
		#Test2()
		#Test6()
		#Test4()    
#    Test5()    
#    Test5_bis()
		#Test7()
		#Test8()
		#Test9()
	#Test_save_as_xml()
	_path="/Users/labcontenuti/Documents/Demo/Data/Prelim.mpg.xml"
	video = get_bb2(_path)
	print video._frame_number