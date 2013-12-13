

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
	all_image = os.listdir(base_directory+"db/")

	for img_name in all_image:
#        try:
		print "img_name ",  img_name  
		min_val=1000
		if not os.path.exists(base_directory+"/resplit/s1"):
			os.makedirs(base_directory+"/resplit/s1") 
			#print " creo directory" 
		#sposto questa immagine e la uso come db
		shutil.move(base_directory+"/db/s1/"+img_name, base_directory+"/resplit/s1")

		for i in range(10):
#                if not os.path.exists(base_directory+"/db/s1"):
#                        os.makedirs(base_directory+"/db/s1")                  
			#print  i,  "---------------------"
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
					if val_min[4]>p_confidence2:
						val_min.remove(val_min[4])
						val_min.append(p_confidence2)
						val_min.sort()
						min_val =val_min[4]
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
		if len(rec_image)>=20 :
			print " COPIOOOOOOOO ",base_directory+"/db/s1/"+rec_image[0]  
			shutil.copy(base_directory+"/db/s1/"+rec_image[0], base_directory+"/rec_image/")
		for img in rec_image:
			shutil.remove(base_directory+"/db/s1/"+img)
#              else:
#                  for f in rec_image :
#                                          shutil.copy(base_directory+"/db/s1/"+f, base_directory+"/mix_image/")

		shutil.rmtree(base_directory+"db/s1")    
	print "---------------------------------"
	print "========FINE   RESPLIT ======================"





if __name__ == "__main__":
	print "start multi image recognition"
	base_directory="/home/felix/Pictures/multi_image/"
	labels={}
	out_dir = None
	all_image = os.listdir(base_directory+"mix_image")

	for img_name in all_image:
#        try:
		print "img_name ",  img_name  
		min_val=1000
		if not os.path.exists(base_directory+"/db/s1"):
			os.makedirs(base_directory+"/db/s1") 
			#print " creo directory" 
		#sposto questa immagine e la uso come db
		shutil.move(base_directory+"/mix_image/"+img_name, base_directory+"/db/s1")
		#shutil.copy(base_directory+"/mix_image/"+img_name, base_directory+"/db/s1")

		for i in range(10):
#                if not os.path.exists(base_directory+"/db/s1"):
#                        os.makedirs(base_directory+"/db/s1")                  
			#print  i,  "---------------------"
			[X,y] = read_images(   base_directory+"/db/", (92,112))
			model = cv2.createLBPHFaceRecognizer()
			#print "model ",  model 
			#model = cv2.createEigenFaceRecognizer()
			model.train(np.asarray(X), np.asarray(y))        
			#model.load(base_directory+"model")      
			font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1,1,0,3,8)

			other_image = os.listdir(base_directory+"mix_image")
			#print "other_image ",  other_image
			if i==0:
				val_min=[1000, 1001, 1002, 1003, 1004]
			for other_i in  other_image :
				#print "read_1_image ",  base_directory+"/mix_image/"+other_i
				[X2,y2] = read_1_image(base_directory+"/mix_image/"+other_i, (92,112))        
				[p_label2, p_confidence2] = model.predict(np.asarray(X2[0]))
				#print  "image ",  labels[p_label2], " confidence ",str(p_confidence2) 
				#print "------------------",   i
				if i==0:
					if val_min[4]>p_confidence2:
						val_min.remove(val_min[4])
						val_min.append(p_confidence2)
						val_min.sort()
						min_val =val_min[4]
						print " minore quindi appendo ",  p_confidence2
					print val_min                                                                                                            
#                        if     p_confidence2 < min_val :
#                            if p_confidence2> 1:
#                                min_val = p_confidence2
				#print "min_val ",  min_val,  "   lista ", val_min
				if i >0:
					#print base_directory+"/mix_image/"+other_i,  "  ",  p_confidence2      
#                      try:
					if  p_confidence2 < min_val+1 :
						#shutil.copy(base_directory+"/mix_image/"+other_i, base_directory+"/db/s1")
						shutil.move(base_directory+"/mix_image/"+other_i, base_directory+"/db/s1")
						#print " move image ",base_directory+"/mix_image/"+other_i  
						try:
							all_image.remove(other_i)
						except:
							pass
#                      except:
#                        print " shutil move ",base_directory+"/mix_image/"+other_i    

		rec_image = os.listdir(base_directory+"/db/s1")
		print "-----------------rec_image ",  rec_image
		if len(rec_image)>=20 :
			print " COPIOOOOOOOO ",base_directory+"/db/s1/"+rec_image[0]  
			shutil.copy(base_directory+"/db/s1/"+rec_image[0], base_directory+"/rec_image/")
#              else:
#                  for f in rec_image :
#                                          shutil.copy(base_directory+"/db/s1/"+f, base_directory+"/mix_image/")

		shutil.rmtree(base_directory+"db/s1")    
	print "---------------------------------"
	print "========FINE======================"

