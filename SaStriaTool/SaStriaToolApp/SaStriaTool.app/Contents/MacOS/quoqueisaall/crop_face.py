import numpy as np
import sys, math, Image
import cv2,  os,  cv
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
    

if __name__ == "__main__":

#  image =  Image.open("img/mind1176.bmp")
#  image11= image.crop((574,84,574+92,84+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/1.pgm")
#
#  xc=571
#  yc=88
#  num=2
#  image =  Image.open("img/mind1177.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/"+str(num)+".pgm")
#  
#  
#  xc=568
#  yc=87
#  num=3
#  image =  Image.open("img/mind1178.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/"+str(num)+".pgm")
#  
#    
#  xc=580
#  yc=91
#  num=4
#  image =  Image.open("img/mind1179.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/"+str(num)+".pgm")
#  
#  xc=575
#  yc=98
#  num=5
#  image =  Image.open("img/mind1180.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/"+str(num)+".pgm")
#  
#  
#  
#  xc=884
#  yc=124
#  num=6
#  image =  Image.open("img/mind1224.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/"+str(num)+".pgm")
#  
#  
#  xc=805
#  yc=143
#  num=7
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/"+str(num)+".pgm")

  
#  xc=805
#  yc=143
#  num=8
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+59,yc+118))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/"+str(num)+".pgm")
#  
  
#  xc=805
#  yc=143
#  num=9
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+53,yc+110))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/"+str(num)+".pgm")
#  
#  
#  
#    
#  xc=787
#  yc=143
#  num=10
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/"+str(num)+".pgm")
#  
#  
#  xc=766
#  yc=142
#  num=11
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+92,yc+112))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/"+str(num)+".pgm")
#  ss= SizeSuggest((0, 0, 92, 112)) 
#  print "(0, 0, 92, 112)  ",ss,  "  ",  float(92)/112
#  print "---"  
#  ss= SizeSuggest((0, 0, 92, 200)) 
#  print "(0, 0, 92, 200)  ", ss,  "  ",  ss[2]/ss[3]
#  print "---"  
#  ss= SizeSuggest((0, 0, 92, 300)) 
#  print "(0, 0, 92, 300)  ", ss,  "  ",  ss[2]/ss[3]  
#  print "---"  
#  ss= SizeSuggest((0, 0, 200, 112)) 
#  print "(0, 0, 200, 112)  ",ss,  "  ",  ss[2]/ss[3]
#  print "---"  
#  ss= SizeSuggest((0, 0, 9, 112)) 
#  print "(0, 0, 9, 112)  ", ss,  "  ",  ss[2]/ss[3]
#  print "---"  
#  ss= SizeSuggest((0, 0, 200, 200)) 
#  print "(0, 0, 200, 112)  ",ss,  "  ",  ss[2]/ss[3]
#  xc=766
#  yc=142  
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+int(ss[2]),yc+int(ss[3]) ))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/provacrop.pgm")
#  
#  print "---"  
#  ss= SizeSuggest((0, 0, 59, 118))   
#  xc=805
#  yc=143
#  num=81
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+int(ss[2]),yc+int(ss[3]) ))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/"+str(num)+".pgm")
#  
#  
#  print "--- "  
#  ss= SizeSuggest((0, 0, 92, 112))   
#  print "(0, 0, 92, 112)  ",ss,  "  ",  ss[2]/ss[3]
#  xc=805
#  yc=143
#  num=71
#  image =  Image.open("img/mind1209.bmp")
#  image11= image.crop((xc,yc,xc+int(ss[2]),yc+int(ss[3]) ))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_test/sa/"+str(num)+".pgm")  

#  xc=82
#  yc=29
#  image =  Image.open("img/paolo_bitta.png")
#  image11= image.crop((xc,yc,xc+46,yc+56))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("img_crop/bitta.pgm")


#  image =  Image.open("faces_tmp/face_50.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s42/1.pgm")
#  
#  
#  image =  Image.open("faces_tmp/face_54.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s42/2.pgm")
#  
#
#  image =  Image.open("faces_tmp/face_62.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s42/3.pgm")  
#  
#  
#  image =  Image.open("/home/felix/Pictures/luca_nervi.jpg")
#  image11= image.crop((18,18,73,104))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s43/1.pgm")
#  
#  
#  image =  Image.open("/home/felix/Pictures/luca_nervi2.jpg")
#  image11= image.crop((33,32,115,171))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s43/2.pgm")
#  
#
#  
#  image =  Image.open("/home/felix/Pictures/luca_nervi3.jpg")
#  image11= image.crop((8,8,46,62))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s43/3.pgm")  
  
#  image =  Image.open("faces_tmp/face_32.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s43/4.pgm")
#  
#  
#  image =  Image.open("faces_tmp/face_37.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s43/5.pgm")
#  
#
#  image =  Image.open("faces_tmp/face_34.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s43/6.pgm") 
#  
#  
#  
#  
#  image =  Image.open("faces_tmp/face_10.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s42/4.pgm")
#  
#  
#  image =  Image.open("faces_tmp/face_22.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s42/5.pgm")
#  
#
#  image =  Image.open("faces_tmp/face_39.png")
#  image11 = image.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s42/6.pgm") 

#  xc=55
#  yc=31
#  num=1
#  image =  Image.open("/home/felix/Pictures/eva_mendes/1.jpeg")
#  image11= image.crop((xc,yc,142,156))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s44/"+str(num)+".pgm")
#  
#  xc=32
#  yc=23
#  num=2
#  image =  Image.open("/home/felix/Pictures/eva_mendes/2.jpeg")
#  image11= image.crop((xc,yc,85,70))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s44/"+str(num)+".pgm")  
#  
#  xc=210
#  yc=139
#  num=3
#  image =  Image.open("/home/felix/Pictures/eva_mendes/3.jpeg")
#  image11= image.crop((xc,yc,597,514))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s44/"+str(num)+".pgm")  
#
#  xc=196
#  yc=171
#  num=4
#  image =  Image.open("/home/felix/Pictures/eva_mendes/4.jpeg")
#  image11= image.crop((xc,yc,390,405))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s44/"+str(num)+".pgm")  
#  
#  
#  
#  xc=330
#  yc=126
#  num=5
#  image =  Image.open("/home/felix/Pictures/eva_mendes/5.jpeg")
#  image11= image.crop((xc,yc,484,284))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s44/"+str(num)+".pgm")   
#  
#  
#   
#  xc=102
#  yc=57
#  num=6crop_face.py
#  image =  Image.open("/home/felix/Pictures/eva_mendes/6.jpeg")
#  image11= image.crop((xc,yc,178,144))
#  image11 = image11.resize((92,112), Image.ANTIALIAS)
#  image11.save("att_faces/s44/"+str(num)+".pgm")   

  #lst=[12, 14, 15, 16, 41]
  lst=range(34, 39)
  i=9
  for l in lst:
      i=i+1
      #image = cv2.imread(os.path.join("faces_tmp/face_"+str(l)+".png"), cv2.IMREAD_GRAYSCALE)
      image = cv2.imread(os.path.join("faces_tmp/face_"+str(l)+".pgm"), cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image, (92,112) )
      #cv.EqualizeHist( cv.fromarray(image) , cv.fromarray(image))
      cv2.imwrite("att_faces/s51/"+str(i)+".pgm",  image)
      #image =  Image.open("faces_tmp/face_"+str(l)+".png", cv2.IMREAD_GRAYSCALE)
      #image11 = image.resize((92,112), Image.ANTIALIAS)
      i#mage11.save("att_faces/s46/"+str(i)+".pgm")

  print "croped"
  #CropFace(image, eye_left=(280,322), eye_right=(435,395), offset_pct=(0.3,0.3), dest_sz=(200,200)).save("arnie_10_10_200_200.jpg")
  #CropFace(image, eye_left=(252,364), eye_right=(420,366), offset_pct=(0.2,0.2), dest_sz=(200,200)).save("arnie_20_20_200_200.jpg")
  #CropFace(image, eye_left=(252,364), eye_right=(420,366), offset_pct=(0.3,0.3), dest_sz=(200,200)).save("arnie_30_30_200_200.jpg")
  #CropFace(image, eye_left=(252,364), eye_right=(420,366), offset_pct=(0.2,0.2)).save("arnie_20_20_70_70.jpg")
  #CropFace(image, eye_left=(400,400), eye_right=(420,366), offset_pct=(0.2,0.2)).save("arnie_400_400_70_70.jpg")
