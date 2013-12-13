import shutil
import os
import random




class DBManager():
        def __init__(self,dir_base_fr="/Users/labcontenuti/Documents/FaceRecognition/resource",training="/training",dir_image="/att_faces4/" ):
                
                self.dir_base_fr=dir_base_fr
                self.training= dir_base_fr+training
                self.dir_image =dir_base_fr+dir_image
        def prepare(self):
                #shutil.rmtree(self.training)
                ldir= os.listdir(self.dir_base_fr)
                print self.training, self.training+"_"+str(len(ldir))
                os.rename(self.training, self.training+"_"+str(len(ldir)))
                os.makedirs( self.training) 
                #os.rename(self.dir_image, self.dir_image+"_"+str(len(ldir)))
                os.makedirs( self.dir_image)                 
        def make_training_db(self):  
                if not self.dir_image.endswith("/"):
                        self.dir_image=self.dir_image+"/"
                list_s= os.listdir(self.dir_image)
                for l in list_s:
                        print l
                        all_image = os.listdir(self.dir_image+l+"/s"+l)
                        #print len(all_image)
                        for i in range(10):
                                im_name= all_image[random.randint(0, len(all_image) -1)]
                                if not os.path.exists(  self.training+"/"+str (l)  ):
                                        os.makedirs( self.training+"/"+str (l) )
                                if not os.path.exists(  self.training+"/"+str (l) +"/s"+str (l) ):
                                        os.makedirs( self.training+"/"+str (l) +"/s"+str (l) ) 
                                shutil.copy(self.dir_image+"/"+str (l) +"/s"+str (l)+"/"+str(im_name),  self.training+"/"+str (l) +"/s"+str (l)+"/"+str(im_name) )   

        def ceck_db(self):
                list_dir=os.listdir(self.dir_image)
                list_dir_int=[]
                for l in list_dir:
                        list_dir_int.append(int(l))
                list_dir_int.sort()
                for l in list_dir_int:
                        print l
                        all_image=os.listdir(self.dir_image+"/"+str(l)+"/s"+str(l))
                        all_image_int=[]
                        for a in all_image:
                                all_image_int.append(int(a[:-4]))
                        all_image_int.sort()        
                        print "cartella ", l, "  numero immagini ", len(all_image_int)        
                        if len(all_image_int) < 20:
                                print "rimuovo cartella ", self.dir_image+"/"+str (l) , " e cartella ",  self.training+"/"+str (l)
                                # rimuovo la cartella da attface4 e da training
                                shutil.rmtree(self.dir_image+"/"+str (l) )
                                shutil.rmtree(self.training+"/"+str (l) )
                        if len(all_image_int)>=20:
                                for i in range(1,6):
                                        #shutil.rmtree(self.dir_image+"/"+str (l) +"/s"+str (l)+"/"+str(all_image_int[-i])+".pgm")
                                        os.remove(self.dir_image+"/"+str (l) +"/s"+str (l)+"/"+str(all_image_int[-i])+".pgm")
if __name__ == "__main__":
        dir_base_fr="/Users/labcontenuti/Documents/FaceRecognition/resource"
        ldir= os.listdir(dir_base_fr)
        training="/training"
        dir_image="/att_faces4"
        db=DBManager(dir_base_fr,training,dir_image)
        db.prepare()
        db.make_training_db()
        db.ceck_db()
        print "ok"