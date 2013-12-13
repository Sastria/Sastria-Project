import sys
import shelve
import time
import getopt
import xml.etree.cElementTree as et


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
    def get_value(self):
        try:
                     return  self.val_atteso[0]/self.val_atteso[1]
        except:
            print "error in values print_value"
            return -1

def read_frames(_path ):
            fl=open(_path, "r+")
            s2xml=fl.read()
            tree=et.fromstring(s2xml)

            for el in tree.findall('Frame'):
                print '-------------------'
                for ch in el.getchildren():
                    #print '{:>15}: {:<30}'.format(ch.tag, ch.text) 
                    if ch.tag == "time" :
                        print ch.text
                    if ch.tag == "frame" :
                        print ch.text        

def read_conf(_path ):
            fl=open(_path, "r+")
            s2xml=fl.read()
            tree=et.fromstring(s2xml)
            for el in tree.findall('Frames'):
                print '--------FRAMES-----------'
                for ch in el.getchildren():
                            if ch.tag == "_" :
                                for el2 in el.findall('_'):
                                    print '--------___________-----------'
                                    for el3 in el2.findall('Frame'):
                                        print '-------------------'
                                        for ch in el3.getchildren():
                                            if ch.tag == "frameId" :
                                                print ch.text        
            for el in tree.findall('BBoxes'):
                print '-------------------'
                for ch in el.getchildren():
                    #print '{:>15}: {:<30}'.format(ch.tag, ch.text) 
                    if ch.tag == "time" :
                        print ch.text
                    if ch.tag == "frame" :
                        print ch.text        
                    if ch.tag == "bboxId" :
                        print ch.text                                
if __name__ == "__main__":
        
#    try:
#            opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
#            print args
#            myhope, confidence,  confidence_max, lowThreshold,max_lowThreshold, ratioTh , kernel_size,  video_path,  db_path  = read_conf(args[0],  myhope, confidence,  confidence_max, lowThreshold,max_lowThreshold, ratioTh , kernel_size,  video_path,  db_path )
#            #read_conf(args[0])
#    except getopt.error, msg:
#            #read_conf("fr_cc.xml")
#            print " usage: file_conf.xml"       
    print "read_conf"       
    try:
            #read_conf("/home/felix/Desktop/facerec-master/py/apps/videofacerec/facere-big-face_info.xml" )
            read_conf("/home/felix/Downloads/data_alex.xml" )
    except getopt.error, msg:
            #read_conf("fr_cc.xml")
            print " usage: file_conf.xml"      
    print "end"
#            sxml="""
#            <encspot>
#              <file>
#               <Name>some filename.mp3</Name>
#               <Encoder>Gogo (after 3.0)</Encoder>
#               <Bitrate>131</Bitrate>
#              </file>
#              <file>
#               <Name>another filename.mp3</Name>
#               <Encoder>iTunes</Encoder>
#               <Bitrate>128</Bitrate>  
#              </file>
#            </encspot>
#            """
#            
#            s2xml=""" 
#            <BBoxs>
#    <BBox>
#        <time>9136</time> 
#        <id>9136</id> 
#        <x>9136</x>
#        <y>9136</y>
#        <w>9136</w>
#        <h>9136</h>
#        <text>9136</text>
#     </BBox> 
#    <BBox>
#        <time>222</time> 
#        <id>222</id> 
#        <x>222</x>
#        <y>222</y>
#        <w>222</w>
#        <h>222</h>
#        <text>222</text>
#     </BBox>      
#</BBoxs>             
#            """
#            tree=et.fromstring(s2xml)
#
#            for el in tree.findall('BBox'):
#                print '-------------------'
#                for ch in el.getchildren():
#                    print '{:>15}: {:<30}'.format(ch.tag, ch.text) 
#
##            print "\nan alternate way:"  
##            el=tree.find('BBox[2]/text')  # xpath
##            print '{:>15}: {:<30}'.format(el.tag, el.text)          
