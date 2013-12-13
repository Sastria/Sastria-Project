/*
 * DetectionToRecognition.cpp
 *
 *  Created on: 22/mag/2013
 *      Author: Alessandro Romanino
 */

#include "DetectionToRecognition.h"


/*
FileStorage fsR(".././Parameters/test.xml", FileStorage::READ);
	cout << "Letto";
	cout<< endl;


	FileStorage fs(".././Parameters/test.xml", FileStorage::WRITE);

	    fs << "Frames" << "[";
	    for( int i = 0; i < 3; i++ )
	    {
	        int x;
	        int y;
	        int w;
	        int h;
	        fs  << "{" << "Frame" << "{" ;
	         	 	 	 	 fs << "idFrame" << i;
	         	 	 	 	 fs << "millsecFrame" << 4;
	         	 	 	 	 fs << "detectionTime" << 5;
	         	 	 	 	 fs << "BBoxes" << "[";
	         	 	 	 	for( int b = 0; b < 2; b++ ){
	         	 	 	 	     x = rand() % 640;
	         	 	 	 		 y = rand() % 480;
	         	 	 	 		 w = rand() % 640;
	         	 	 	 		 h = rand() % 480;
	         	 	 	 		 fs << "{" << "BBox" << "{" << "idBox" << b << "x" << x << "y" << y << "w" << w << "h" << h << "}" << "}";
	         	 	 	 	}
	         	 	 	 fs << "]" << "}" <<"}";
	    }
	    fs << "]";
	    fs.release();
*/





DetectionToRecognition::DetectionToRecognition(string fileNameIn,string fileNameOut) {
	// TODO Auto-generated constructor stub
	fs.open(fileNameIn, FileStorage::READ);
	fsOut.open(fileNameOut, FileStorage::WRITE);
}


bool DetectionToRecognition::bBoxListIsNotVoid(){
bool returnValue;

	if(framesList.getSize()>0)
		returnValue=true;
	else
		returnValue=false;

	return(returnValue);

}

BBoxAtFrame DetectionToRecognition::getNextBBox(){

BBoxAtFrame *returnV;

if(bBoxListIsNotVoid()){
		returnV=&BBoxAtFrame(framesList.getFrame(0).getTimePoint(),framesList.getFrame(0).GetBBox(0));

	}else
		returnV=&BBoxAtFrame(-1,cvRect(-1,-1,-1,-1));
	return(*returnV);
}


void DetectionToRecognition::readFile(){

	cout << endl << "print";
	cout << endl;



	cout << endl << "TOTF: "<< (int)fs["nrFrames"];

	FileNode Frames = fs["Frames"];
	FileNodeIterator it = Frames.begin(), it_end = Frames.end();

	FileNode bboxes;
	FileNodeIterator idBoxes,idBoxes_end;

	// iterate through a sequence using FileNodeIterator
	for( ; it != it_end; ++it )
	{

		cout << endl << "Id = " << (int)(*it)["Frame"]["frameId"];
		cout << endl << "millisec = " << (int)(*it)["Frame"]["millsecFrame"];
		cout << endl << "detectionTime = " << (int)(*it)["Frame"]["detectionTime"];
		bboxes=(*it)["Frame"]["BBoxes"];
		cout << endl << "Size = " << bboxes.size() << " none:" << bboxes.isNone() << " Empty:" << bboxes.empty();
		idBoxes=bboxes.begin();
		idBoxes_end=bboxes.end();
		for( ; idBoxes != idBoxes_end; ++idBoxes ){
			cout<< endl << "   Box " << (int)(*idBoxes)["BBox"]["idBox"] << " X:" << (int)(*idBoxes)["BBox"]["x"]<< " Y:" << (int)(*idBoxes)["BBox"]["y"]<< " W:" << (int)(*idBoxes)["BBox"]["w"]<< " H:" << (int)(*idBoxes)["BBox"]["h"];
		}
		cout << endl << "-----------------------" << endl;

	}
	cout << endl;
}

/*
void DetectionToRecognition::loadFile(){
	SingleFrame frameSingle;
	FramesList listFrame;

    for(int frames=0;frames<3;++frames){
    	for(int count=0;count<3;++count){
    		frameSingle=SingleFrame();
    		frameSingle.addBBox(count+frames,count+frames,count+frames,count+frames);
    		listFrame.addFrame(frameSingle);
    	}
    }

	listFrame.printFrameList();



}*/

int DetectionToRecognition::getTotalFrames(){
	return(totalFrames);
}


void DetectionToRecognition::loadFile(){


	int x,y,w,h;

	SingleFrame frameSingle(-1);
	totalFrames=(int)fs["nrFrames"];
	FileNode Frames = fs["Frames"];
	FileNodeIterator it = Frames.begin(), it_end = Frames.end();

	FileNode bboxes;
	FileNodeIterator idBoxes,idBoxes_end;



	// iterate through a sequence using FileNodeIterator
	for( ; it != it_end; ++it )
	{
		frameSingle=SingleFrame((int)(*it)["Frame"]["frameId"]);
		bboxes=(*it)["Frame"]["BBoxes"];
		if(!bboxes.isNone()){
			idBoxes=bboxes.begin();
			idBoxes_end=bboxes.end();
			for( ; idBoxes != idBoxes_end; ++idBoxes ){
				x = (int)(*idBoxes)["BBox"]["x"];
				y = (int)(*idBoxes)["BBox"]["y"];
				w = (int)(*idBoxes)["BBox"]["w"];
				h = (int)(*idBoxes)["BBox"]["h"];
				frameSingle.addBBox(x,y,w,h);
			}
			framesList.addFrame(frameSingle);
		}
	}

 cout << endl << "PRINT FILE";
 framesList.printFrameList();
 cout << endl;
}



void DetectionToRecognition::addFace(){
	 fsOut  << "{" << "Face" << "{" ;
	 fsOut << "BBoxes" << "[";
}

void DetectionToRecognition::closeFace(){
	 fsOut << "]" << "}" <<"}";
}

void DetectionToRecognition::addFacePosition(int frameID,CvRect& bbox){
	//framesList.deleteBBoxOverlaped(frameID,bbox);
	fsOut << "{" << "BBox" << "{" << "frameId" << frameID << "x" << bbox.x << "y" << bbox.y << "w" << bbox.width << "h" << bbox.height << "}" << "}";
}

void DetectionToRecognition::setFaceAlreadyCheckedOrInvalid(int frameID,CvRect& bbox){
	framesList.deleteBBoxOverlaped(frameID,bbox);
}

void DetectionToRecognition::initDetToRec(){
	 fsOut << "nrFrames" << totalFrames;
	 fsOut << "Faces" << "[";
}

void DetectionToRecognition::closeDetToRec(){
	 fsOut << "]";
}


bool DetectionToRecognition::ckeckIfTracking(int frameID, CvRect& pbox){
	return(framesList.ckeckIfTracking(frameID,pbox));
}

DetectionToRecognition::~DetectionToRecognition() {
	fs.release();
	fsOut.release();
	// TODO Auto-generated destructor stub
}
