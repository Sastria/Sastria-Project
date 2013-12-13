/*
 * FramesList.cpp
 *
 *  Created on: 24/mag/2013
 *      Author: Alessandro Romanino
 */

#include "FramesList.h"


FramesList::FramesList() {
	// TODO Auto-generated constructor stub

}


void FramesList::addFrame(SingleFrame frame){
	frames.push_back(frame);
}

SingleFrame& FramesList::getFrame(int index){
	return(frames[index]);
}


int FramesList::getFramePos(int idFrame){

    int framesNumber=frames.size();
	int returnV;
	int count=0;
	while((count<framesNumber)&&(idFrame>frames[count].getTimePoint())){
			++count;
	}

	if(idFrame==frames[count].getTimePoint()){
				returnV=count;
	}else returnV=-1;

	return(returnV);

}

void FramesList::deleteFrameIfVoid(int framePos){

	vector<SingleFrame>::iterator posFrame=frames.begin()+framePos;
	int frameSize=((SingleFrame)*posFrame).getSize();
	//cout << endl << "FrameID: " << ((SingleFrame)*posFrame).getTimePoint();
	//cout << endl;
	if(frameSize==0)
		frames.erase(posFrame);
}

void FramesList::deleteBBoxOverlaped(int frameID, CvRect& bbox){


	int framePos=getFramePos(frameID);
		float bestOverlap=-1;
		float currentOverlap;
		vector<int> bestOverlapBbox;
		CvRect cbbox;
		bool ovlp=false;

		if(framePos!=-1){
			SingleFrame& frameBB=getFrame(framePos);
			int bboxesNumb=frameBB.getSize();
			int count=0;
			while(count<bboxesNumb){
				cbbox=frameBB.GetBBox(count);
				currentOverlap=bbOverlap(bbox,cbbox);
				if((bbIntersection(bbox,cbbox)>0.5)||(bbIntersection(cbbox,bbox)>0.5)){
					if(((currentOverlap>=0.2))&& (currentOverlap>bestOverlap)){
						bestOverlap=currentOverlap;
						bbox.x=(bbox.x+4*cbbox.x)/5;
						bbox.y=(bbox.y+4*cbbox.y)/5;
						bbox.width=(bbox.width+4*cbbox.width)/5;
						bbox.height=(bbox.height+4*cbbox.height)/5;
					}
					bestOverlapBbox.push_back(count);
					ovlp=true;
				}
				++count;
			}
			if(ovlp){
				while(bestOverlapBbox.size()>0){
				frameBB.deleteBBox(bestOverlapBbox.back());
				bestOverlapBbox.pop_back();
				}
			}
			deleteFrameIfVoid(framePos);
		}
}


bool FramesList::isInternal(CvRect bboxHull, CvRect bbox){

	if( (bbox.x>=bboxHull.x) &&
		((bbox.x+bbox.width)<=(bboxHull.x+bboxHull.width)) &&
		(bbox.y>=bboxHull.y) &&
		((bbox.y+bbox.height)<=(bboxHull.y+bboxHull.height)))
		return true;
	else
		return false;
}


bool FramesList::ckeckIfTracking(int frameID, CvRect& pbox){
	int framePos=getFramePos(frameID);
	float bestOverlap=-1;
	float currentOverlap,currentOverlapPbox;

	vector<int> bestOverlapBbox;
	CvRect cbbox;
	bool ovlp=false;

	if(framePos!=-1){
		SingleFrame& frameBB=getFrame(framePos);
		int bboxesNumb=frameBB.getSize();
		int count=0;
		while(count<bboxesNumb){
			cbbox=frameBB.GetBBox(count);
			currentOverlap=bbOverlap(pbox,cbbox);

			if((bbIntersection(pbox,cbbox)>0.5)||(bbIntersection(cbbox,pbox)>0.5)){
				if(((currentOverlap>=0.2)) && (currentOverlap>bestOverlap)){
					bestOverlap=currentOverlap;
					pbox.x=cbbox.x;
					pbox.y=cbbox.y;
					pbox.width=cbbox.width;
					pbox.height=cbbox.height;
				}
				bestOverlapBbox.push_back(count);
				ovlp=true;
			}
			++count;
		}
		if(ovlp){
			while(bestOverlapBbox.size()>0){
			frameBB.deleteBBox(bestOverlapBbox.back());
			bestOverlapBbox.pop_back();
			}
		}
		deleteFrameIfVoid(framePos);
	}


	return(ovlp);
}



void FramesList:: printFrameList(){

	vector<SingleFrame>::iterator start=frames.begin();
	vector<SingleFrame>::iterator end=frames.end();

	while(start!=end){
		cout << endl << "------------------------" << ((SingleFrame)*start).getTimePoint();
		((SingleFrame)*start).printBoxes();
		++start;
	}
}



float FramesList::bbOverlap(const CvRect& box1,const CvRect& box2){
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}


float FramesList::bbIntersection(const CvRect& boxHull,const CvRect& box){
  if (boxHull.x > box.x+box.width) { return 0.0; }
  if (boxHull.y > box.y+box.height) { return 0.0; }
  if (boxHull.x+boxHull.width < box.x) { return 0.0; }
  if (boxHull.y+boxHull.height < box.y) { return 0.0; }

  float colInt =  min(boxHull.x+boxHull.width,box.x+box.width) - max(boxHull.x, box.x);
  float rowInt =  min(boxHull.y+boxHull.height,box.y+box.height) - max(boxHull.y,box.y);

  float intersection = colInt * rowInt;
  float area1 = box.width*box.height;
  return intersection / (area1);
}



int FramesList::getSize(){
	return(frames.size());
}


FramesList::~FramesList() {
	// TODO Auto-generated destructor stub
}
