/*
 * SingleFrame.cpp
 *
 *  Created on: 24/mag/2013
 *      Author: Alessandro Romanino
 */

#include "SingleFrame.h"

SingleFrame::SingleFrame(int timeP) {
	// TODO Auto-generated constructor stub
	frameId=timeP;
}

int SingleFrame::getTimePoint(){
	return(frameId);
}

void SingleFrame::addBBox(int x, int y, int w, int h){
	//CvRect bb=cvRect(x,y,w,h);
	bBoxes.push_back(cvRect(x,y,w,h));
}


int SingleFrame::getSize(){
	return(bBoxes.size());
}


void SingleFrame::deleteBBox(int index){
	vector<CvRect>::iterator start=bBoxes.begin()+index;
	bBoxes.erase(start);
}

CvRect& SingleFrame::GetBBox(int index){
	return(bBoxes[index]);
}



void SingleFrame::printBoxes(){
	/*
	vector<CvRect>::iterator start=bBoxes.begin();
	vector<CvRect>::iterator end=bBoxes.end();
	*/
	/*
	cout<< endl << " value:" << value;
	int cont=0;
	while(cont<bBoxes.size()){
 		 //cout<< endl << "  Box  X:" << ((CvRect&)start).x << " Y:" << ((CvRect&)start).y << " W:" << ((CvRect&)start).width << " H:" << ((CvRect&)start).height;
		  cout<< endl << " X:" << ((int)bBoxes[cont]);
		  ++cont;
		}
	cout << endl;
	*/



	vector<CvRect>::iterator start=bBoxes.begin();
	vector<CvRect>::iterator end=bBoxes.end();
	cout << endl<< "BBOXS";
	while(start!=end){
		   cout<< endl << "  Box  X:" << ((CvRect)*start).x << " Y:" << ((CvRect)*start).y << " W:" << ((CvRect)*start).width << " H:" << ((CvRect)*start).height;
		   //cout<< endl << "  Box  X:" << ((int)*start);
			++start;
	}
	cout << endl << "BBOXE";

}

SingleFrame::~SingleFrame() {
	// TODO Auto-generated destructor stub
}
