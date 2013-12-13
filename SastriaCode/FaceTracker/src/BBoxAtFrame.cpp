/*
 * BBoxAtFrame.cpp
 *
 *  Created on: 29/mag/2013
 *      Author: Alessandro Romanino
 */

#include "BBoxAtFrame.h"

BBoxAtFrame::BBoxAtFrame(int frameId,CvRect bboxCrop) {
	// TODO Auto-generated constructor stub
  idFrame=frameId;
  bBox=bboxCrop;
}


int BBoxAtFrame::getFrameId(){
	return(idFrame);
}

CvRect BBoxAtFrame::getBBoxCrop(){
	return(bBox);
}

BBoxAtFrame::~BBoxAtFrame() {
	// TODO Auto-generated destructor stub
}
