/*
 * FacePosition.cpp
 *
 *  Created on: 30/mag/2013
 *      Author: Alessandro Romanino
 */

#include "FacePosition.h"

FacePosition::FacePosition() {
	// TODO Auto-generated constructor stub

}


CvRect FacePosition::getFacePosition(){
	return(bBox);
}
int FacePosition::getFramePosition(){
	return(frameId);
}

void FacePosition::addFrameFacePosition(int idF,CvRect bb){
	frameId=idF;
	bBox=bb;
}

FacePosition::~FacePosition() {
	// TODO Auto-generated destructor stub
}
