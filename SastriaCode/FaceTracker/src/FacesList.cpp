/*
 * FacesList.cpp
 *
 *  Created on: 30/mag/2013
 *      Author: Alessandro Romanino
 */

#include "FacesList.h"

FacesList::FacesList() {
	// TODO Auto-generated constructor stub

}


void FacesList::addFace(PositionList face){
	faces.push_back(face);
}

PositionList FacesList::getFace(int index){
	return(faces[index]);
}

int FacesList::getSize(){
	return(faces.size());
}


FacesList::~FacesList() {
	// TODO Auto-generated destructor stub
}
