/*
 * PositionList.cpp
 *
 *  Created on: 30/mag/2013
 *      Author: Alessandro Romanino
 */

#include "PositionList.h"

PositionList::PositionList() {
	// TODO Auto-generated constructor stub

}


void PositionList::addPosition(FacePosition fp){
	positions.push_back(fp);
}

FacePosition PositionList::getPosition(int index){
	return(positions[index]);
}

PositionList::~PositionList() {
	// TODO Auto-generated destructor stub
}
