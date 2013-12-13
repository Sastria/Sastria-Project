/*
 * FacesList.h
 *
 *  Created on: 30/mag/2013
 *      Author: Alessandro Romanino
 */

#include "PositionList.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef FACESLIST_H_
#define FACESLIST_H_

class FacesList {
public:
	FacesList();
	virtual ~FacesList();
	void addFace(PositionList face);
	PositionList getFace(int index);
	int getSize();

private:
	vector<PositionList> faces;

};

#endif /* FACESLIST_H_ */
