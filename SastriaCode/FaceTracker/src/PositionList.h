/*
 * PositionList.h
 *
 *  Created on: 30/mag/2013
 *      Author: Alessandro Romanino
 */

#include "FacePosition.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef POSITIONLIST_H_
#define POSITIONLIST_H_

class PositionList {
public:
	PositionList();
	virtual ~PositionList();
	void addPosition(FacePosition fp);
	FacePosition getPosition(int index);
private:
	vector<FacePosition> positions;
};

#endif /* POSITIONLIST_H_ */
