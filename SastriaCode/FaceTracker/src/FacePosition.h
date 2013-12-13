/*
 * FacePosition.h
 *
 *  Created on: 30/mag/2013
 *      Author: Alessandro Romanino
 */


#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef FACEPOSITION_H_
#define FACEPOSITION_H_

class FacePosition {
public:
	FacePosition();
	virtual ~FacePosition();
	CvRect getFacePosition();
	int getFramePosition();
	void addFrameFacePosition(int idF,CvRect bb);
private:
	CvRect bBox;
	int frameId;
};

#endif /* FACEPOSITION_H_ */
