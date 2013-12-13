/*
 * BBoxAtFrame.h
 *
 *  Created on: 29/mag/2013
 *      Author: Alessandro Romanino
 */

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef BBOXATFRAME_H_
#define BBOXATFRAME_H_

class BBoxAtFrame {
public:
	BBoxAtFrame(int frameId,CvRect bboxCrop);
	int getFrameId();
	CvRect getBBoxCrop();
	virtual ~BBoxAtFrame();
private:
	int idFrame;
	CvRect bBox;
};

#endif /* BBOXATFRAME_H_ */
