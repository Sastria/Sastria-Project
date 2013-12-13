/*
 * FramesList.h
 *
 *  Created on: 24/mag/2013
 *      Author: Alessandro Romanino
 */
#include <opencv2/opencv.hpp>
#include "SingleFrame.h"
using namespace cv;
using namespace std;


#ifndef FRAMESLIST_H_
#define FRAMESLIST_H_

class FramesList {
public:
	FramesList();
	virtual ~FramesList();
	void addFrame(SingleFrame frame);
	void deleteBBoxOverlaped(int frameID,CvRect& bbox);
	void deleteFrameIfVoid(int framePos);
	SingleFrame& getFrame(int index);
	int getFramePos(int idFrame);
	void printFrameList();
	int getSize();
	float bbOverlap(const CvRect& box1,const CvRect& box2);
	bool ckeckIfTracking(int frameID,CvRect& pbox);
	bool isInternal(CvRect bboxHull, CvRect bbox);
	float bbIntersection(const CvRect& boxHull,const CvRect& box);

private:
	vector<SingleFrame> frames;

};

#endif /* FRAMESLIST_H_ */
