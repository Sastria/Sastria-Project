/*
 * SingleFrame.h
 *
 *  Created on: 24/mag/2013
 *      Author: Alessandro Romanino
 */
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef SINGLEFRAME_H_
#define SINGLEFRAME_H_

class SingleFrame {
public:
	SingleFrame(int timeP);
	virtual ~SingleFrame();
	void addBBox(int x, int y, int w, int h);
	CvRect& GetBBox(int index);
	void deleteBBox(int index);
	int getTimePoint();
	void printBoxes();
	int getSize();
private:
	vector<CvRect> bBoxes;
	int frameId;


};

#endif /* SINGLEFRAME_H_ */
