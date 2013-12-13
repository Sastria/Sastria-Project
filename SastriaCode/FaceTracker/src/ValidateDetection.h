/*
 * ValidateDetection.h
 *
 *  Created on: 04/giu/2013
 *      Author: Alessandro Romanino
 */
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef VALIDATEDETECTION_H_
#define VALIDATEDETECTION_H_

class ValidateDetection {
public:
	ValidateDetection();
	virtual ~ValidateDetection();
	bool detectNose( Mat& img);
	bool detectEyes(Mat& img);
	bool detectMouth(Mat& img);
	bool detectGlass(Mat& img);
	bool detectFacialFeatures(Mat& img);
	bool detectFacialFeaturesAlt2(Mat& img);
	bool validateDetection(Mat& img);
	bool detectEyeSmall( Mat& img);
	bool detectEyeBig( Mat& img);
	bool detectProfFace( Mat& img);
	bool checkFaceTrak(Mat& img,CvRect& pbox);
	float bbOverlap(const CvRect& box1,const CvRect& box2);
private:
	CascadeClassifier cascade;
	CascadeClassifier cascadeAlt2;
	CascadeClassifier cascade_eye;
	CascadeClassifier cascade_nose;
	CascadeClassifier cascade_mouth;
	CascadeClassifier cascade_glass;
	CascadeClassifier cascade_eyeBig;
	CascadeClassifier cascade_eyeSmall;
	CascadeClassifier cascade_profFace;
};

#endif /* VALIDATEDETECTION_H_ */
