/*
 * DetectionToRecognition.h
 *
 *  Created on: 22/mag/2013
 *      Author: Alessandro Romanino
 */
#include <opencv2/opencv.hpp>
#include "SingleFrame.h"
#include "FramesList.h"
#include "FacesList.h"
#include "BBoxAtFrame.h"

//#include <iostream>
//#include <string>
using namespace cv;
using namespace std;

#ifndef DETECTIONTORECOGNITION_H_
#define DETECTIONTORECOGNITION_H_

class DetectionToRecognition {
public:
	DetectionToRecognition(string fileNameIn,string fileNameOut);
	void readFile();
	void loadFile();
	void initDetToRec();
	void closeDetToRec();
	void addFace();
	void addFacePosition(int frameID,CvRect& bbox);
	void closeFace();
	bool ckeckIfTracking(int frameID,CvRect& pbox);
	void setFaceAlreadyCheckedOrInvalid(int frameID,CvRect& bbox);
	virtual ~DetectionToRecognition();
	bool bBoxListIsNotVoid();
	BBoxAtFrame getNextBBox();
	int getTotalFrames();
private:
	FileStorage fs;
	FileStorage fsOut;
	FramesList framesList;
	int totalFrames;
	//FacesList facesList;


};

#endif /* DETECTIONTORECOGNITION_H_ */
