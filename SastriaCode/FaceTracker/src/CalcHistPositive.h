/*
 * CalcHistPositive.h
 *
 *  Created on: 17/gen/2013
 *      Author: Alessandro Romanino
 */
#include <opencv2/opencv.hpp>
#ifndef CALCHISTPOSITIVE_H_
#define CALCHISTPOSITIVE_H_

using namespace cv;
using namespace std;

class CalcHistPositive {
public:
	CalcHistPositive(int channelA,int channelB,int w, int h);
	void updadeHistPositive(Mat& image,CvRect selectionBoxPositive);
	void updadeHistPositive(Mat& image,Mat& histogramDetection,Mat& negativeHistogramDetection,CvRect selectionBoxPositive);
	void calcPositive(Mat& image, Mat& weightPixel);
	void calcPositive(Mat& image, Mat& weightPixel, Mat detection);
	void calcNegative(Mat& image, Mat& weightPixel);
	void calcNegative(Mat& image, Mat& weightPixel, Mat& detection);
	Mat& getHistogram();
	void calcBackProjection(Mat& image, Mat& backProjection,Mat& backProjectionFake);
	void calcBackProjection(Mat& image, Mat& backProjection);
	void getParticleDetails(Mat& backProjection,CvRect selectionBoxPositive);
	void getRealFrameProjection(cv::Mat& frameHSV,cv::Mat& backProjection,CvRect lastboxTrack);
	//void getMinMaxXYPointR(cv::Mat& image,cv::Mat& colorImage,CvRect lastboxTrack,int& xLeft,int& xRight, int& yTop, int& yBottom);
	void getDetectionByHistogram(cv::Mat& frameHSV,cv::Mat& image,cv::Mat& backProjection,cv::Mat& detection,cv::Mat& negativeDetection,CvRect lastboxTrack,CvRect& blobDetection);
	void getBackProjectionNegative(cv::Mat& image,cv::Mat& imageNegative);
	void removeFakeColorFromRealFrame(cv::Mat& backProjection,CvRect bBox);//cv::Mat& frameHSV,cv::Mat& backProjection,cv::Mat& backProjectionCopy,cv::Mat& realEliminaterFrame,int row, int col);
	int calcDensity(cv::Mat detection,cv::Mat& debug);
	void clear();
	virtual ~CalcHistPositive();
private:
	float bbOverlapLastBB(CvRect& box1,CvRect& box2);
	bool getIfOverlap(CvRect& boxE,CvRect& boxI);
	CvRect getNewBox(CvRect& boxE,CvRect& boxI);
	Mat histPositive;
	Mat histPNorm;
	Mat histNNorm;
	Mat fakeProjection;
};

#endif /* CALCHISTPOSITIVE_H_ */
