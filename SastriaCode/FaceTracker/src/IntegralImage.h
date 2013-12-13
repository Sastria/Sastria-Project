/*
 * IntegralImage.h
 *
 *  Created on: 03/lug/2012
 *      Author: Alessandro Romanino
 */
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


#ifndef INTEGRALIMAGE_H_
#define INTEGRALIMAGE_H_

class IntegralImage {
public:
	IntegralImage();
	IntegralImage(Mat frameToIntegral);
	int getBoxSum(CvRect box);
	float getBoxSumMedian(CvRect box);
	Mat& getIntegralImageMat();
	virtual ~IntegralImage();
private:
Mat integralImageFrame;
CvSize integralImageSize;
CvPoint integralImageCenter;

};

#endif /* INTEGRALIMAGE_H_ */
