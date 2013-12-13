/*
 * frameWk.h
 *
 *  Created on: Jun 14, 2011
 *      Author: Alessandro Romanino
 *
 */

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//#include <algorithm>
#pragma once



void drawBox(cv::Mat& image, CvRect box, cv::Scalar color = cvScalarAll(255), int thick=1); 



void getMinMaxXYPointR(cv::Mat& image,int x,int y,int& xLeft,int& xRight, int& yTop, int& yBottom);
CvRect getConnectedComponent(cv::Mat& image,int x,int y,CvSize WinSize);
void removeIsolatedPoints(cv::Mat& image,CvSize& winSearch);
int getBlackPixel(cv::Mat& image);


void drawPoints(cv::Mat& image, std::vector<cv::Point2f> points,cv::Scalar color=cv::Scalar::all(255));

cv::Mat createMask(const cv::Mat& image, CvRect box);

float median(std::vector<float> v);
float medianAle(std::vector<float> v);

std::vector<int> index_shuffle(int begin,int end);

