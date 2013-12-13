/*
 * IntegralImage.cpp
 *
 *  Created on: 03/lug/2012
 *      Author: Alessandro Romanino
 */

#include "IntegralImage.h"

IntegralImage::IntegralImage() {
	// TODO Auto-generated constructor stub

}

/*
  cout << endl << "Frame To Integral Type | depth: " << frameToIntegral.type() << " " << frameToIntegral.depth();
 cout << endl << "Integral Image    Type | depth: " << integralImageFrame.type() << " " << integralImageFrame.depth();
 cvWaitKey(0);
 cout << endl << "Frame To Integral cout" << frameToIntegral << endl;
 cvWaitKey(0);
 cout << "Integral Image Values element by element";
  for(int r=0;r<frameToIntegral.rows;++r){
	  cout << endl;
	  for(int c=0;c<frameToIntegral.cols;++c){
		  cout <<(int) frameToIntegral.at<unsigned char>(r,c) << " ";
  	  }
}
  cvWaitKey(0);
  cout << endl <<"Integral Image Out "<< endl << integralImageFrame << endl;
cvWaitKey(0);
  cout << endl << "Integral Image Values elementbyElement";
   for(int r=0;r<integralImageFrame.rows;++r){
 	  cout << endl;
 	  for(int c=0;c<integralImageFrame.cols;++c){
 		  cout <<(int)integralImageFrame.at<signed int>(r,c) << " ";
   	  }
 }
  cvWaitKey(0);

 */

IntegralImage::IntegralImage(Mat frameToIntegral){
 integralImageSize=cvSize(frameToIntegral.cols+1,frameToIntegral.rows+1);
 integralImageCenter=cvPoint(integralImageSize.width/2,integralImageSize.height/2);
// integralImageFrame=Mat(integralImageSize,frameToIntegral.type());
 integral(frameToIntegral,integralImageFrame,frameToIntegral.depth());

}

Mat& IntegralImage::getIntegralImageMat(){
	return integralImageFrame;
}

int IntegralImage::getBoxSum(CvRect box){
	int xS,yS,xE,yE;
	int a,b,c,d;
	Mat printingimg;

	xS=box.x;
	yS=box.y;
	xE=box.width+box.x-1;
	yE=box.height+box.y-1;
	a=(int)integralImageFrame.at<signed int>(yS,xS);
	b=(int)integralImageFrame.at<signed int>(yS,xE);
	c=(int)integralImageFrame.at<signed int>(yE,xS);
	d=(int)integralImageFrame.at<signed int>(yE,xE);

	//cout << endl << integralImageFrame << endl;
/*
	cout << "PatchSize:(W,H):(" << integralImageSize.width << "," << integralImageSize.height << ") XS,YS,XE,YE:("<< xS << "," << yS << "," << xE << "," << yE << ") "  << "a:" << a << " b:" << b << " c:" << c << " d:" << d << " d+a-b-c:" << d+a-b-c  << endl;
	if((d+a-b-c)<=0){
		printingimg=integralImageFrame(cvRect(xS-3,yS-3,xE-xS+3,yE-yS+3));
		cout << endl <<printingimg << endl;
	}

	cout << endl;*/

	return(d+a-b-c);
}

float IntegralImage::getBoxSumMedian(CvRect box){
	int xS,yS,xE,yE;
	float a,b,c,d;
	int Area=box.width*box.height;

	//cout << endl << integralImageFrame << endl;
	xS=box.x;
	yS=box.y;
	xE=box.width+box.x;
	yE=box.height+box.y;
	a=(int)integralImageFrame.at<signed int>(yS,xS);
	b=(int)integralImageFrame.at<signed int>(yS,xE);
	c=(int)integralImageFrame.at<signed int>(yE,xS);
	d=(int)integralImageFrame.at<signed int>(yE,xE);


	if((a>b) || (a>c) || (b>d) || (c>d)){
	 cout <<"XS,YS,XE,YE:("<< xS << "," << yS << "," << xE << "," << yE << ") "  << "a:" << a << " b:" << b << " c:" << c << " d:" << d << " d+a-b-c:" << d+a-b-c << " d+a-b-c/area:" << (d+a-b-c)/Area  << endl;
	 cout << endl;
	 cvWaitKey(0);
	}
	// cout <<"XS,YS,XE,YE:("<< xS << "," << yS << "," << xE << "," << yE << ") "  << "a:" << a << " b:" << b << " c:" << c << " d:" << d << " d+a-b-c:" << d+a-b-c << " d+a-b-c/area:" << (d+a-b-c)/Area  << endl;

//	return(d+a-b-c);
	return((float)((d+a-b-c))/Area);
}

IntegralImage::~IntegralImage() {
	// TODO Auto-generated destructor stub
}
