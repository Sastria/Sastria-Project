/*
 * frameWk.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: Alessandro Romanino
 *
 */

#include "frameWk.h"
#include <algorithm>
using namespace cv;
using namespace std;

void drawBox(Mat& image, CvRect box, Scalar color, int thick){
  rectangle( image, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),color, thick);
} 

void drawPoints(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i )
      {
      Point center( cvRound(i->x ), cvRound(i->y));
      circle(image,center,2,color,1);
      }
}


CvRect getConnectedComponent(cv::Mat& image,int x,int y,CvSize WinSize){
	Mat visitedImage;
	int xLeft,xRight,yTop,yBottom,xS,xE,yS,yE;
	CvRect bBox;


	xLeft=image.cols;
	xRight=0;
	yTop=image.rows;
	yBottom=0;

	image.copyTo(visitedImage);
	xS=x-(WinSize.width/2);
	xE=x+(WinSize.width/2);

	yS=y-(WinSize.height/2);
	yE=y+(WinSize.height/2);


	for(int countx=xS;countx<=xE;++countx)
		for(int county=yS;county<=yE;++county)
			getMinMaxXYPointR(visitedImage,countx,county,xLeft,xRight,yTop,yBottom);


	bBox=cvRect(xLeft,yTop,xRight-xLeft,yBottom-yTop);

	return(bBox);


}


void getMinMaxXYPointR(cv::Mat& image,int x,int y,int& xLeft,int& xRight, int& yTop, int& yBottom){


	//cout << endl << "imgvalue("<<x<<";"<<y<<"):" <<((int)image.at<uchar>(y,x));
	//cvWaitKey(0);

  if((x>=0) && (x<image.cols) && (y>=0) && (y< image.rows) && (((int)image.at<uchar>(y,x))!=0)){

	  image.at<uchar>(y,x)=0;
	  //imshow("getminmax",image);
	  //cvWaitKey(0);
	  if(xLeft>x) xLeft=x;
	  else if(xRight<x) xRight=x;

	  if(yTop>y) yTop=y;
	  else if(yBottom<y) yBottom=y;

	  // TL
	  getMinMaxXYPointR(image,x-1,y-1,xLeft,xRight,yTop,yBottom);
	  // T
	  getMinMaxXYPointR(image,x,y-1,xLeft,xRight,yTop,yBottom);
	  // TR
	  getMinMaxXYPointR(image,x+1,y-1,xLeft,xRight,yTop,yBottom);
	  // L
	  getMinMaxXYPointR(image,x-1,y,xLeft,xRight,yTop,yBottom);
	  // R
	  getMinMaxXYPointR(image,x+1,y,xLeft,xRight,yTop,yBottom);
	  // LB
	  getMinMaxXYPointR(image,x-1,y+1,xLeft,xRight,yTop,yBottom);
	  // B
	  getMinMaxXYPointR(image,x,y+1,xLeft,xRight,yTop,yBottom);
	  // RB
	  getMinMaxXYPointR(image,x+1,y+1,xLeft,xRight,yTop,yBottom);
  }

}


int getBlackPixel(cv::Mat image){
	int values=0;

	//imshow("patchimg",image);
	for(int riga=0;riga<image.rows;++riga){
			for(int col=0;col<image.cols;++col){
				//cout << endl << riga << "," << col;
				if(((int)image.at<uchar>(riga,col))==0)
					values++;
			}
	}
	return(values);
}

void removeIsolatedPoints(cv::Mat& image,CvSize& winSearch){
    CvRect bbox=cvRect(0,0,0,0);
    int xDiff=winSearch.width/2;
    int yDiff=winSearch.height/2;
    int thresholdVal;
    int blckP;
    Mat tmpImg;

    image.copyTo(tmpImg);

	for(int riga=0;riga<image.rows;++riga){
		//imshow("filtro",image);
	    //cvWaitKey(0);
		for(int col=0;col<image.cols;++col){
			if(((int)image.at<uchar>(riga,col))!=0){
			bbox.x= (col -  xDiff) >0 ? (col - xDiff) : 0;
			bbox.y= (riga - yDiff) >0 ? (riga - yDiff) : 0;
			bbox.width=(col+xDiff)-bbox.x;
			bbox.width=(bbox.width+bbox.x) < image.cols ? bbox.width : image.cols-bbox.x;
			bbox.height=(riga+yDiff)-bbox.y;
			bbox.height=(bbox.height+bbox.y) < image.rows ? bbox.height : image.rows-bbox.y;
			thresholdVal=((bbox.width*bbox.height))/2;
			//cout << endl << "r,c:" << image.rows << "," << image.cols;
			//cout << endl << "prima di getblack " << bbox.x <<"," << bbox.y<<"," <<bbox.width <<","<< bbox.height;
			blckP=getBlackPixel(tmpImg(bbox));
			//cout << endl << "dopo di getblack";
			if(blckP>thresholdVal)
				image.at<uchar>(riga,col)=0;
			}
		}
	}

}


Mat createMask(const Mat& image, CvRect box){
  Mat mask = Mat::zeros(image.rows,image.cols,CV_8U);
  drawBox(mask,box,Scalar::all(255),CV_FILLED);
  return mask;
}

float median(vector<float> v)
{
    int n = floor(v.size() / 2);
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

float medianAle(vector<float> v)
{
	int n = floor(v.size() / 2);
	float sum=0;
	sort(v.begin(),v.end());
	cout << endl;
	for(int i=0; i< v.size();++i){
		sum+=v[i];
		if(i==n)
			 cout << "##" <<v[i] <<"##";
		else cout << v[i] <<"-";
	}

	//nth_element(v.begin(), v.begin()+n, v.end());
	sum=(sum/v.size());
	cout << endl << "v[n]=" << v[n];
	cout << endl << "sum=" << sum;
	cout << endl;

	cvWaitKey(0);
	return(sum);
}


vector<int> index_shuffle(int begin,int end){
  vector<int> indexes(end-begin);
  for (int i=begin;i<end;i++){
    indexes[i]=i;
  }
  random_shuffle(indexes.begin(),indexes.end());
  return indexes;
}

