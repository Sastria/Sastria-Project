/*
 * myClassifier.h
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 *      review: Alessandro Romanino
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "IntegralImage.h"
#include <algorithm>
using namespace cv;
using namespace std;

class myClassifier{
private:
  float thr_fern;
  int structSize;
  int nstructs;
  float valid;
  float ncc_thesame;
  float thr_nn;
  float best_calculated_thr_nn;
  int acum;
  float thr_nn_valid;
public:
  //Parameters

  float getNegativeThreshold(){return (thrN);}
  void read(const cv::FileNode& file);
  void prepare(const std::vector<cv::Size>& scales);
  void getFeatures(const cv::Mat& image,IntegralImage& integralFrame,const int& scale_idx,std::vector<int>& fern);
  void update(const std::vector<int>& fern, int C, int N);
  void forcedUpdate(const vector<int>& fern);
  void forcedTrainFern(vector<std::pair<vector<int>,int> >& ferns,int resample);
  float measure_forest(std::vector<int> fern);
  void trainFern(std::vector<std::pair<std::vector<int>,int> >& ferns,int resample);
  void trainNNN(const std::vector<cv::Mat>& nn_examples);
  void NNConf(const cv::Mat& example,std::vector<int>& isin,float& rsconf,float& csconf);
  void evaluateThV(const std::vector<cv::Mat>& nExT);
  void show();
  //Ferns Members
  int getNumStructs(){return nstructs;}
  float getFernTh(){return thr_fern;}
  float getThr_nn_Valid(){return thr_nn_valid;}
  float getBestCalculatedThrNN(){return best_calculated_thr_nn;}
  float getNNTh(){return thr_nn;}
 // float getNNTh(){return best_calculated_thr_nn;}
  struct Feature
      {

          int x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1(_x1), y1(_y1), x2(_x2), y2(_y2)
          {}
          int operator ()(const cv::Mat& patch,IntegralImage& integralFrame) const
          {
        	  int firstPatch,secondPatch,thirdPatch,fourthPatch;
        	  int LeftSide,RightSide;
        	  RNG& rng = theRNG();

        	 // Mat tmpPatch;

        	//  cout << endl << "X1 X2 Y1 Y2 " <<(int) x1 << " " <<(int) x2 << " " << (int)y1 << " " <<(int) y2 << endl;

            //  patch.copyTo(tmpPatch);
        	  //cout << endl << "patch.size(" << patch.cols << "," << patch.rows << ") [x1,y1,x2,y2]=[" << (int)x1 << "," << (int)y1 << "," << (int)x2 << "," << (int)y2 <<"]" << endl;
        	  firstPatch= integralFrame.getBoxSum(cvRect(x1,y1,x2,y2/2));
        	  secondPatch=integralFrame.getBoxSum(cvRect(x1,y1+(y2/2),x2,y2/2));
        	  thirdPatch= integralFrame.getBoxSum(cvRect(x1,y1,x2/2,y2));
        	  fourthPatch=integralFrame.getBoxSum(cvRect(x1+(x2/2),y1,x2/2,y2));
        	  //if(firstPatch<0 || secondPatch<0)
        		//  cvWaitKey(0);
        	// rectangle(tmpPatch,cvRect(x1,y1,x2,y2/2),cvScalar(0),1);
        	// rectangle(tmpPatch,cvRect(x1,y1+(y2/2),x2,y2/2),cvScalar(254),1);
        	// rectangle(tmpPatch,cvRect(x1,y1,x2/2,y2),cvScalar(0),1);
        	 //rectangle(tmpPatch,cvRect(x1+(x2/2),y1,x2/2,y2),cvScalar(254),1);
        	  //circle(tmpPatch,cvPoint(XPatchCenter,YPatchCenter),5,cvScalar(254),1);
        	  //imshow("Patch",tmpPatch);
        	 // cvWaitKey(0);
        	  //return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2);
        	 // cout << "fP:"<<firstPatch << " sp:" << secondPatch << " bVal:"<< (firstPatch>secondPatch) << endl;
        	  if(firstPatch<0 || secondPatch<0 || thirdPatch<0 || fourthPatch<0){
        		  cout << "fernclassifie.h Errore in integral Image: p1=" << firstPatch << " p2=" << secondPatch << " p3=" << thirdPatch << " p4=" << fourthPatch << " X1=" << (int)x1 << " y1=" << (int)y1 << " x2=" << (int)x2 << " y2=" << (int)y2 ;
        		  cout << endl;
        		  cvWaitKey(0);
        	  }

        	  if(firstPatch==secondPatch==thirdPatch==fourthPatch){
        		  LeftSide=(((float)rng)) > 0.5 ? 2:0;
        		  RightSide=(((float)rng)) > 0.5 ? 1:0;

        	  }else{

        	  LeftSide= firstPatch > secondPatch ? 0 : 2;
        	  RightSide= thirdPatch > fourthPatch ? 0 : 1;
        	  }

        	  return (LeftSide + RightSide);
          }
      };
  std::vector<std::vector<Feature> > features; //Ferns features (one std::vector for each scale)
  std::vector< std::vector<int> > nCounter; //negative counter
  std::vector< std::vector<int> > pCounter; //positive counter
  std::vector< std::vector<float> > posteriors; //Ferns popExsteriors
  float thrN; //Negative threshold
  float thrP;  //Positive thershold
  //NN Members
  std::vector<cv::Mat> pEx; //NN positive examples
  std::vector<cv::Mat> nEx; //NN negative examples
  std::vector<int> pFx;
  std::vector<int> nFx;
};
