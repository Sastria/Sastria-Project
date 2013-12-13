/*
 * FeaturesTrack.h
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 *      revised: Alessandro Romanino
 *
 */


#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include "TrackerKanade.h"
#include "myClassifier.h"
#include <features.h>
#include "fstream"
#include "IntegralImage.h"
#include <opencv2/legacy/legacy.hpp>
#include "ValidateDetection.h"
using namespace cv;
using namespace std;

//Bounding Boxes
struct BoundingBox : public cv::Rect {
  BoundingBox(){}
  BoundingBox(cv::Rect r): cv::Rect(r){}
public:
  float overlap;        //Overlap with current Bounding Box
  int sidx;             //scale index
};

//Detection structure
struct DetStruct {
    std::vector<int> bb;
    std::vector<std::vector<int> > patt;
    std::vector<float> conf1;
    std::vector<float> conf2;
    std::vector<std::vector<int> > isin;
    std::vector<cv::Mat> patch;
  };
//Temporal structure
  struct TempStruct {
    std::vector<std::vector<int> > patt;
    std::vector<float> conf;
  };

struct OComparator{
  OComparator(const std::vector<BoundingBox>& _grid):grid(_grid){}
  std::vector<BoundingBox> grid;
  bool operator()(int idx1,int idx2){
    return grid[idx1].overlap > grid[idx2].overlap;
  }
};
struct CComparator{
  CComparator(const std::vector<float>& _conf):conf(_conf){}
  std::vector<float> conf;
  bool operator()(int idx1,int idx2){
    return conf[idx1]> conf[idx2];
  }
};


class FeaturesTrack{
private:


  cv::PatchGenerator generator;
  myClassifier classifier;
  TrackerKanade tracker;
  ///Parameters
  //int bbox_step;
  int framesInconfidence;
  int min_win;
  int patch_size;
  //initial parameters for positive examples
  int num_closest_init;
  int num_warps_init;
  int noise_init;
  float angle_init;
  float shift_init;
  float scale_init;
  //update parameters for positive examples
  int num_closest_update;
  int num_warps_update;
  int noise_update;
  float angle_update;
  float shift_update;
  float scale_update;
  //parameters for negative examples
  float bad_overlap;
  float bad_patches;
  ///Variables
//Integral Images
  cv::Mat iisum;
  cv::Mat iisqsum;

  float var;
//Training data
  std::vector<std::pair<std::vector<int>,int> > pX; //positive ferns <features,labels=1>
  std::vector<std::pair<std::vector<int>,int> > nX; // negative ferns <features,labels=0>
  cv::Mat pEx;  //positive NN example
  std::vector<cv::Mat> nEx; //negative NN examples
//Test data
  std::vector<std::pair<std::vector<int>,int> > nXT; //negative data to Test
  std::vector<cv::Mat> nExT; //negative NN examples to Test
//Last frame data
  BoundingBox lastbox;
  BoundingBox lastTrackedbox;
  bool lastvalid;
  float lastconf;
//Current frame data
  //Tracker data
  bool tracked;
  BoundingBox tbb;
  bool tvalid;
  float tconf;
  //Detector data
  TempStruct tmp;
  DetStruct dt;
  std::vector<BoundingBox> dbb;
  std::vector<bool> dvalid;
  std::vector<float> dconf;
  bool detected;
  std::vector<cv::Mat> lastFiveFace; //negative NN examples to Test
 // std::vector<std::pair<std::vector<int>,int> > ; //last five face 15x15

  //Bounding Boxes
  std::vector<BoundingBox> grid;
  std::vector<cv::Size> scales;
  std::vector<int> good_boxes; //indexes of bboxes with overlap > 0.6
  std::vector<int> bad_boxes; //indexes of bboxes with overlap < 0.2
  BoundingBox bbhull; // hull of good_boxes
  BoundingBox best_box; // maximum overlapping bbox
  int best_box_index;
  float matchingMean;
  int modelSize;
  ValidateDetection validateDetection;
  //FileNode& fileName;

public:
  //Constructors
  FeaturesTrack();
  FeaturesTrack(const cv::FileNode& file);
  void read(const cv::FileNode& file);
  bool isInternal(CvRect bboxHull, CvRect bbox);
  //Methods
  void init(const cv::Mat& frame1,const cv::Rect &box, FILE* bb_file);
  void generatePositiveData(const cv::Mat& frame, int num_warps);
  void generateNegativeData(const cv::Mat& frame);
  void processFrame(const cv::Mat& img1,const cv::Mat& img2,const cv::Mat& frameTo,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2,BoundingBox& bbnext,CvRect& histogramDetectionBlob,bool& lastboxfound, bool tl,FILE* bb_file);
  //bool processFrameale(const cv::Mat& img1,const cv::Mat& img2,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound, bool tl,FILE* bb_file);
  void track(const cv::Mat& img1, const cv::Mat& img2,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2,bool allP);
  int detect(const cv::Mat& frame,vector<BoundingBox>& cbb,vector<float>& cconf,float nn_th,CvRect bboxToDetect);
  void detectAle(const cv::Mat& frame);
  void clusterConf(const std::vector<BoundingBox>& dbb,const std::vector<float>& dconf,std::vector<BoundingBox>& cbb,std::vector<float>& cconf);
  void clusterConfAle(const std::vector<BoundingBox>& dbb,std::vector<BoundingBox>& cbb);
  void evaluate();
  void learnModel(const Mat& frame,CvRect& pbox);
  void forceLearnModel(const Mat& img);
  int getMin(int a , int b);
  int getMax(int a , int b);
  //Tools
  void buildGridSW(const cv::Mat& img, const cv::Rect& box);
  float bbOverlap(const BoundingBox& box1,const BoundingBox& box2);
  void getOverlappingBoxes(int num_closest);
  void getBBHull();
  void getPatterns(const cv::Mat& img, cv::Mat& pattern,cv::Scalar& mean,cv::Scalar& stdev);
  //void getSmallFace(const cv::Mat& img, cv::Mat& pattern,cv::Scalar& mean,cv::Scalar& stdev);
  //void bbPoints(std::vector<cv::Point2f>& points, const BoundingBox& bb);
  void bbPointsOlD(vector<cv::Point2f>& points,const BoundingBox& bb);
  void bbPoints(std::vector<cv::Point2f>& points, const BoundingBox& bb, const Mat& frameFrom,const Mat& frameTo);
  void bbPointsAllP(std::vector<cv::Point2f>& points, const BoundingBox& bb, const Mat& frameFrom,const Mat& frameTo);
  void bbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,
  const BoundingBox& bb1,BoundingBox& bb2);
  void bbPredictAle(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,
        const BoundingBox& bb1,BoundingBox& bb2,const cv::Mat& img);
  double getVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum);
  bool bbComp(const BoundingBox& bb1,const BoundingBox& bb2);
  int clusterBB(const std::vector<BoundingBox>& dbb,std::vector<int>& indexes);
  bool SurfExtraction(Mat& img_object,Mat& img_objectMultiresolution ,Mat& img_scene,CvRect& pbox,bool goOn);
  void mySift( Mat& img_object, Mat& img_scene);
  bool examineSurf( Mat& img_object, Mat& img_scene,CvRect& pbox);
  bool examineSurfPatch( Mat& img_object, Mat& img_scene,CvRect& pbox);
  bool examineSurfMatch( Mat& img_object, Mat& img_scene);
  bool myTrk(Mat& frameFrom, Mat& frameTo,CvRect& pbox);
  void InitLastFiveFace(Mat& frame,CvRect& pbox,const FileNode& file);
  void realignFace(Mat& frame,CvRect& pbox,CvRect& pboxHull);
  float getMatching(Mat& face);
  void updateNewFace(Mat& face,float matchV);
  void updateNewFaceWithResize(Mat& face);
  float getMatchMean();
  float getModelSize();
  bool getSurfMatching(Mat& frame);
  void showExamples();
};

