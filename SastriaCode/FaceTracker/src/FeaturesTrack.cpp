

/*
 * FeaturesTrack.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 *      revised: Alessandro Romanino
 *
 */

#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "FeaturesTrack.h"
//#include <opencv2/legacy/legacy.hpp>
#define MINBBOX 15


FeaturesTrack::FeaturesTrack()
{
}

FeaturesTrack::FeaturesTrack(const FileNode& file){
  read(file);
  //fileName=file;
}

void FeaturesTrack::read(const FileNode& file){
  ///Bounding Box Parameters
  //fileName=file;
  min_win = (int)file["min_win"];
  ///Genarator Parameters
  //initial parameters for positive examples
  patch_size = (int)file["patch_size"];
  num_closest_init = (int)file["num_closest_init"];
  num_warps_init = (int)file["num_warps_init"];
  noise_init = (int)file["noise_init"];
  angle_init = (float)file["angle_init"];
  shift_init = (float)file["shift_init"];
  scale_init = (float)file["scale_init"];
  //update parameters for positive examples
  num_closest_update = (int)file["num_closest_update"];
  num_warps_update = (int)file["num_warps_update"];
  noise_update = (int)file["noise_update"];
  angle_update = (float)file["angle_update"];
  shift_update = (float)file["shift_update"];
  scale_update = (float)file["scale_update"];
  //parameters for negative examples
  bad_overlap = (float)file["overlap"];
  bad_patches = (int)file["num_patches"];
  classifier.read(file);
}

void FeaturesTrack::init(const Mat& frame1,const Rect& box,FILE* bb_file){
  //bb_file = fopen("bounding_boxes.txt","w");
  //Get Bounding Boxes
  framesInconfidence=0;
  buildGridSW(frame1,box);
  printf("Created %d bounding boxes\n",(int)grid.size());
  ///Preparation
  //allocation
  iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);
  iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
  dconf.reserve(100);
  dbb.reserve(100);
 // bbox_step =7;
  //tmp.conf.reserve(grid.size());

  //Detector Data
  tmp.conf = vector<float>(grid.size());
  tmp.patt = vector<vector<int> >(grid.size(),vector<int>(classifier.getNumStructs(),0));

  //tmp.patt.reserve(grid.size());
  dt.bb.reserve(grid.size());
  good_boxes.reserve(grid.size());
  bad_boxes.reserve(grid.size());
  pEx.create(patch_size,patch_size,CV_64F);
  //Init Generator
  generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);
  getOverlappingBoxes(num_closest_init);
  printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
  printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
  printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);

  /*
  cout << "BOX" << endl;
  for (int i=0;i<good_boxes.size();++i)
	  cout << i << ": " << grid[good_boxes[i]].overlap << " scale: " << grid[good_boxes[i]].sidx << endl;
*/

  //Correct Bounding Box
  lastbox=best_box;
  lastconf=1;
  lastvalid=true;
  //Print
  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  //Prepare Classifier
  classifier.prepare(scales);
  ///Generate Data
  // Generate positive data
  generatePositiveData(frame1,num_warps_init);
  // Set variance threshold
  Scalar stdev, mean;
  meanStdDev(frame1(best_box),mean,stdev);
  integral(frame1,iisum,iisqsum);
  var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);
  cout << "variance: " << var << endl;
  //check variance
  double vr =  getVar(best_box,iisum,iisqsum)*0.5;
  cout << "check variance: " << vr << endl;
  //cvWaitKey(0);
  // Generate negative data
  generateNegativeData(frame1);
  //Split Negative Ferns into Training and Testing sets (they are already shuffled)
  int half = (int)nX.size()*0.5f;
  nXT.assign(nX.begin()+half,nX.end());
  nX.resize(half);
  ///Split Negative NN Examples into Training and Testing sets
  half = (int)nEx.size()*0.5f;
  nExT.assign(nEx.begin()+half,nEx.end());
  nEx.resize(half);
  //Merge Negative Data with Positive Data and shuffle it
  vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
  vector<int> idx = index_shuffle(0,ferns_data.size());
  int a=0;

  for (int i=0;i<pX.size();i++){
      ferns_data[idx[a]] = pX[i];
      a++;
  }
  for (int i=0;i<nX.size();i++){
      ferns_data[idx[a]] = nX[i];
      a++;
  }
  //Data already have been shuffled, just putting it in the same vector
  vector<cv::Mat> nn_data(nEx.size()+1);
  nn_data[0] = pEx;
  for (int i=0;i<nEx.size();i++){
      nn_data[i+1]= nEx[i];
  }
  ///Training
  classifier.trainFern(ferns_data,2); //bootstrap = 2
  classifier.trainNNN(nn_data);
  ///Threshold Evaluation on testing sets
  //classifier.evaluateThV(nXT,nExT);
}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void FeaturesTrack::generatePositiveData(const Mat& frame, int num_warps){
  Scalar mean;
  Scalar stdev;
  IntegralImage integralPatch;

  getPatterns(frame(best_box),pEx,mean,stdev);

  //Get Fern features on warped patches
  Mat img;
  Mat warped;
  GaussianBlur(frame,img,Size(9,9),1.5);
 // frame.copyTo(img);
  warped = img(bbhull);
  RNG& rng = theRNG();
  // bbhull center
  Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);
  vector<int> fern(classifier.getNumStructs());
  pX.clear();
  Mat patch;
  if (pX.capacity()<num_warps*good_boxes.size())
    pX.reserve(num_warps*good_boxes.size());
  int idx;
  for (int i=0;i<num_warps;i++){
     if (i>0)
       generator(frame,pt,warped,bbhull.size(),rng);

     	// imshow("FRAME", frame);
     	 //imshow("IMG", img);
     	 //imshow("WarP", warped);
     	 //cvWaitKey(0);

       for (int b=0;b<good_boxes.size();b++){
    	 //cout << endl << "good_boxes: " << b << " ";
         idx=good_boxes[b];
		 patch = img(grid[idx]);
		 integralPatch=IntegralImage(patch);
		// imshow("New Features",frame);
		 //cout << endl << "goodBox N° " << b;
         classifier.getFeatures(patch,integralPatch,grid[idx].sidx,fern);
         pX.push_back(make_pair(fern,1));
         //for(int t=0;t<pX[i].first.size();++t)
        	//  cout << fern[t] << "-";
     }
    // imshow("ImgTransform",img);
    // cvWaitKey(0);
  }

  /*
  for(int i=0; i<pX.size();++i){
	  cout << endl << "good_boxes: " << i << " ";
	  for(int t=0;t<pX[i].first.size();++t)
	  cout << pX[i].first[t] << "-";
  }
*/
//  cvWaitKey(0);

  printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}

void FeaturesTrack::getPatterns(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
  //Output: resized Zero-Mean patch
  resize(img,pattern,Size(patch_size,patch_size));
  meanStdDev(pattern,mean,stdev);
  pattern.convertTo(pattern,CV_32F);
  pattern = pattern-mean.val[0];
}



void FeaturesTrack::generateNegativeData(const Mat& frame){
/* Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 */
  IntegralImage integralPatch;
  random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
  int idx;
  //Get Fern Features of the boxes with big variance (calculated using integral images)
  int a=0;
  //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
  printf("negative data generation started.\n");
  vector<int> fern(classifier.getNumStructs());
  nX.reserve(bad_boxes.size());
  Mat patch;
  for (int j=0;j<bad_boxes.size();j++){
	 // cout << endl << "index " << j;
      idx = bad_boxes[j];
          if (getVar(grid[idx],iisum,iisqsum)<var*0.5f)
            continue;
      patch =  frame(grid[idx]);
     // imshow("Patch Negative",patch);
      integralPatch=IntegralImage(patch);
	  classifier.getFeatures(patch,integralPatch,grid[idx].sidx,fern);
      nX.push_back(make_pair(fern,0));
      a++;
  }
  printf("Negative examples generated: ferns: %d ",a);
  //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
  Scalar dum1, dum2;
  bad_patches=bad_boxes.size() > bad_patches ? bad_patches : bad_boxes.size();
  nEx=vector<Mat>(bad_patches);
  for (int i=0;i<bad_patches;i++){
      idx=bad_boxes[i];
	  patch = frame(grid[idx]);
	  //imshow("PatchExample",patch);
      getPatterns(patch,nEx[i],dum1,dum2);
  }
  printf("NN: %d\n",(int)nEx.size());
}

double FeaturesTrack::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  //cout << "MEAN: " << mean << " SQMEAN: " << sqmean << " returnValue:" << sqmean-mean*mean << endl;
  return sqmean-mean*mean;
}

/*
void FeaturesTrack::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound, bool tl, FILE* bb_file){

 // processFrameale(img1.clone(),img2.clone(),points1,points2,bbnext,lastboxfound,tl,bb_file);
  vector<BoundingBox> cbb;
  vector<float> cconf;
  int confident_detections=0;
  int didx; //detection index
  Mat printFrame=img2.clone();
  Mat frameBox=img2.clone();
  float bestConf=0;
  ///Track

  if(lastboxfound && tl){
      track(img1,img2,points1,points2);
  }
  else{
      tracked = false;
  }
  ///Detect
  detect(img2);
  ///Integration
  if (tracked){
      bbnext=tbb;
      lastconf=tconf;
      lastvalid=tvalid;
      printf("Tracked\n");
      if(detected){                                               //   if Detected
          clusterConf(dbb,dconf,cbb,cconf);                       //   cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          for (int i=0;i<cbb.size();i++){
              if (bbOverlap(tbb,cbb[i])<0.6 && cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
                  confident_detections++;
                  if(bestConf<cconf[i]){
                  didx=i; //detection index
                  bestConf=cconf[i];
                  }
                  drawBox(printFrame,cbb[i],0.5,1);
              }
          }

          if (confident_detections>=1){                                //if there is ONE such a cluster, re-initialize the tracker
              printf("Found a better match..reinitializing tracking\n");
              bbnext=cbb[didx];
              lastconf=cconf[didx];
              lastvalid=false;
              printf("\n\rConfident<= %d\n\r",confident_detections);
              printf("box Found=%d\n\r",cbb.size());
              drawBox(printFrame,bbnext,1,2);
              imshow("clusterconf",printFrame);

            //  if(confident_detections>=1)
            	 // cvWaitKey(0);
          }
          else {
              printf("%d confident cluster was found\n",confident_detections);
              int cx=0,cy=0,cw=0,ch=0;
              int close_detections=0;
              for (int i=0;i<cbb.size();i++){
                  if(bbOverlap(tbb,cbb[i])>0.7){                     // Get mean of close detections
                      cx += cbb[i].x;
                      cy +=cbb[i].y;
                      cw += cbb[i].width;
                      ch += cbb[i].height;
                      close_detections++;
                      printf("weighted detection: %d %d %d %d\n",cbb[i].x,cbb[i].y,cbb[i].width,cbb[i].height);
                  }
              }
              if (close_detections>0){
                  bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
                  bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                  bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                  bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                  printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                  printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                  printf("Weighting %d close detection(s) with tracker..\n",close_detections);
              }
              else{
                printf("%d close detections were found\n",close_detections);
              }
          }
      }
  }
  else{                                       //   If NOT tracking
      printf("Not tracking..\n");
      lastboxfound = false;
      lastvalid = false;
      if(detected){                           //  and detector is defined
          clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
          printf("Found %d clusters\n",(int)cbb.size());

          if (cconf.size()==1){
              bbnext=cbb[0];
              lastconf=cconf[0];
              printf("Confident detection..reinitializing tracker\n");
              lastboxfound = true;
          }
      }
  }
  lastbox=bbnext;
  if (lastboxfound)
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  else
    fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
  if (lastvalid && tl)
    learn(img2);
}

*/

int FeaturesTrack::getMin(int a, int b){

	return(a<b ? a: b);

}

int FeaturesTrack::getMax(int a, int b){

	return(a>b ? a: b);

}



void FeaturesTrack::mySift( Mat& img_object, Mat& img_scene){



	  //-- Step 1: Detect the keypoints using SURF Detector
	  int minHessian = 400;
	  int x,y;

	  SiftFeatureDetector detector( 0.04, 10 );

	  std::vector<KeyPoint> keypoints_object, keypoints_scene;

	  detector.detect( img_object, keypoints_object );
	  detector.detect( img_scene, keypoints_scene );

	  //-- Step 2: Calculate descriptors (feature vectors)
	  SiftDescriptorExtractor extractor;

	  Mat descriptors_object, descriptors_scene;

	  extractor.compute( img_object, keypoints_object, descriptors_object );
	  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcher;
	  std::vector< DMatch > matches;
	  matcher.match( descriptors_object, descriptors_scene, matches );

	  double max_dist = 0; double min_dist = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
	  }

	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matches;

	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( matches[i].distance < 3*min_dist )
	     { good_matches.push_back( matches[i]); }
	  }
/*
	  Mat img_matches;
	  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
	               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	               */

	  //-- Localize the object
	  std::vector<Point2f> obj;
	  std::vector<Point2f> scene;

	  for( int i = 0; i < good_matches.size(); i++ )
	  {
	    //-- Get the keypoints from the good matches
	    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
	    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	  }

	  Mat H = findHomography( obj, scene, CV_RANSAC );

	  //-- Get the corners from the image_1 ( the object to be "detected" )
	  std::vector<Point2f> obj_corners(4);
	  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
	  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
	  std::vector<Point2f> scene_corners(4);

	  perspectiveTransform( obj_corners, scene_corners, H);

	  y=(scene_corners[0].y+ scene_corners[1].y)/2;
	  scene_corners[0].y=y;
	  scene_corners[1].y=y;

	  x=(scene_corners[0].x+ scene_corners[3].x)/2;
	  scene_corners[0].x=x;
	  scene_corners[3].x=x;

	  x=(scene_corners[1].x+ scene_corners[2].x)/2;
	  scene_corners[1].x=x;
	  scene_corners[2].x=x;

	  y=(scene_corners[2].y+ scene_corners[3].y)/2;
	  scene_corners[2].y=y;
	  scene_corners[3].y=y;

	  //-- Draw lines between the corners (the mapped object in the scene - image_2 )

	  line( img_scene, scene_corners[0] , scene_corners[1], Scalar(125), 4 );
	  line( img_scene, scene_corners[1] , scene_corners[2], Scalar(125), 4 );
	  line( img_scene, scene_corners[2] , scene_corners[3], Scalar(125), 4 );
	  line( img_scene, scene_corners[3] , scene_corners[0], Scalar(125), 4 );

	  //-- Show detected matches
	  imshow( "Good Matches", img_scene );


}


void FeaturesTrack::InitLastFiveFace(Mat& frame,CvRect& pbox,const FileNode& file){

	//Scalar dummyS,dummyM;
	//Mat newFace15;
	//getPatterns(face,newFace15,dummyM,dummyS);
	//lastFiveFace.clear();
	//lastFiveFace.push_back(newFace15);
	//matchingMean=1;
	//modelSize=1;
	classifier.read(file);
	learnModel(frame,pbox);

}

bool FeaturesTrack::examineSurf( Mat& img_object, Mat& img_scene,CvRect& pbox){



	 float percentage=0.25; //0.15
	 int wPercentage=pbox.width*percentage;
	 int hPercentage=pbox.height*percentage;
	 CvRect nextBox;

	 nextBox.x=pbox.x-(wPercentage) > 0  ? pbox.x-(wPercentage) : 0;
	 nextBox.y=pbox.y-(hPercentage) > 0 ? pbox.y-(hPercentage): 0;
	 nextBox.width=pbox.width +(2*wPercentage);
	 nextBox.width=(nextBox.width + nextBox.x) < img_scene.cols ? nextBox.width : (img_scene.cols-nextBox.x);
	 nextBox.height=pbox.height +(2*hPercentage);
	 nextBox.height=(nextBox.height + nextBox.y) < img_scene.rows ? nextBox.height : (img_scene.rows-nextBox.y);

	 bool returnV=true;

	 //-- Step 1: Detect the keypoints using SURF Detector
	 int minHessian = 400;
	 int x,y;

	 SurfFeatureDetector detectorSurf( minHessian );

	 std::vector<KeyPoint> keypoints_object, keypoints_scene;

	 detectorSurf.detect( img_object, keypoints_object );
	 detectorSurf.detect( img_scene, keypoints_scene );

	 //-- Step 2: Calculate descriptors (feature vectors)
	 SurfDescriptorExtractor extractorSurf;

	  Mat descriptors_object, descriptors_scene;

	  extractorSurf.compute( img_object, keypoints_object, descriptors_object );
	  extractorSurf.compute( img_scene, keypoints_scene, descriptors_scene );


	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcherSurf;
	  std::vector< DMatch > matchesSurf;
	  matcherSurf.match( descriptors_object, descriptors_scene, matchesSurf );

	  double max_distSURF = 0; double min_distSURF = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  double distSurf;
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  {
		distSurf = matchesSurf[i].distance;
	    if( distSurf < min_distSURF ) min_distSURF = distSurf;
	    if( distSurf > max_distSURF ) max_distSURF = distSurf;
	  }

	  printf("-- Max dist : %f \n", max_distSURF );
	  printf("-- Min dist : %f \n", min_distSURF );


	  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matchesSurf;

	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( matchesSurf[i].distance < 3*min_distSURF )
	     { good_matchesSurf.push_back( matchesSurf[i]); }
	  }


	  if(good_matchesSurf.size()>=8){
		  cout<< endl << "MachSurf: " << good_matchesSurf.size();
		  //-- Localize the object
		  std::vector<Point2f> obj;
		  std::vector<Point2f> scene;

		  for( int i = 0; i < good_matchesSurf.size(); i++ )
		  {
			  //-- Get the keypoints from the good matches
			  obj.push_back( keypoints_object[ good_matchesSurf[i].queryIdx ].pt );
			  scene.push_back( keypoints_scene[ good_matchesSurf[i].trainIdx ].pt );
		  }

		  Mat H = findHomography( obj, scene, CV_RANSAC );

		  //-- Get the corners from the image_1 ( the object to be "detected" )
		  std::vector<Point2f> obj_corners(4);
		  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
		  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
		  std::vector<Point2f> scene_corners(4);

		  perspectiveTransform( obj_corners, scene_corners, H);

		  y=(scene_corners[0].y+ scene_corners[1].y)/2;
		  y=y>0?y:0;
		  scene_corners[0].y=y;
		  scene_corners[1].y=y;
		  pbox.y=y;

		  x=(scene_corners[0].x+ scene_corners[3].x)/2;
		  x=x>0?x:0;
		  scene_corners[0].x=x;
		  scene_corners[3].x=x;
		  pbox.x=x;

		  x=(scene_corners[1].x+ scene_corners[2].x)/2;
		  x= x<(img_scene.cols-1) ? x: (img_scene.cols-1) ;
		  scene_corners[1].x=x;
		  scene_corners[2].x=x;
		  pbox.width=x-pbox.x;

		  y=(scene_corners[2].y+ scene_corners[3].y)/2;
		  y=y<(img_scene.rows-1) ? y: (img_scene.rows-1) ;
		  scene_corners[2].y=y;
		  scene_corners[3].y=y;
		  pbox.height=y-pbox.y;

		  //-- Draw lines between the corners (the mapped object in the scene - image_2 )



		  if((isInternal(nextBox,pbox))&& (pbox.width>0)&& (pbox.height>0)){
			  line( img_scene, scene_corners[0] , scene_corners[1], Scalar(255), 4 );
			  line( img_scene, scene_corners[1] , scene_corners[2], Scalar(255), 4 );
			  line( img_scene, scene_corners[2] , scene_corners[3], Scalar(255), 4 );
			  line( img_scene, scene_corners[3] , scene_corners[0], Scalar(255), 4 );
		  } else goto NOTRK;


		  //-- Show detected matches


		  // waitKey(0);
	  }else{
		  NOTRK:
		  returnV=false;
		  cout<< endl << "WARNING MachSurf: " << good_matchesSurf.size();
	  }
	  imshow( "Good Matches", img_scene );
	 return(returnV);
}


bool FeaturesTrack::examineSurfPatch( Mat& img_object, Mat& img_scene,CvRect& pbox){



	 bool returnV=true;

	 //-- Step 1: Detect the keypoints using SURF Detector
	 int minHessian = 400;
	 int x,y;

	 SurfFeatureDetector detectorSurf( minHessian );

	 std::vector<KeyPoint> keypoints_object, keypoints_scene;

	 detectorSurf.detect( img_object, keypoints_object );
	 detectorSurf.detect( img_scene, keypoints_scene );

	 //-- Step 2: Calculate descriptors (feature vectors)
	 SurfDescriptorExtractor extractorSurf;

	  Mat descriptors_object, descriptors_scene;

	  extractorSurf.compute( img_object, keypoints_object, descriptors_object );
	  extractorSurf.compute( img_scene, keypoints_scene, descriptors_scene );


	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcherSurf;
	  std::vector< DMatch > matchesSurf;
	  matcherSurf.match( descriptors_object, descriptors_scene, matchesSurf );

	  double max_distSURF = 0; double min_distSURF = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  double distSurf;
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  {
		distSurf = matchesSurf[i].distance;
	    if( distSurf < min_distSURF ) min_distSURF = distSurf;
	    if( distSurf > max_distSURF ) max_distSURF = distSurf;
	  }

	  printf("-- Max dist : %f \n", max_distSURF );
	  printf("-- Min dist : %f \n", min_distSURF );


	  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matchesSurf;

	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( matchesSurf[i].distance < 3*min_distSURF )
	     { good_matchesSurf.push_back( matchesSurf[i]); }
	  }


	  if(good_matchesSurf.size()>=8){
		  cout<< endl << "MachSurf: " << good_matchesSurf.size();
		  //-- Localize the object
		  std::vector<Point2f> obj;
		  std::vector<Point2f> scene;

		  for( int i = 0; i < good_matchesSurf.size(); i++ )
		  {
			  //-- Get the keypoints from the good matches
			  obj.push_back( keypoints_object[ good_matchesSurf[i].queryIdx ].pt );
			  scene.push_back( keypoints_scene[ good_matchesSurf[i].trainIdx ].pt );
		  }

		  Mat H = findHomography( obj, scene, CV_RANSAC );

		  //-- Get the corners from the image_1 ( the object to be "detected" )
		  std::vector<Point2f> obj_corners(4);
		  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
		  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
		  std::vector<Point2f> scene_corners(4);

		  perspectiveTransform( obj_corners, scene_corners, H);

		  y=(scene_corners[0].y+ scene_corners[1].y)/2;
		  y=y>0?y:0;
		  scene_corners[0].y=y;
		  scene_corners[1].y=y;
		  pbox.y=y;

		  x=(scene_corners[0].x+ scene_corners[3].x)/2;
		  x=x>0?x:0;
		  scene_corners[0].x=x;
		  scene_corners[3].x=x;
		  pbox.x=x;

		  x=(scene_corners[1].x+ scene_corners[2].x)/2;
		  x= x<(img_scene.cols-1) ? x: (img_scene.cols-1) ;
		  scene_corners[1].x=x;
		  scene_corners[2].x=x;
		  pbox.width=x-pbox.x;

		  y=(scene_corners[2].y+ scene_corners[3].y)/2;
		  y=y<(img_scene.rows-1) ? y: (img_scene.rows-1) ;
		  scene_corners[2].y=y;
		  scene_corners[3].y=y;
		  pbox.height=y-pbox.y;

		  //-- Draw lines between the corners (the mapped object in the scene - image_2 )




			  line( img_scene, scene_corners[0] , scene_corners[1], Scalar(255), 4 );
			  line( img_scene, scene_corners[1] , scene_corners[2], Scalar(255), 4 );
			  line( img_scene, scene_corners[2] , scene_corners[3], Scalar(255), 4 );
			  line( img_scene, scene_corners[3] , scene_corners[0], Scalar(255), 4 );

			  imshow( "Good Matches", img_scene );

		  //-- Show detected matches


		  // waitKey(0);
	  }else{

		  returnV=false;
		  cout<< endl << "WARNING MachSurf: " << good_matchesSurf.size();
	  }

	 return(returnV);
}





/*
bool FeaturesTrack::mySurf( Mat& img_object,Mat& img_objectMultiresolution, Mat& img_scene,CvRect& pbox){

	bool returnV=true;
	CvRect oldPbox=cvRect(pbox.x,pbox.y,pbox.width,pbox.height);
	CvRect tmpPbox=cvRect(0,0,0,0);
	CvRect segmetedBox=cvRect(0,0,img_object.cols/2,img_object.rows/2);
	Mat tmpObj;
	Mat ims;
	int fraction=10;
	//Mat imo;


	if(!examineSurf(img_object,img_scene,pbox)){
		returnV=false;
		for(int count=0;count<4 && !returnV;++count){
			tmpPbox.x=oldPbox.x+(oldPbox.width/fraction);
			tmpPbox.y=oldPbox.y+(oldPbox.height/fraction);
			tmpPbox.width=oldPbox.width-(2*(oldPbox.width/fraction));
			tmpPbox.height=oldPbox.height-(2*(oldPbox.height/fraction));
			segmetedBox.x=(oldPbox.width/fraction);
			segmetedBox.y=(oldPbox.height/fraction);
			segmetedBox.width=tmpPbox.width;
			segmetedBox.height=tmpPbox.height;
			img_object(segmetedBox).copyTo(tmpObj);
			img_scene.copyTo(ims);
			rectangle(ims,tmpPbox,cvScalar(255,0,0),2);
		    rectangle(ims,pbox,cvScalar(0,255,0),1);
			imshow("scena",ims);
			rectangle(img_object,segmetedBox,cvScalar(0,255,0),2);
			imshow("realObj",img_object);
			imshow("obj",tmpObj);
			//waitKey(0);
			if(examineSurf(tmpObj,img_scene,tmpPbox)){
				returnV=true;
				pbox.x=tmpPbox.x-(tmpPbox.width/fraction);
				pbox.y=tmpPbox.y-(tmpPbox.height/fraction);
				pbox.width=tmpPbox.width+(2*(tmpPbox.width/fraction));
				pbox.height=tmpPbox.height+(2*(tmpPbox.height/fraction));
				img_scene.copyTo(ims);
				rectangle(ims,tmpPbox,cvScalar(255,0,0),2);
			    rectangle(ims,pbox,cvScalar(0,255,0),1);
				imshow("scenadopo",ims);
			}
			fraction-=2;
		}
	}

	 return(returnV);
}
*/
/*
float FeaturesTrack::getMatching(Mat& face){
	Mat ncc(1,1,CV_32F);
	float nccP=0;
	//float bestNccp=0;
	int size=lastFiveFace.size();

	for(int i=0;i<size;++i){
		matchTemplate(face,lastFiveFace[i],ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
		nccP+=(((float*)ncc.data)[0]+1)*0.5; //rescale interval -1:1 to 0:1

	}


	nccP/=size;
	cout << endl << "SIMILARITY: " << nccP << " Size" << size;
	cout << endl;

	return(nccP);
}
*/
float FeaturesTrack::getMatching(Mat& face){
	/*Mat ncc(1,1,CV_32F);
	float nccP=0;
	float bestNccp=0;
	int size=lastFiveFace.size();

	for(int i=0;i<size;++i){
		matchTemplate(face,lastFiveFace[i],ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
		nccP=(((float*)ncc.data)[0]+1)*0.5; //rescale interval -1:1 to 0:1
		if(nccP>bestNccp)
			bestNccp=nccP;
	}


	cout << endl << "SIMILARITY: " << bestNccp << " Size" << size;
	cout << endl;
*/
	float conf,dummy;
	vector<int> isin;
	classifier.NNConf(face,isin,conf,dummy);
	return(conf);
}


bool FeaturesTrack::examineSurfMatch( Mat& img_object, Mat& img_scene){



	 bool returnV=true;

	 //-- Step 1: Detect the keypoints using SURF Detector
	 int minHessian = 400;
	 int x,y;

	 SurfFeatureDetector detectorSurf( minHessian );

	 std::vector<KeyPoint> keypoints_object, keypoints_scene;

	 detectorSurf.detect( img_object, keypoints_object );
	 detectorSurf.detect( img_scene, keypoints_scene );

	 //-- Step 2: Calculate descriptors (feature vectors)
	 SurfDescriptorExtractor extractorSurf;

	  Mat descriptors_object, descriptors_scene;

	  extractorSurf.compute( img_object, keypoints_object, descriptors_object );
	  extractorSurf.compute( img_scene, keypoints_scene, descriptors_scene );


	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcherSurf;
	  std::vector< DMatch > matchesSurf;
	  matcherSurf.match( descriptors_object, descriptors_scene, matchesSurf );

	  double max_distSURF = 0; double min_distSURF = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  double distSurf;
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  {
		distSurf = matchesSurf[i].distance;
	    if( distSurf < min_distSURF ) min_distSURF = distSurf;
	    if( distSurf > max_distSURF ) max_distSURF = distSurf;
	  }

	  printf("-- Max dist : %f \n", max_distSURF );
	  printf("-- Min dist : %f \n", min_distSURF );


	  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matchesSurf;

	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( matchesSurf[i].distance < 3*min_distSURF )
	     { good_matchesSurf.push_back( matchesSurf[i]); }
	  }

	  cout<< endl << "MachSurf: " << good_matchesSurf.size();
	  if(good_matchesSurf.size()>=4){

		  //-- Localize the object
		  std::vector<Point2f> obj;
		  std::vector<Point2f> scene;

		  for( int i = 0; i < good_matchesSurf.size(); i++ )
		  {
			  //-- Get the keypoints from the good matches
			  obj.push_back( keypoints_object[ good_matchesSurf[i].queryIdx ].pt );
			  scene.push_back( keypoints_scene[ good_matchesSurf[i].trainIdx ].pt );
		  }

		  Mat H = findHomography( obj, scene, CV_RANSAC );

		  //-- Get the corners from the image_1 ( the object to be "detected" )
		  std::vector<Point2f> obj_corners(4);
		  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
		  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
		  std::vector<Point2f> scene_corners(4);

		  perspectiveTransform( obj_corners, scene_corners, H);

		  y=(scene_corners[0].y+ scene_corners[1].y)/2;
		  y=y>0?y:0;
		  scene_corners[0].y=y;
		  scene_corners[1].y=y;
		  //pbox.y=y;

		  x=(scene_corners[0].x+ scene_corners[3].x)/2;
		  x=x>0?x:0;
		  scene_corners[0].x=x;
		  scene_corners[3].x=x;
		//  pbox.x=x;

		  x=(scene_corners[1].x+ scene_corners[2].x)/2;
		  x= x<(img_scene.cols-1) ? x: (img_scene.cols-1) ;
		  scene_corners[1].x=x;
		  scene_corners[2].x=x;
		//  pbox.width=x-pbox.x;

		  y=(scene_corners[2].y+ scene_corners[3].y)/2;
		  y=y<(img_scene.rows-1) ? y: (img_scene.rows-1) ;
		  scene_corners[2].y=y;
		  scene_corners[3].y=y;
		//  pbox.height=y-pbox.y;

		  //-- Draw lines between the corners (the mapped object in the scene - image_2 )




	  }else{
		  returnV=false;
		  cout<< endl << "WARNING MachSurf: " << good_matchesSurf.size();
	  }

	 imshow( "Good Matches SE", img_scene );
	 return(returnV);
}


bool FeaturesTrack::getSurfMatching(Mat& frame){

	int size=lastFiveFace.size();
	bool returnV=false;
	Mat ex,tmp;
	double minval;

	imshow("frame",frame);
	waitKey(0);

		for(int i=0;i<size;++i){
			 minMaxLoc(lastFiveFace[i],&minval);
			 lastFiveFace[i].copyTo(ex);
			 ex = ex-minval;
			 ex.convertTo(tmp,CV_8U);
			 imshow("ex",tmp);
			 waitKey(0);
			if(examineSurfMatch(tmp,frame)){
				i=size+1;
				returnV=true;
			}
		}

		return(returnV);
}


bool FeaturesTrack::myTrk(Mat& frameFrom, Mat& frameTo,CvRect& pbox){

	bool returnV=true;
	vector<Point2f> pointsFrom;
	vector<Point2f> pointsTo;
	TrackerKanade tracker;
	int x,y;
	float percentage=0.40; //0.15 //0.40
	int wPercentage=pbox.width*percentage;
	int hPercentage=pbox.height*percentage;
	Mat resutl;
//	CvRect bboxTmp(pbox);
	//CvRect oldBBox=cvRect(pbox.x,pbox.y,pbox.width,pbox.height);
	Mat imgObj,imgScene,imgSceneCopy;

	frameFrom(pbox).copyTo(imgObj);


	CvRect nextBox;

	nextBox.x=pbox.x-(wPercentage) > 0  ? pbox.x-(wPercentage) : 0;
	nextBox.y=pbox.y-(hPercentage) > 0 ? pbox.y-(hPercentage): 0;
	nextBox.width=pbox.width +(2*wPercentage);
	nextBox.width=(nextBox.width + nextBox.x) < frameTo.cols ? nextBox.width : (frameTo.cols-nextBox.x);
	nextBox.height=pbox.height +(2*hPercentage);
	nextBox.height=(nextBox.height + nextBox.y) < frameTo.rows ? nextBox.height : (frameTo.rows-nextBox.y);

	frameTo(nextBox).copyTo(imgScene);

	bbPointsOlD(pointsFrom,(BoundingBox)pbox);


	if(tracker.trackf2f(frameFrom,frameTo,pointsFrom,pointsTo)){

		Mat H = findHomography( pointsFrom, pointsTo, CV_RANSAC );

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(pbox.x,pbox.y); obj_corners[1] = cvPoint( pbox.x+pbox.width, pbox.y );
		obj_corners[2] = cvPoint( pbox.x+pbox.width, pbox.y+pbox.height ); obj_corners[3] = cvPoint( pbox.x, pbox.y+pbox.height );
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform( obj_corners, scene_corners, H);

		y=(scene_corners[0].y+ scene_corners[1].y)/2;
		y=y>0?y:0;
		scene_corners[0].y=y;
		scene_corners[1].y=y;
		pbox.y=y;

		x=(scene_corners[0].x+ scene_corners[3].x)/2;
		x=x>0?x:0;
		scene_corners[0].x=x;
		scene_corners[3].x=x;
		pbox.x=x;

		x=(scene_corners[1].x+ scene_corners[2].x)/2;
		x= x<(frameTo.cols-1) ? x: (frameTo.cols-1) ;
		scene_corners[1].x=x;
		scene_corners[2].x=x;
		pbox.width=x-pbox.x;

		y=(scene_corners[2].y+ scene_corners[3].y)/2;
		y=y<(frameTo.rows-1) ? y: (frameTo.rows-1) ;
		scene_corners[2].y=y;
		scene_corners[3].y=y;
		pbox.height=y-pbox.y;



		frameTo.copyTo(resutl);

		if((isInternal(nextBox,pbox))&& (pbox.width>0)&& (pbox.height>0)){
				line( resutl, scene_corners[0] , scene_corners[1], Scalar(255), 4 );
				line( resutl, scene_corners[1] , scene_corners[2], Scalar(255), 4 );
				line( resutl, scene_corners[2] , scene_corners[3], Scalar(255), 4 );
				line( resutl, scene_corners[3] , scene_corners[0], Scalar(255), 4 );
				realignFace(frameTo,pbox,nextBox);
				rectangle(resutl,pbox,cvScalar(0),2);
				//imshow("imgscene",imgScene);
				//waitKey(0);
				returnV=true;
				imshow("result",resutl);

			} else returnV=false;
		} else returnV=false;

//waitKey(0);
	 return(returnV);
}




void FeaturesTrack::realignFace(Mat& frame,CvRect& pbox,CvRect& pboxHull){

	CvRect workBox=cvRect(pbox.x,pbox.y,pbox.width,pbox.height);
	CvRect bestBB=cvRect(pbox.x,pbox.y,pbox.width,pbox.height);
	Mat face15,face;
	Scalar dummyS,dummyM;
	float matchValue;
	float bestMatch=0;


	float percentage=0.10; //0.15 //0.40
	int wPercentage=pbox.width*percentage;
	int hPercentage=pbox.height*percentage;

	pboxHull.x=pbox.x-(wPercentage) > 0  ? pbox.x-(wPercentage) : 0;
	pboxHull.y=pbox.y-(hPercentage) > 0 ? pbox.y-(hPercentage): 0;
	pboxHull.width=pbox.width +(2*wPercentage);
	pboxHull.width=(pboxHull.width + pboxHull.x) < frame.cols ? pboxHull.width : (frame.cols-pboxHull.x);
	pboxHull.height=pbox.height +(2*hPercentage);
	pboxHull.height=(pboxHull.height + pboxHull.y) < frame.rows ? pboxHull.height : (frame.rows-pboxHull.y);




	int xEnd= (pboxHull.width - pbox.width);
	int yEnd= (pboxHull.height - pbox.height);

	RNG& rng = theRNG();

	frame(pbox).copyTo(face);
	getPatterns(face,face15,dummyM,dummyS);
	bestMatch=getMatching(face15);

	for(int x=0; x<200;++x)
			workBox.x=(((float)rng))*xEnd + pboxHull.x;
			workBox.y=(((float)rng))*yEnd + pboxHull.y;
			frame(workBox).copyTo(face);
			getPatterns(face,face15,dummyM,dummyS);
			matchValue=getMatching(face15);
			if(matchValue>bestMatch){
				bestMatch=matchValue;
				bestBB.x=workBox.x;
				bestBB.y=workBox.y;
				bestBB.height=workBox.width;
				bestBB.width=workBox.height;
				//waitKey(0);
		}

			pbox.x=(bestBB.x+pbox.x)/2;
			pbox.y=(bestBB.y+pbox.y)/2;

}

bool FeaturesTrack::SurfExtraction( Mat& img_object,Mat& img_objectMultiresolution, Mat& img_scene,CvRect& pbox,bool goOn){

	bool returnV=true;

	Mat tmpObj;
	Mat ims;
	int xOff,yOff,xMov,yMov,wMov,hMov;
	Mat oldFace,newFace,oldFace15,newFace15;
	Mat ncc(1,1,CV_32F);
	float nccP,matchValue;
	Scalar dummyM,dummyS;

	 RNG& rng = theRNG();

	 CvSize percentageSize=cvSize((pbox.width/4),(pbox.height/4));
	 xMov=(pbox.x-percentageSize.width)  > 0 ? percentageSize.width  : pbox.x;
	 yMov=(pbox.y-percentageSize.height) > 0 ? percentageSize.height : pbox.y;
	 wMov=((pbox.x -xMov) + pbox.width  + 2*percentageSize.width)  < img_scene.cols ? (2*percentageSize.width) :  ((img_scene.cols-(pbox.x -xMov))-pbox.width);
	 hMov=((pbox.y -yMov) + pbox.height + 2*percentageSize.height) < img_scene.rows ? (2*percentageSize.height) : ((img_scene.rows-(pbox.y -yMov))-pbox.height);
	 CvSize newSize=cvSize(pbox.width-(wMov/2),pbox.height-(hMov/2));

	 CvRect oldPbox=cvRect(pbox.x,pbox.y,pbox.width,pbox.height);

	img_objectMultiresolution(pbox).copyTo(oldFace);
	getPatterns(oldFace,oldFace15,dummyM,dummyS);

	//Mat imo;


	if(!examineSurf(img_object,img_scene,pbox)){
		returnV=false;

		if(goOn){
			pbox.x=oldPbox.x-xMov;
			pbox.y=oldPbox.y-yMov;
			pbox.width=oldPbox.width+wMov;
			pbox.height=oldPbox.height+hMov;

			xOff=wMov/2;
			yOff=hMov/2;

			CvRect tmpPbox=cvRect(0,0,xOff,yOff);

			for(int count=0;count<5 && !returnV;++count){
				rng.operator float();
				xMov=(((float)rng))*xOff;
				yMov=(((float)rng))*yOff;


				tmpPbox.x=pbox.x+xMov;
				tmpPbox.y=pbox.y+yMov;
				tmpPbox.width=newSize.width;
				tmpPbox.height=newSize.height;
				img_objectMultiresolution(tmpPbox).copyTo(tmpObj);
				img_objectMultiresolution.copyTo(ims);
				rectangle(ims,tmpPbox,cvScalar(0),6);
				rectangle(ims,pbox,cvScalar(125),4);
				rectangle(ims,oldPbox,cvScalar(255),2);
				imshow("scena",ims);
				//rectangle(img_object,tmpPbox,cvScalar(0,255,0),2);
				//imshow("realObj",img_object);
				//imshow("obj",tmpObj);

				img_scene.copyTo(ims);
				rectangle(ims,tmpPbox,cvScalar(0),8);

				if(examineSurf(tmpObj,img_scene,tmpPbox)){
					returnV=true;
					pbox.x=oldPbox.x + (tmpPbox.x-(pbox.x+xMov));
					pbox.x=pbox.x>0?pbox.x:0;
					pbox.y=oldPbox.y + (tmpPbox.y-(pbox.y+yMov));
					pbox.y=pbox.y>0? pbox.y:0;

					pbox.width=oldPbox.width*((float)tmpPbox.width/newSize.width);
					pbox.width=((pbox.x+pbox.width) < img_scene.cols) ? pbox.width : img_scene.cols-pbox.x;
					pbox.height=oldPbox.height*((float)tmpPbox.height/newSize.height);
					pbox.height=((pbox.y+pbox.height) < img_scene.rows) ? pbox.height : img_scene.rows-pbox.y;
					//img_scene.copyTo(ims);
					rectangle(ims,tmpPbox,cvScalar(50),6);
					rectangle(ims,pbox,cvScalar(125),4);
					rectangle(ims,oldPbox,cvScalar(255),2);
					imshow("scenadopo",ims);
				}
			}

			//cout << "\007";
			//waitKey(0);

		}
	}

	 return(returnV);
}





void FeaturesTrack::updateNewFace(Mat& face,float matchV){

	if(matchV>(getMatchMean()-0.01)){
		cout << endl << "UPDATED MeanMatch: " << getMatchMean() ;
		if(lastFiveFace.size()>=20)
			lastFiveFace.pop_back();

		lastFiveFace.insert(lastFiveFace.begin(),face);
	} else cout << endl << "NOT UPDATED NOT MeanMatch: " << getMatchMean() ;


	cout << endl;
	//waitKey(0);
}

void FeaturesTrack::updateNewFaceWithResize(Mat& face){
	Mat newFace15;
	Scalar dummyM,dummyS;
	getPatterns(face,newFace15,dummyM,dummyS);
	float matching=getMatching(newFace15);

	modelSize++;
	matchingMean+=matching;
	if(lastFiveFace.size()>=20)
			lastFiveFace.pop_back();
	lastFiveFace.insert(lastFiveFace.begin(),newFace15);


	cout << endl << "UPDATED FROM DETECTION MeanMatch: " << getMatchMean() ;
//waitKey(0);
}

void FeaturesTrack::showExamples(){
	int rows=lastFiveFace.size()>30 ? 30:lastFiveFace.size();
	int cols=1+(int)lastFiveFace.size()/30;
	//cols=cols>1?cols:1;
	  Mat examples(rows*lastFiveFace[0].rows,cols*lastFiveFace[0].cols,CV_8U);
	  double minval;
	  Mat ex(lastFiveFace[0].rows,lastFiveFace[0].cols,lastFiveFace[0].type());
	  for (int i=0;i<lastFiveFace.size();i++){
	    minMaxLoc(lastFiveFace[i],&minval);
	    lastFiveFace[i].copyTo(ex);
	    ex = ex-minval;
	    cout<< endl << "rows:" << (i%rows) << "rowsE:" << (i%rows)+1;
	    cout << endl;
	    Mat tmp1 = examples.rowRange(Range((i%rows)*lastFiveFace[i].rows,((i%rows)+1)*lastFiveFace[i].rows));
	    Mat tmp= tmp1.colRange(Range((i/rows)*lastFiveFace[i].cols,((i/rows)+1)*lastFiveFace[i].cols));
	    ex.convertTo(tmp,CV_8U);
	  }
	 imshow("Examples",examples);
}


float FeaturesTrack::getMatchMean(){
	return((classifier.getBestCalculatedThrNN()+classifier.getNNTh())/2);
	//return(classifier.getBestCalculatedThrNN());
}




void FeaturesTrack::processFrame(const cv::Mat& img1,const cv::Mat& img2,const cv::Mat& frameTo,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,CvRect& histogramDetectionBlob,bool& lastboxfound, bool tl, FILE* bb_file){

 // processFrameale(img1.clone(),img2.clone(),points1,points2,bbnext,lastboxfound,tl,bb_file);
  vector<BoundingBox> cbb;
  vector<float> cconf;
 // vector<float> cconfN;
  int confident_detections=0;
  int didx; //detection index
  Mat printFrame=img2.clone();
  Mat frameBox=img2.clone();
  int detections;
  float bestConf=0;

  ReExamine:
  ///Track
  //imshow("frame From ",frameFrom);
  //imshow("frame To",frameTo);
  if(lastboxfound && tl){
      //track(frameFrom,frameTo,points1,points2,false);
	  track(img1,img2,points1,points2,false);
      if(!tracked)
    	  track(img1,img2,points1,points2,true);

  }
  else{
      tracked = false;
      lastconf=0;
  }

  ///Integration
  if (tracked){


	  if((tbb.height < MINBBOX)||(tbb.width < MINBBOX)){
		  CvRect upperBB=cvRect(0,0,0,0);
		  Mat patchMom;
		  frameTo(tbb).copyTo(patchMom);
		  int xct=tbb.width/2;
		  int yct=tbb.height/2;
		  getConnectedComponent(patchMom,xct,yct,cvSize(patchMom.rows/2,patchMom.cols/2));
		  float areat=tbb.width*tbb.height;
		  float areah=upperBB.height*upperBB.width;
		  if(areah>areat){
			  tbb.width=upperBB.width;
			  tbb.height=upperBB.height;
			  tbb.x=upperBB.x;
			  tbb.y=upperBB.y;
		  }
		 // cvWaitKey(0);
	  }




	    BoundingBox selectionBox(cvRect(0,0,0,0));

	    selectionBox.x=histogramDetectionBlob.x-(histogramDetectionBlob.width/4) > 0  ? histogramDetectionBlob.x-(histogramDetectionBlob.width/4) : 0;
    	selectionBox.y=histogramDetectionBlob.y-(histogramDetectionBlob.height/4) > 0 ? histogramDetectionBlob.y-(histogramDetectionBlob.height/4): 0;
        selectionBox.width=histogramDetectionBlob.width +(histogramDetectionBlob.width/2);
        selectionBox.width=(selectionBox.width + selectionBox.x) < frameTo.cols ? selectionBox.width : (frameTo.cols-selectionBox.x);
        selectionBox.height=histogramDetectionBlob.height +(histogramDetectionBlob.height/2);
        selectionBox.height=(selectionBox.height + selectionBox.y) < frameTo.rows ? selectionBox.height : (frameTo.rows-selectionBox.y);



	    Mat patchMom(frameTo(selectionBox));
	    Moments mom=moments(patchMom);
	    int xc=(mom.m10/mom.m00);
	    xc=xc>0 ?xc:0;
	    int yc=(mom.m01/mom.m00);
	    yc =yc>0 ? yc:0;

	    // erode(patchMom,patchMom,Mat());
	    //threshold(patchMom,patchMom,0,255,CV_THRESH_BINARY);
	    if(xc<0){
	   // imshow("erode",patchMom);
	    Mat frametemp;
	    frameTo.copyTo(frametemp);
	    rectangle(frametemp,(CvRect)selectionBox,cvScalar(255),2);
	   // imshow("frame erode after",frametemp);
	  cvWaitKey(0);
	    }
	    //std::vector<std::vector<cv::Point> > contours;
	    //findContours(patchMom,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);



//	    RotatedRect boundingRectangle=minAreaRect(patchMom);


	     xc=selectionBox.x + xc;
	   	 yc=selectionBox.y + yc;

	     img2.copyTo(patchMom);
	   	rectangle(patchMom,tbb,cvScalar(0),2);
	   	rectangle(patchMom,selectionBox,cvScalar(100),2);

	   //	tbb.width= tbb.width > selectionBox.width ? selectionBox.width : tbb.width;
	   //	tbb.height= tbb.height > selectionBox.height ? selectionBox.height : tbb.height;



	  detections=detect(img2,cbb,cconf,classifier.getBestCalculatedThrNN(),(CvRect)selectionBox);


		  selectionBox.width=(tbb.width+histogramDetectionBlob.width)/2;
		  selectionBox.height=(tbb.height+histogramDetectionBlob.height)/2;
		  xc=(xc+(tbb.x+(tbb.width/2))+(histogramDetectionBlob.x+(histogramDetectionBlob.width/2)))/3;
		  yc=(yc+(tbb.y+(tbb.height/2))+(histogramDetectionBlob.y+(histogramDetectionBlob.height/2)))/3;
		  selectionBox.x= xc-(selectionBox.width/2);
		  selectionBox.y= yc-(selectionBox.height/2);


	//   selectionBox.x=(tbb.x+histogramDetectionBlob.x)/2;
	//     selectionBox.y=(tbb.y+histogramDetectionBlob.y)/2;

      selectionBox.x=selectionBox.x>0 ? selectionBox.x:0;
	  selectionBox.y=selectionBox.y>0 ? selectionBox.y:0;
	  selectionBox.width=(selectionBox.width+selectionBox.x) < img2.cols ? selectionBox.width : img2.cols-selectionBox.x;
	  selectionBox.height=(selectionBox.height+selectionBox.y) < img2.rows ? selectionBox.height : img2.rows-selectionBox.y;



	  	   // drawContours(patchMom,contours,-1,cv::Scalar(255),2);
	  	   // rectangle(patchMom,selectionBox,cvScalar(255),1);
	  	   // rectangle(patchMom,histogramDetectionBlob,cvScalar(150),1);
	  	  //  rectangle(patchMom,boundingRectangle.,cvScalar(150),3);
	  	   // circle(patchMom,cvPoint(xc,yc),4,cvScalar(255),4);
	  	 //   imshow("moments tbb",patchMom);


	  	   // Finr aggointo




	  	  //detect(img2,cbb,cconf,classifier.getBestCalculatedThrNN(),cvRect(0,0,img2.cols,img2.rows));

	  	  tbb=selectionBox;
	  	  Mat patch;
	  	  Scalar means,stdevi;
	  	  float dummy;
	  	  vector<int> isin;
	  	  getPatterns(img2(tbb),patch,means,stdevi);                //  Get pattern within bounding box
	  	  classifier.NNConf(patch,isin,tconf,dummy);



	  bbnext=tbb;
      lastconf=tconf;
      lastvalid=tvalid;
      printf("Tracked\n");
      if(detected){                                               //   if Detected
          //clusterConf(dbb,dconf,cbb,cconf,cconfN);                       //   cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          for (int i=0;i<cbb.size();i++){
              //if (bbOverlap(tbb,cbb[i])<0.65 && cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
        	  if (cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
                  confident_detections++;
                  if(bestConf<cconf[i]){
                  didx=i; //detection index
                  bestConf=cconf[i];
                  }
                  drawBox(printFrame,(CvRect)cbb[i],0.5,1);
              }
          }

          if (confident_detections>=1){                                //if there is ONE such a cluster, re-initialize the tracker
              printf("Found a better match..reinitializing tracking\n");
              bbnext=cbb[didx];
              lastconf=cconf[didx];
              lastvalid=false;
              printf("\n\rConfident<= %d\n\r",confident_detections);
              printf("box Found=%d\n\r",cbb.size());
              drawBox(printFrame,(CvRect)bbnext,1,2);
            //  imshow("clusterconf",printFrame);

            //  if(confident_detections>=1)
            	 // cvWaitKey(0);
          }
          else {
              printf("%d confident cluster was found\n",confident_detections);
              int cx=0,cy=0,cw=0,ch=0;
              int close_detections=0;
              for (int i=0;i<dbb.size();i++){
                  if(bbOverlap(tbb,dbb[i])>0.7){                     // Get mean of close detections
                      cx += dbb[i].x;
                      cy +=dbb[i].y;
                      cw += dbb[i].width;
                      ch += dbb[i].height;
                      close_detections++;
                      printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                  }
              }
              if (close_detections>0){
                  bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
                  bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                  bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                  bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                  printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                  printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                  printf("Weighting %d close detection(s) with tracker..\n",close_detections);
                  Mat patch;
                  Scalar means,stdevi;
                  float dummy;
                  vector<int> isin;
                  getPatterns(img2(bbnext),patch,means,stdevi);                //  Get pattern within bounding box
                  classifier.NNConf(patch,isin,lastconf,dummy);
              }
              else{
                printf("%d close detections were found\n",close_detections);
              }
          }
      }
  }
  else{
	  //   If NOT tracking
	 // cvWaitKey(0);
	  printf("Not tracking..\n");
	  detect(img2,cbb,cconf,classifier.getBestCalculatedThrNN(),cvRect(0,0,img2.cols,img2.rows));
      lastboxfound = false;
      lastvalid = false;
      if(detected){                           //  and detector is defined
          //clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          float maxcconf=0;
          for(int i=0 ; i < cbb.size();++i){
        	  if (cconf[i]>maxcconf){
        		  maxcconf=cconf[i];
        		  bbnext=cbb[i];
        		  lastconf=cconf[i];
        		  printf("Confident detection..reinitializing tracker\n");
        		  lastboxfound = true;
        	  }
          }
          drawBox(printFrame,(CvRect)bbnext,1,2);
       //   imshow("clusterconfnot tracking",printFrame);
      } else {
    	  lastconf=0;
    	  framesInconfidence=1001;
      }
  }


  lastbox=bbnext;

  if (lastconf >= classifier.getBestCalculatedThrNN()){
	  framesInconfidence=0;
	  lastboxfound=true;
	 // learnModel(img2,detected);
	  //if(!detected) goto ReExamine;
	  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  }else{
	  framesInconfidence++;
	  if(framesInconfidence>3){
		  lastboxfound=false;
		  cout << endl << "frameIconfidence >10 ";
		  cout << endl;
		  fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
		  //cvWaitKey(0);
	  }else {
		  cout << endl <<"inconfidence Frame";
		  cout << endl;



		  //learnModel(img2,detected);
		  drawBox(printFrame,cvRect(0,0,100,100),1,2);
		  drawBox(printFrame,(CvRect)bbnext,1,2);
		//  imshow("inconfidence Frame",printFrame);
		  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
		  //forceLearnModel(img2);
		  //cvWaitKey(0);
		  //if(!detected) goto ReExamine;
	  }


  }


  printf("\n\n\n thr_nn=%f / thr_nn_Valid=%f / best_thr_nn=%f \n\n\n",classifier.getNNTh(),classifier.getThr_nn_Valid(),classifier.getBestCalculatedThrNN());

}



/*

void FeaturesTrack::processFrame(const cv::Mat& img1,const cv::Mat& img2,const cv::Mat& frameTo,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,CvRect& histogramDetectionBlob,bool& lastboxfound, bool tl, FILE* bb_file){

 // processFrameale(img1.clone(),img2.clone(),points1,points2,bbnext,lastboxfound,tl,bb_file);
  vector<BoundingBox> cbb;
  vector<float> cconf;
 // vector<float> cconfN;
  int confident_detections=0;
  int didx; //detection index
  Mat printFrame=img2.clone();
  Mat frameBox=img2.clone();
  int detections;
  float bestConf=0;

  ReExamine:
  ///Track
  //imshow("frame From ",frameFrom);
  //imshow("frame To",frameTo);
  lastboxfound=true;
  if(lastboxfound && tl){
      //track(frameFrom,frameTo,points1,points2,false);
	  track(img1,img2,points1,points2,false);
      if(!tracked)
    	  track(img1,img2,points1,points2,true);

  }
  else{
      tracked = false;
      lastconf=0;
  }

  ///Integration
  if (tracked){

	    BoundingBox selectionBox(cvRect(0,0,0,0));

	    selectionBox.x=histogramDetectionBlob.x-(histogramDetectionBlob.width/4) > 0  ? histogramDetectionBlob.x-(histogramDetectionBlob.width/4) : 0;
    	selectionBox.y=histogramDetectionBlob.y-(histogramDetectionBlob.height/4) > 0 ? histogramDetectionBlob.y-(histogramDetectionBlob.height/4): 0;
        selectionBox.width=histogramDetectionBlob.width +(histogramDetectionBlob.width/2);
        selectionBox.width=(selectionBox.width + selectionBox.x) < frameTo.cols ? selectionBox.width : (frameTo.cols-selectionBox.x);
        selectionBox.height=histogramDetectionBlob.height +(histogramDetectionBlob.height/2);
        selectionBox.height=(selectionBox.height + selectionBox.y) < frameTo.rows ? selectionBox.height : (frameTo.rows-selectionBox.y);

	    Mat patchMom(frameTo(selectionBox));
	    Moments mom=moments(patchMom);
	    int xc=(mom.m10/mom.m00);
	    xc=xc>0 ?xc:0;
	    int yc=(mom.m01/mom.m00);
	    yc =yc>0 ? yc:0;

	    if(xc<0){
	  	    Mat frametemp;
	  	    frameTo.copyTo(frametemp);
	  	    rectangle(frametemp,(CvRect)selectionBox,cvScalar(255),2);
	  	    cvWaitKey(0);
	    }
	     xc=selectionBox.x + xc;
	   	 yc=selectionBox.y + yc;

	   selectionBox.width=(tbb.width+histogramDetectionBlob.width)/2;
	   selectionBox.height=(tbb.height+histogramDetectionBlob.height)/2;
	   xc=(xc+(tbb.x+(tbb.width/2))+(histogramDetectionBlob.x+(histogramDetectionBlob.width/2)))/3;
	   yc=(yc+(tbb.y+(tbb.height/2))+(histogramDetectionBlob.y+(histogramDetectionBlob.height/2)))/3;
	   selectionBox.x= xc-(selectionBox.width/2);
	   selectionBox.y= yc-(selectionBox.height/2);

	   selectionBox.x=selectionBox.x>0 ? selectionBox.x:0;
	   selectionBox.y=selectionBox.y>0 ? selectionBox.y:0;
	   selectionBox.width=(selectionBox.width+selectionBox.x) < img2.cols ? selectionBox.width : img2.cols-selectionBox.x;
	   selectionBox.height=(selectionBox.height+selectionBox.y) < img2.rows ? selectionBox.height : img2.rows-selectionBox.y;

	   tbb=selectionBox;
	   bbnext=tbb;
      } else{
    	  lastboxfound=false;
      }

  lastbox=bbnext;


  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);

}
*/

 //aleclassifier
 /*
bool FeaturesTrack::processFrameale(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound, bool tl, FILE* bb_file){
  vector<BoundingBox> cbb;
  vector<float> cconf;
  int confident_detections=0;
  int didx; //detection index
  Mat printFrame=img2.clone();
  float best_Conf=0;
  bool stop=false;
  detect(img2,cbb,cconf);
  ///Integration
          for (int i=0;i<dbb.size();i++){
              if (bbOverlap(tbb,dbb[i])<0.7 && dconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
                  confident_detections++;
                  if(best_Conf<dconf[i]){
                	  didx=i; //detection index
                	  best_Conf=dconf[i];
                  }
                  drawBox(printFrame,dbb[i],0.5,1);
              }
          }
          if (confident_detections>=1){                                //if there is ONE such a cluster, re-initialize the tracker
              printf("Found a better match..reinitializing tracking\n");
              bbnext=dbb[didx];
              drawBox(printFrame,bbnext,1,2);
              printf("\n\rbox Found ALe=%d",dbb.size());
              imshow("clusterconfale",printFrame);
              cvWaitKey(0);
              stop=true;
          }
return(stop);
}
*/

void FeaturesTrack::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,bool allP){
  /*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
  //Generate points
  //calculateGoodFeatures(img1,lastbox,points1);
  //bbPoints(points1,lastbox);
 if(!allP)
  //bbPoints(points1,lastbox,img1,img2);
	 bbPointsOlD(points1,lastbox);
 else bbPointsAllP(points1,lastbox,img1,img2);


  printf("\nMaxCorners=%d",points1.size());
  if (points1.size()<1){
      printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
      tvalid=false;
      tracked=false;
      return;
  }
 // vector<Point2f> points = points1;
  //Frame-to-frame tracking with forward-backward error cheking
  tracked = tracker.trackf2f(img1,img2,points1,points2);


  if (tracked){
      //Bounding box prediction
      bbPredict(points1,points2,lastbox,tbb);
     // cout << endl << "ttb.x:" << tbb.x << " ttb.y:" << tbb.y << " ttb.w:" << tbb.width << " ttb.h:" << tbb.height << " ttb.br.x:" << tbb.br().x << " ttb.br.y:" << tbb.br().y << endl;
     // cvWaitKey(0);
      if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){
          tvalid =false; //too unstable prediction or bounding box out of image
          tracked = false;
          printf("Too unstable predictions FB error=%f\n",tracker.getFB());

      } else{
      //Estimate Confidence and Validity
      Mat pattern;
      Scalar mean, stdev;
      BoundingBox bb;
      bb.x = max(tbb.x,0);
      bb.y = max(tbb.y,0);
      bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
      bb.width= (bb.width+bb.x)< img2.cols? (bb.width): img2.cols-bb.x;
      bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
      bb.height= (bb.height+bb.y)< img2.rows? (bb.height): img2.rows-bb.y;
      tbb.x=bb.x;
      tbb.y=bb.y;
      tbb.height=bb.height;
      tbb.width=bb.width;
      getPatterns(img2(bb),pattern,mean,stdev);
      vector<int> isin;
      float dummy;
      classifier.NNConf(pattern,isin,tconf,dummy); //Conservative Similarity
     // tvalid = lastvalid;
      if (tconf>classifier.getBestCalculatedThrNN()){
          tvalid =true;
         // tracked=true;
      }
      else {
    	  tvalid=false;
    	  //tracked=false;
      }
      }
  }
  else{
    printf("No points tracked\n");
    //cvWaitKey(0);
  }

}




// bbBoints for good features to track
void FeaturesTrack::bbPointsOlD(vector<cv::Point2f>& points,const BoundingBox& bb){
  int margin_h=0;
  int margin_v=0;
  int max_pts=20;

  if(!(bb.height>max_pts && bb.width>max_pts))
	  max_pts=bb.height<bb.width ? bb.height :bb.width;

  int stepx = ceil((bb.width-2*margin_h)/max_pts);
  int stepy = ceil((bb.height-2*margin_v)/max_pts);
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
    		  points.push_back(Point2f(x,y));
      }
  }
}




//void FeaturesTrack::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){

void FeaturesTrack::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb,const Mat& frameFrom,const Mat& frameTo){

	float quality=0.02;
	float minDistance=1;
	Mat mask=Mat().zeros(cvSize(frameFrom.cols,frameFrom.rows),CV_8UC1);
	mask(bb)=cvScalar(1);
	goodFeaturesToTrack(frameFrom,points,frameFrom.cols*frameFrom.rows,quality,minDistance,mask);

	int max_pts=10;
	int margin_h=0;
	int margin_v=0;
	int stepx = ceil((bb.width-2*margin_h)/max_pts);
	stepx=stepx>0?stepx:1;
	int stepy = ceil((bb.height-2*margin_v)/max_pts);
	stepy=stepy>0?stepy:1;
	for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
    	  //cout << endl << "frame " << x << " " << y << " = " <<(int) frame.at<uchar>(y,x);
    	  //circle(frame,cvPoint(x,y),2,cvScalar(1),1);
    	  //imshow("punti",frame);
    	  //cvWaitKey(0);
    	  if((int)frameFrom.at<uchar>(y,x)>0)
    		  points.push_back(Point2f(x,y));
      }
  }
//  cvWaitKey(0);
}

void FeaturesTrack::bbPointsAllP(vector<cv::Point2f>& points,const BoundingBox& bb,const Mat& frameFrom,const Mat& frameTo){



		float quality=0.01;
		float minDistance=0.8;
		Mat mask=Mat().zeros(cvSize(frameFrom.cols,frameFrom.rows),CV_8UC1);
		mask(bb)=cvScalar(1);
		goodFeaturesToTrack(frameFrom,points,frameFrom.cols*frameFrom.rows,quality,minDistance,mask);


		int max_pts=20;
		int margin_h=0;
		int margin_v=0;
		int stepx = ceil((bb.width-2*margin_h)/max_pts);
		stepx=stepx>0?stepx:1;
		int stepy = ceil((bb.height-2*margin_v)/max_pts);
		stepy=stepy>0?stepy:1;
		for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
	      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
	    	  //cout << endl << "frame " << x << " " << y << " = " <<(int) frame.at<uchar>(y,x);
	    	  //circle(frame,cvPoint(x,y),2,cvScalar(1),1);
	    	  //imshow("punti",frame);
	    	  //cvWaitKey(0);
	    	 // if((int)frameFrom.at<uchar>(y,x)>0)
	    		  points.push_back(Point2f(x,y));
	      }
	  }


}


/*
void FeaturesTrack::bbPredictAle(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2,const Mat& img)    {

  int xFromMin,xFromMax,yFromMin,yFromMax;
  int xToMin,xToMax,yToMin,yToMax;
  float xStart,yStart,wSize,hSize;
  Mat frame;

  xFromMin=yFromMin=xToMin=yToMin=99999;
  xFromMax=yFromMax=xToMax=yToMax=0;


  int npoints = (int)points1.size();
  vector<float> xoff(npoints);
  vector<float> yoff(npoints);
  printf("\n*****************************************************");
  printf("\n tracked points : %d",npoints);

  img.copyTo(frame);

  for (int i=0;i<npoints;i++){
	  xFromMin=(xFromMin < points1[i].x) ? xFromMin : points1[i].x;
	  xFromMax=(xFromMax > points1[i].x) ? xFromMax : points1[i].x;

	  yFromMin=(yFromMin < points1[i].y) ? yFromMin : points1[i].y;
	  yFromMax=(yFromMax > points1[i].y) ? yFromMax : points1[i].y;


	  xToMin=(xToMin < points2[i].x) ? xToMin : points2[i].x;
	  xToMax=(xToMax > points2[i].x) ? xToMax : points2[i].x;

	  yToMin=(yToMin < points2[i].y) ? yToMin : points2[i].y;
	  yToMax=(yToMax > points2[i].y) ? yToMax : points2[i].y;
  }

  printf("\n XFomMin=%d xFromMax=%d yFromMin=%d yFromMax=%d",xFromMin,xFromMax,yFromMin,yFromMax);
  printf("\n XToMin=%d xToMax=%d yToMin=%d yToMax=%d",xToMin,xToMax,yToMin,yToMax);

  xStart=xFromMin-bb1.x;
  yStart=yFromMin-bb1.y;
  wSize=((float)xToMax-xToMin)/(xFromMax-xFromMin);
  hSize=((float)yToMax-yToMin)/(yFromMax-yFromMin);

  printf("\n xStart=%f yStart=%f",xStart,yStart);
  printf("\n wSize =%f hSize =%f",wSize,hSize);

  xStart=xToMin-wSize*xStart;
  yStart=yToMin-hSize*yStart;
  wSize=wSize*bb1.width;
  hSize=hSize*bb1.height;

  xStart=(xStart > 0) ? xStart : 0;
  yStart=(yStart > 0) ? yStart : 0;

  bb2.x = round(xStart);
  bb2.y = round(yStart);
  bb2.width = round(wSize);
  bb2.height = round(hSize);
  printf("\n Started   bb: %d %d %d %d",bb1.x,bb1.y,bb1.br().x,bb1.br().y);
  printf("\n predicted bb: %d %d %d %d",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
  printf("\n###################################################################");
  drawBox(frame,cvRect(bb2.x,bb2.y,bb2.width,bb2.height),cvScalar(255),2);

 // bbPredict(points1,points2,bb1,bb2);
  drawBox(frame,cvRect(bb2.x,bb2.y,bb2.width,bb2.height),cvScalar(0),2);

  imshow("BBOX",frame);
  //cvWaitKey(0);

}*/


void FeaturesTrack::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);
  vector<float> yoff(npoints);
  printf("\n tracked points : %d",npoints);
  for (int i=0;i<npoints;i++){
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }
  float dx = median(xoff);
  float dy = median(yoff);
  float s;
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++){
              d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
          }
      }
      s = median(d);
  }
  else {
      s = 1.0;
  }
  float s1 = 0.5*(s-1)*bb1.width;
  float s2 = 0.5*(s-1)*bb1.height;
  printf("\n s= %f s1= %f s2= %f",s,s1,s2);
  bb2.x = round( bb1.x + dx -s1);
  bb2.y = round( bb1.y + dy -s2);
  bb2.width = round(bb1.width*s);
  bb2.height = round(bb1.height*s);
  printf("\n predicted bb: %d %d %d %d",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}


// detect Ale
void FeaturesTrack::detectAle(const cv::Mat& frame){
  //cleaning
  dbb.clear();
  dconf.clear();
  dt.bb.clear();
  double t = (double)getTickCount();
  Mat img(frame.rows,frame.cols,CV_8U);
  integral(frame,iisum,iisqsum);
  GaussianBlur(frame,img,Size(9,9),1.5);
  int numtrees = classifier.getNumStructs();
  float fern_th = classifier.getFernTh();
  vector <int> ferns(classifier.getNumStructs());
  float conf;
  int a=0;
  Mat patch;
  double currentPatchVariance;
  double varplus15percent= (2*var*0.5)+2*var;
  double varless15percent= 2*var-(2*var*0.5);
  IntegralImage integralFrame;

  for (int i=0;i<grid.size();i++){//FIXME: BottleNeck
	  currentPatchVariance=getVar(grid[i],iisum,iisqsum);
	  //cout << endl << "Var:" << var << " currentvar:" << currentPatchVariance << " Var-15:" << varless15percent << " Val+15:" << varplus15percent;

     // if (currentPatchVariance>=var){
	  if((currentPatchVariance>=varless15percent)&& (currentPatchVariance<=varplus15percent)){
    	//  cvWaitKey(0);
          a++;
		  patch = img(grid[i]);
		  integralFrame=IntegralImage(patch);
          classifier.getFeatures(patch,integralFrame,grid[i].sidx,ferns);
          conf = classifier.measure_forest(ferns);
          tmp.conf[i]=conf;
          tmp.patt[i]=ferns;
          if (conf>numtrees*fern_th){
              dt.bb.push_back(i);
          }
      }
      else
        tmp.conf[i]=0.0;
  }
  int detections = dt.bb.size();
  printf("%d Bounding boxes passed the variance filter\n",a);
  printf("%d Initial detection from Fern Classifier\n",detections);
  if (detections>100){
      nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      dt.bb.resize(100);
      detections=100;
  }

  for (int i=0;i<detections;i++){
        drawBox(img,(CvRect)grid[dt.bb[i]]);
    }
 // imshow("detectionsAle",img);

  if (detections==0){
        detected=false;
        return;
      }
  printf("Fern detector made %d detections ",detections);
  t=(double)getTickCount()-t;
  printf("in %gms\n", t*1000/getTickFrequency());
                                                                       //  Initialize detection structure
  dt.patt = vector<vector<int> >(detections,vector<int>(classifier.getNumStructs(),0));        //  Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
  int idx;
  Scalar mean, stdev;
  float nn_th = classifier.getNNTh();
  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      idx=dt.bb[i];                                                       //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPatterns(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
      classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      if (dt.conf1[i]>nn_th){                                               //  idx = dt.conf1 > FeaturesTrack.model.thr_nn; % get all indexes that made it through the nearest neighbour
          dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
          dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
      }
  }                                                                         //  end
  if (dbb.size()>0){
      printf("Found %d NN matches\n",(int)dbb.size());
      detected=true;
  }
  else{
      printf("No NN matches found.\n");
      detected=false;
  }
}


bool FeaturesTrack::isInternal(CvRect bboxHull, CvRect bbox){

	if( (bbox.x>=bboxHull.x) &&
		((bbox.x+bbox.width)<=(bboxHull.x+bboxHull.width)) &&
		(bbox.y>=bboxHull.y) &&
		((bbox.y+bbox.height)<=(bboxHull.y+bboxHull.height)))
		return true;
	else
		return false;
}


int FeaturesTrack::detect(const cv::Mat& frame,vector<BoundingBox>& cbb,vector<float>& cconf,float nn_th,CvRect bboxToDetect){
  //cleaning
  //detectAle(frame.clone());
  dbb.clear();
  dconf.clear();
  dt.bb.clear();
  cconf.clear();
  cbb.clear();
  vector<BoundingBox> tmpCbb;

  double t = (double)getTickCount();
  Mat img(frame.rows,frame.cols,CV_8U);
  Mat imgDetection(frame.rows,frame.cols,CV_8U);
  Mat imgClusterConf(frame.rows,frame.cols,CV_8U);
  integral(frame,iisum,iisqsum);
  GaussianBlur(frame,img,Size(9,9),1.5);
  //frame.copyTo(img);
  int numtrees = classifier.getNumStructs();
  float fern_th = classifier.getFernTh();
  vector <int> ferns(numtrees);
  float conf;
  int a=0;
  Mat patch;
  Mat patch15;
  IntegralImage integralPatch;
  //double currentPatchVariance;
  //double varplus15percent= (2*var*0.5)+2*var;
  //double varless15percent= 2*var-(2*var*0.5);
  for (int i=0;i<grid.size();i++){//FIXME: BottleNeck
	  if(isInternal((CvRect)bboxToDetect,(CvRect)grid[i])){
		  if (getVar(grid[i],iisum,iisqsum)>=var){
			  a++;
			  patch = img(grid[i]);
			  integralPatch=IntegralImage(patch);
			  classifier.getFeatures(patch,integralPatch,grid[i].sidx,ferns);
			  conf = classifier.measure_forest(ferns);
			  tmp.conf[i]=conf;
			  tmp.patt[i]=ferns;
			  if (conf>numtrees*fern_th){
				  dt.bb.push_back(i);
			  }
		  } else tmp.conf[i]=0.0;
	  } else tmp.conf[i]=0.0;
  }

  int detections = dt.bb.size();
  printf("%d Bounding boxes passed the variance filter\n",a);
  printf("%d Initial detection from Fern Classifier\n",detections);
  if (detections>100){
      nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      //nth_element(dbb.begin(),dbb.begin()+100,dbb.end(),CComparator(tmp.conf));
      dt.bb.resize(100);
      //dbb.resize(100);
      detections=100;
  }

  img.copyTo(imgDetection);
  img.copyTo(imgClusterConf);
  /*
  for (int i=0;i<detections;i++){

    }
    */


  if (detections==0){
        detected=false;
        return detections;
      }
  printf("Fern detector made %d detections ",detections);
  t=(double)getTickCount()-t;
  printf("in %gms\n", t*1000/getTickFrequency());

  dt.patt = vector<vector<int> >(detections,vector<int>(classifier.getNumStructs(),0));        //  Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
  int idx;
  Scalar mean, stdev;

  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      idx=dt.bb[i];                                                       //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPatterns(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
      classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      drawBox(img,(CvRect)grid[idx]);
      dbb.push_back(grid[idx]);
  }



  cout<< endl << endl << endl << "CkusterConr" << endl << endl << endl;
  clusterConfAle(dbb,tmpCbb);
  detections=tmpCbb.size();


  //float nn_th = classifier.getBestCalculatedThrNN();

/*
  //################   TEXT   #####################
  //
  	  double hscale = 0.5;
  	  double vscale = 0.3;
  	  double shear = 0.0;
  	  int thickness = 2;
  	  int line_type = 8;
  	  CvFont font1;
  	  cvInitFont(&font1,CV_FONT_HERSHEY_DUPLEX,hscale,vscale,shear,thickness,line_type);
  	  IplImage tmpIplImg;
  	  char str[20]=" ";
  //
  //################ END TEXT #####################
*/

  vector<int> isin;
  float conf1,conf2;
  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      //idx=dt.bb[i];                                                       //  Get the detected bounding box index
	 // patch = frame(tmpCbb[i]);
	  drawBox(imgClusterConf,(CvRect)tmpCbb[i]);
      getPatterns(frame(tmpCbb[i]),patch15,mean,stdev);                //  Get pattern within bounding box
      classifier.NNConf(patch15,isin,conf1,conf2);  //  Evaluate nearest neighbour classifier
      /*
      imshow("patch",patch);
      imshow("dt.patch[i]",dt.patch[i]);
      cvWaitKey(0);
*/
      //cout << endl << dt.patch[i];
      //dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      if (conf1>nn_th){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
          //cco.push_back(cbb[i]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
          cconf.push_back(conf1);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
          cbb.push_back(tmpCbb[i]);
          /*
          tmpIplImg=imgDetection.operator _IplImage();
          sprintf(str,"%f",dt.conf1[i]);
          cvPutText(&tmpIplImg,str,cvPoint(grid[idx].x,grid[idx].y),&font1,cvScalar(255,0,0));
          */
          drawBox(imgDetection,(CvRect)tmpCbb[i]);

      }
  }                                                                         //  end

  if (cbb.size()>0){
      printf("Found %d NN matches\n",(int)cbb.size());
      detected=true;
  }
  else{
      printf("No NN matches found.\n");
      detected=false;
  }

 // imshow("detection match",imgDetection);
  //rectangle(img,bboxToDetect,cvScalar(0),2);
 // imshow("detections Ferns",img);
 // imshow("detection Unite",imgClusterConf);

  return cbb.size();
 // cvWaitKey(0);
}



/*
void FeaturesTrack::evaluate(){
}
*/

void FeaturesTrack::learnModel(const Mat& frame,CvRect& pbox){

  vector<int> isin;
  //float dummy, conf;
  Mat patch;
  Rect tmpBox(0,0,0,0);
  Rect workBox(0,0,0,0);
  RNG& rng = theRNG();
  //Mat printfr;

 // frame.copyTo(printfr);
 // rectangle(printfr,pbox,cvScalar(255),1);

  vector<Mat> nn_examples;
  vector<Mat> nn_examples_NX;
  vector<Mat> nn_examples_NXT;

  Mat tmppatch;
  Scalar dum1,dum2;
  int xMov,yMov;
  int nPatch=3;
 // nn_examples.reserve(dt.bb.size()+1);


  //positive Face
  tmppatch=frame(pbox);
  getPatterns(tmppatch,patch,dum1,dum2);
  nn_examples.push_back(patch);


  //TOP
  tmpBox.x=(pbox.x-pbox.width)>0 ? (pbox.x-pbox.width):0;
  tmpBox.y=(pbox.y-pbox.height)>0 ? (pbox.y-pbox.height):0;
  tmpBox.width=((pbox.x+(2*pbox.width)))< frame.cols ?  ((pbox.x+(2*pbox.width))-tmpBox.x) : (frame.cols-tmpBox.x);
  tmpBox.height=(tmpBox.y + pbox.height)<= pbox.y ? pbox.height : (pbox.y-tmpBox.y);
 // rectangle(printfr,tmpBox,cvScalar(255),2);
 // imshow("printfdeb",printfr);
 // waitKey(0);
  if(tmpBox.height>=(pbox.height/2)){
	  for (int i=0;i<nPatch;i++){
		  rng.operator float();
		  xMov=tmpBox.x+(((float)rng))*(tmpBox.width-pbox.width);
		  workBox.x=xMov;
		  workBox.y=tmpBox.y;
		  workBox.width=pbox.width;
		  workBox.height=tmpBox.height;
		  tmppatch=frame(workBox);
		  getPatterns(tmppatch,patch,dum1,dum2);
		  nn_examples_NX.push_back(patch);
		//  rectangle(printfr,workBox,cvScalar(255),2);
		//  imshow("printfdeb",printfr);
		//  waitKey(0);
	  }
  }

  //RIGHT
  tmpBox.x=(pbox.x+pbox.width);
  tmpBox.y=(pbox.y-pbox.height)>0 ? (pbox.y-pbox.height):0;
  tmpBox.width=(tmpBox.x + pbox.width)<= frame.cols ? pbox.width : (frame.cols-tmpBox.x);
  tmpBox.height=((pbox.y+(2*pbox.height)))< frame.rows ?  ((pbox.y+(2*pbox.height))-tmpBox.y) : (frame.rows-tmpBox.y);
 // rectangle(printfr,tmpBox,cvScalar(255),2);
 // imshow("printfdeb",printfr);
//  waitKey(0);
  if(tmpBox.width>=(pbox.width/2)){
	  for (int i=0;i<nPatch;i++){
		  rng.operator float();
		  yMov=tmpBox.y+(((float)rng))*(tmpBox.height-pbox.height);
		  workBox.x=tmpBox.x;
  		  workBox.y=yMov;
  		  workBox.width=tmpBox.width;
  		  workBox.height=pbox.height;
  		  tmppatch=frame(workBox);
  		  getPatterns(tmppatch,patch,dum1,dum2);
  		  nn_examples_NX.push_back(patch);
	//	  rectangle(printfr,workBox,cvScalar(255),2);
	//	  imshow("printfdeb",printfr);
		//  waitKey(0);
  	  }
    }


  //Bottom
    tmpBox.x=(pbox.x-pbox.width)>0 ? (pbox.x-pbox.width):0;
    tmpBox.y=pbox.y+pbox.height;
    tmpBox.width=((pbox.x+(2*pbox.width)))< frame.cols ?  ((pbox.x+(2*pbox.width))-tmpBox.x) : (frame.cols-tmpBox.x);
    tmpBox.height=(tmpBox.y + pbox.height)<= frame.rows ? pbox.height : (frame.rows-tmpBox.y);
  //  rectangle(printfr,tmpBox,cvScalar(255),2);
  //  imshow("printfdeb",printfr);
    //waitKey(0);
    if(tmpBox.height>=(pbox.height/2)){
  	  for (int i=0;i<nPatch;i++){
  		  rng.operator float();
  		  xMov=tmpBox.x+(((float)rng))*(tmpBox.width-pbox.width);
  		  workBox.x=xMov;
  		  workBox.y=tmpBox.y;
  		  workBox.width=pbox.width;
  		  workBox.height=tmpBox.height;
  		  tmppatch=frame(workBox);
  		  getPatterns(tmppatch,patch,dum1,dum2);
  		  nn_examples_NX.push_back(patch);
	//	  rectangle(printfr,workBox,cvScalar(255),2);
	//	  imshow("printfdeb",printfr);
		 // waitKey(0);
  	  }
    }


    //LEFT
      tmpBox.x=(pbox.x-pbox.width)>0 ? (pbox.x-pbox.width):0;
      tmpBox.y=(pbox.y-pbox.height)>0 ? (pbox.y-pbox.height):0;
      tmpBox.width=(tmpBox.x + pbox.width)<= pbox.x ? pbox.width : (pbox.x-tmpBox.x);
      tmpBox.height=((pbox.y+(2*pbox.height)))< frame.rows ?  ((pbox.y+(2*pbox.height))-tmpBox.y) : (frame.rows-tmpBox.y);
  //    rectangle(printfr,tmpBox,cvScalar(255),2);
  //    imshow("printfdeb",printfr);
     // waitKey(0);
      if(tmpBox.width>=(pbox.width/2)){
    	  for (int i=0;i<nPatch;i++){
    		  rng.operator float();
     		  yMov=tmpBox.y+(((float)rng))*(tmpBox.height-pbox.height);
     		  workBox.x=tmpBox.x;
     		  workBox.y=yMov;
     		  workBox.width=tmpBox.width;
     		  workBox.height=pbox.height;
     		  tmppatch=frame(workBox);
     		  getPatterns(tmppatch,patch,dum1,dum2);
     		  nn_examples_NX.push_back(patch);
    	//	  rectangle(printfr,workBox,cvScalar(255),2);
    	//	  imshow("printfdeb",printfr);

     	  }
       }

      //waitKey(0);


      random_shuffle(nn_examples_NX.begin(),nn_examples_NX.end());

      int half = (int)nn_examples_NX.size()*0.5f;
      nn_examples_NXT.assign(nn_examples_NX.begin()+half,nn_examples_NX.end());
      nn_examples_NX.resize(half);
      for(int i=0;i< nn_examples_NX.size();++i){
    	  nn_examples.push_back(nn_examples_NX[i]);
      }

  /// Classifiers update


  classifier.trainNNN(nn_examples);
  classifier.evaluateThV(nn_examples_NXT);
  //classifier.show();
}



void FeaturesTrack::forceLearnModel(const Mat& img){
	printf("[Force Learning]");
	vector <int> ferns(classifier.getNumStructs());
	float conf;
	Mat patch;
	Mat patch15;
	int numtrees = classifier.getNumStructs();
	float fern_th = classifier.getFernTh();
	float thrP=numtrees*fern_th;
	vector<pair<vector<int>,int> > fern_examples;
	float dummy;
	vector<int> isin;
	int acm=0;

	good_boxes.clear();
	good_boxes.push_back(best_box_index);
    generatePositiveData(img,num_warps_update);
    ferns=pX[0].first;
    conf = classifier.measure_forest(ferns);

	if(conf<thrP){
		fern_examples.clear();
		fern_examples.reserve(pX.size());
		fern_examples.assign(pX.begin(),pX.end());

		while(conf<thrP){
			cout << endl << " acc:" << acm++ << "conf:"<< conf << " thrp:" << thrP ;
			cout << endl;
			classifier.forcedTrainFern(fern_examples,2);
			conf = classifier.measure_forest(ferns);
			//cvWaitKey(0);
		}
	}


	Scalar mean,stdev;
	thrP=classifier.getBestCalculatedThrNN();
	//getPattern(img(lastbox),patch15,mean,stdev);                //  Get pattern within bounding box
	classifier.NNConf(pEx,isin,conf,dummy);  //  Evaluate nearest neighbour classifier
	vector<Mat> nn_examples;
	nn_examples.push_back(pEx);
	while(conf<thrP){
      cout << endl << "Pattern Maching Before Call conf:"<< conf << " thrnn:" << thrP;
      cout << endl;
      classifier.trainNNN(nn_examples);
      classifier.NNConf(pEx,isin,conf,dummy);
     // cvWaitKey(0);
	}

}



// crete the all bounding box
void FeaturesTrack::buildGridSW(const cv::Mat& img, const cv::Rect& box){
  const float SHIFT = 0.1;
  /*
  const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                          0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                          2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
  */
  int scalesnumber=21;
  int hlafscalesnumber=scalesnumber/2;
  float SCALES[scalesnumber];



  int minSizeW=box.height>box.width ? box.width:box.height;
  SCALES[0]=(float)min_win/minSizeW;
  float increment= (1-SCALES[0])/hlafscalesnumber;
  for(int i=1;i<=hlafscalesnumber;++i){
	  SCALES[i]=SCALES[i-1]+increment;
  }

  int maxSizeW=box.height>box.width ? box.height:box.width;
  SCALES[scalesnumber-1]=(float)img.cols/maxSizeW;
  increment=(SCALES[scalesnumber-1]-1)/hlafscalesnumber;
  for(int i=hlafscalesnumber+1;i<scalesnumber;++i){
  	  SCALES[i]=SCALES[i-1]+increment;
    }

  cout << endl << "min_win:" << min_win << " Frme.W:" << img.cols << " frame.H:" << img.rows << " MinW:" << minSizeW << " MaxW:" << maxSizeW << endl;
  for(int i=0;i<scalesnumber;++i){
    	  cout << SCALES[i] << " - ";
      }

  cout << endl;
  //cvWaitKey(0);



  int width, height, min_bb_side;
  //Rect bbox;
  BoundingBox bbox;
  Size scale;
  int sc=0;
  for (int s=0;s<21;s++){
    width = round(box.width*SCALES[s]);
    height = round(box.height*SCALES[s]);
    min_bb_side = min(height,width);
    if (min_bb_side < min_win || width > img.cols || height > img.rows)
      continue;
    scale.width = width;
    scale.height = height;
    scales.push_back(scale);
    for (int y=1;y<img.rows-height;y+=round(SHIFT*min_bb_side)){
      for (int x=1;x<img.cols-width;x+=round(SHIFT*min_bb_side)){
        bbox.x = x;
        bbox.y = y;
        bbox.width = width;
        bbox.height = height;
        bbox.overlap = bbOverlap(bbox,BoundingBox(box));
        bbox.sidx = sc;
        grid.push_back(bbox);
       // if(bbox.overlap>0.9)
       // cout << "sc: " << sc << " s:" << s << ": owelap:" << bbox.overlap << endl ;
      }
    }
    sc++;
  }
}

float FeaturesTrack::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}

void FeaturesTrack::getOverlappingBoxes(int num_closest){
  float max_overlap = 0;
  for (int i=0;i<grid.size();i++){
      if (grid[i].overlap > max_overlap) {
          max_overlap = grid[i].overlap;
          best_box = grid[i];
          best_box_index=i;
      }
      if (grid[i].overlap > 0.6){
          good_boxes.push_back(i);
      }
      else if (grid[i].overlap < bad_overlap){
    //  else if (grid[i].overlap < 0.35){
          bad_boxes.push_back(i);
      }
  }

    //Get the best num_closest (10) boxes and puts them in good_boxes
  if (good_boxes.size()>num_closest){
    std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
    good_boxes.resize(num_closest);
  }

  getBBHull();
}

void FeaturesTrack::getBBHull(){
  int x1=INT_MAX, x2=0;
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<good_boxes.size();i++){
      idx= good_boxes[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  bbhull.x = x1;
  bbhull.y = y1;
  bbhull.width = x2-x1;
  bbhull.height = y2 -y1;
}

bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
  FeaturesTrack t;
    if (t.bbOverlap(b1,b2)<0.45)
      return false;
    else
      return true;
}


int FeaturesTrack::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
  //FIXME: Conditional jump or move depends on uninitialised value(s)
  const int c = dbb.size();
  //1. Build proximity matrix
  Mat D(c,c,CV_32F);
  float d;
  for (int i=0;i<c;i++){
      for (int j=i+1;j<c;j++){
        d = 1-bbOverlap(dbb[i],dbb[j]);
        D.at<float>(i,j) = d;
        D.at<float>(j,i) = d;
      }
  }
  //2. Initialize disjoint clustering
 float L[c-1]; //Level
 int nodes[c-1][2];
 int belongs[c];
 int m=c;
 for (int i=0;i<c;i++){
    belongs[i]=i;
 }
 for (int it=0;it<c-1;it++){
 //3. Find nearest neighbor
     float min_d = 1;
     int node_a, node_b;
     for (int i=0;i<D.rows;i++){
         for (int j=i+1;j<D.cols;j++){
             if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                 min_d = D.at<float>(i,j);
                 node_a = i;
                 node_b = j;
             }
         }
     }
     if (min_d>0.5){
         int max_idx =0;
         bool visited;
         for (int j=0;j<c;j++){
             visited = false;
             for(int i=0;i<2*c-1;i++){
                 if (belongs[j]==i){
                     indexes[j]=max_idx;
                     visited = true;
                 }
             }
             if (visited)
               max_idx++;
         }
         return max_idx;
     }

 //4. Merge clusters and assign level
     L[m]=min_d;
     nodes[it][0] = belongs[node_a];
     nodes[it][1] = belongs[node_b];
     for (int k=0;k<c;k++){
         if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
           belongs[k]=m;
     }
     m++;
 }
 return 1;

}


//prende in ingresso i detected bb e la relativa confidenza e restituisce i cbb e cconf
//overosia raggruppa i dbb il cui overlap è maggiore del 50% e restitusce per ogni gruppo su cbb e dconf
// il bb medio.
void FeaturesTrack::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
  int numbb =dbb.size();
  vector<int> T;
  float space_thr = 0.5;
  int c=1;
  switch (numbb){
  case 1:
    cbb=vector<BoundingBox>(1,dbb[0]);
    cconf=vector<float>(1,dconf[0]);
    return;
    break;
  case 2:
    T =vector<int>(2,0);
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){
      T[1]=1;
      c=2;
    }
    break;
  default:
    T = vector<int>(numbb,0);
    c = partition(dbb,T,(*bbcomp));
    //c = clusterBB(dbb,T);
    break;
  }
  cconf=vector<float>(c);
  cbb=vector<BoundingBox>(c);
 // cout << endl << "C Value:" << c;
  printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){
      float cnf=0;
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){
          if (T[j]==i){
              printf("%d ",i);
              cnf=cnf+dconf[j];
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){
          cconf[i]=cnf/N;
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;
      }
  }
  printf("\n");
 // cvWaitKey(0);
}
//prende in ingresso i detected bb e la relativa confidenza e restituisce i cbb e cconf
//overosia raggruppa i dbb il cui overlap è maggiore del 50% e restitusce per ogni gruppo su cbb e dconf
// il bb medio.
/*
void FeaturesTrack::clusterConfAle(const vector<BoundingBox>& dbb,vector<BoundingBox>& cbb){
  int numbb =dbb.size();
  vector<int> T;
  float space_thr = 0.5;
  int c=1;
  switch (numbb){
  case 1:
    cbb=vector<BoundingBox>(1,dbb[0]);
    return;
    break;
  case 2:
    T =vector<int>(2,0);
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){
      T[1]=1;
      c=2;
    }
    break;
  default:
    T = vector<int>(numbb,0);
    c = partition(dbb,T,(*bbcomp));
    //c = clusterBB(dbb,T);
    break;
  }
 cbb=vector<BoundingBox>(c);
 // cout << endl << "C Value:" << c;
  printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){
          if (T[j]==i){
              printf("%d ",i);
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;
      }
  }
  printf("\n");
 // cvWaitKey(0);
}
*/


void FeaturesTrack::clusterConfAle(const vector<BoundingBox>& dbb,vector<BoundingBox>& cbb){
  int numbb =dbb.size();
  int conf;
  cbb.clear();

  if (numbb==1){
    cbb=vector<BoundingBox>(1,dbb[0]);
    return;
  }

  for(int det=0;det<numbb;++det){
	  for(conf=0;conf<numbb;++conf){

		  if((conf!=det)){
			  if(isInternal((CvRect)dbb[conf],(CvRect)dbb[det])){
				  conf=(2*numbb)-1;
			  }
		  }
	  }

	  if(conf!=(2*numbb)){
		  cbb.push_back(dbb[det]);
	  }
  }


}
