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
#include <iostream>
#include <sstream>
#include "FeaturesTrack.h"
#include <stdio.h>
//#include "IntegralImage.h"
#include "CalcHistPositive.h"
#include "DetectionToRecognition.h"
#include "ValidateDetection.h"
#include <time.h>
#include "DataSharing.h"



using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = false;
bool rep = false;
bool fromfile=false;
string video;
string videoOut;
string xmlData;
string ymlData;
string xmlFaceList;

void mySurf( Mat& img_object, Mat& img_scene);
void mySift( Mat& img_object, Mat& img_scene);


void readBB(char* file){
  ifstream bb_file (file);
  string line;
  getline(bb_file,line);
  istringstream linestream(line);
  string x1,y1,x2,y2;
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}



void read_options(VideoCapture& capture,FileStorage &fs,int device){

	IplImage *menuScelta=cvCreateImage(cvSize(500,500),IPL_DEPTH_8U,3);

	CvFont font1;

	// Text variables
	double hscale = 1.0;
	double vscale = 0.8;
	double shear = 0.2;
	int thickness = 2;
	int line_type = 8;
	short int keyPressed;
	cvInitFont(&font1,CV_FONT_HERSHEY_DUPLEX,hscale,vscale,shear,thickness,line_type);

	if(device>1){

	//font1=fontQt("Times",-1,cvScalar(255),1,1,1);
	//addText(menuScelta, "Scegli il Video da tracciare", Point(100,50), font);

	cvPutText(menuScelta,"Scegli il Video:",cvPoint(10,30),&font1,cvScalar(255,0,0));
	cvPutText(menuScelta,"1. Videolina",cvPoint(10,80),&font1,cvScalar(0,0,255));
	cvPutText(menuScelta,"2. Preliminary",cvPoint(10,120),&font1,cvScalar(0,0,255));
	cvPutText(menuScelta,"3. test2",cvPoint(10,160),&font1,cvScalar(0,0,255));

	cvShowImage("Scelta Video",menuScelta);
	keyPressed=waitKey(0);

	printf("\nChar=%c int=%i\n",keyPressed,keyPressed);
	//cvWaitKey(0);

	switch (keyPressed) {
		case '1':video = ".././Video/DemoProva.mov";
				 videoOut = ".././VideoOut/DemoProva";
				 xmlData=".././Data/DemoProva.mov.xml";
				 xmlFaceList=".././Data/faceListDemoProva.mov.xml";
				 break;
		case '2':video = ".././Video/Prelim.mpg";
					 videoOut = ".././VideoOut/Prelim";
						xmlData=".././Data/Prelim.mpg.xml";
						xmlFaceList=".././Data/faceListPrelim.mpg.xml";
					 break;
		case '3':video = ".././Video/test2.mpg";
							 videoOut = ".././VideoOut/test2";
								xmlData=".././Data/test2.mpg.xml";
								xmlFaceList=".././Data/faceListtest2.mpg.xml";
							 break;

		default: video = ".././Video/videodemo.mov";
				 videoOut = ".././VideoOut/videodemo";
				 break;
	}

	//video = ".././Video/alec.mp4";
	//video = ".././Video/Panda.mp4";
	capture.open(video);
	fromfile = true;
	}


	fs.open(".././Parameters/parameters.yml", FileStorage::READ);
	tl = true;

}


void read_optionsAutomatics(VideoCapture& capture,FileStorage &fs,int device,string fileVideoName){

	if(device>1){
		video = "./" + fileVideoName;
		//video = ".././Video/" + fileVideoName;
  	    videoOut = "./VideoOut/"+fileVideoName;
		xmlData="./Data/" + fileVideoName +".xml";
		ymlData="./Data/" + fileVideoName +".yml";
		xmlFaceList="./Data/faceList" + fileVideoName +".xml";
   	    capture.open(video);
	    fromfile = true;
	}
	fs.open("./Parameters/parameters.yml", FileStorage::READ);
	tl = true;
}


/*
void * soketFun(void *);

void * soketFun(void *){

}*/


int main(int argc, char **argv){

	Mat objSRF,sceneSRF;
	clock_t start,end;
	double time;
	bool oldObj=true;
	DataSharing soketTpcConn;
	int totalFrames;
	int percentageProgress=-1;
	int newP;
	bool connectedToSocket;
	int detectionPoints;
	int trackerSegmentSize;

	//pthread_t tId;
	//pthread_create(&tId,NULL,soketFun,NULL);



	  VideoCapture capture;
	 // capture.open(0);
	  FileStorage fs;
	  // IntegralImage integralImg;
	  //Read options
	  if(argc>1)
		  read_optionsAutomatics(capture,fs,2,(argv[1]));
	  else
	  read_options(capture,fs,2);


	  if(soketTpcConn.connectToSoket("127.0.0.1")){
		  cout << "Connesso" ;
		  cout << endl;
		  connectedToSocket=true;
		 // waitKey(0);
	  }else{
		  cout << "NON Connesso" ;
		  cout << endl;
		  connectedToSocket=false;
		 // waitKey(0);
	  }


	  DetectionToRecognition dtToRe(xmlData,xmlFaceList);
	  dtToRe.readFile();
	  dtToRe.loadFile();
	  totalFrames=dtToRe.getTotalFrames();
	  FileStorage metrics(ymlData, FileStorage::APPEND);

	 // soketTpcConn.writeToSoket(dtToRe.getTotalFrames());

	  //Init camera
	  if (!capture.isOpened())
	  {
		cout << "capture device failed to open!" << endl;
	    return 1;
	  }
	  else cout << "video File OPENED!" << endl;
	  //Register mouse callback to draw the bounding box
	  cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
	  cvSetMouseCallback( "TLD", mouseHandler, NULL );
	  //TLD framework
	  FeaturesTrack tld;
	  //Read parameters file
	  Mat frame;
	  Mat last_gray;
	  Mat first;
	  if (fromfile){
	      capture >> frame;
	      cvtColor(frame, last_gray, CV_RGB2GRAY,1);
	      frame.copyTo(first);
	  }else{
	      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
	      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	  }

  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");
  //TLD initialization


  //CalcHistPositive histTrack(180,256,frame.cols,frame.rows);
  //Mat backProjection,frameAfterHist;
  //int hbins = 90, sbins=125;
  //int histSize[] = {hbins,sbins};
  // hue varies from 0 to 179, see cvtColor
  float hranges[] = { 0, 180 };
  float sranges[] = { 0, 256 };
  // saturation varies from 0 (black-gray-white) to
  // 255 (pure spectrum color)
  //const float* ranges[] = {hranges,sranges};
  MatND hist;
  // we compute the histogram from the 0-th and 1-st channels
 // int channels[] = {0,1},dims=2;

  Mat frameFrom,frameTo,frameProjection,frameProjectionFake;
 // CvRect selectionBox=cvRect(0,0,0,0);
 // CvRect selectionBoxInternal=cvRect(0,0,0,0);

  ///Run-time
  Mat current_gray;
  CvRect pbox;
  vector<Point2f> pts1;
  vector<Point2f> pts2;
  bool status=true;
  int frames = 1;
  int detections = 1;
 // int readKey=0,WaitCode=33;

  CvSize sizeOriginal = cvSize((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));

//  double maxV;
  Mat backProjectionInpaint,backPj;
  Mat histogramDetection,negativeHistogramDetection;
  CvRect histogramDetectionBlob;
  CvSize winMediaS=cvSize(9,9);
  CvSize winMediaSmall=cvSize(9,9);

  BBoxAtFrame *nFrame;
  int cFrame;
  int cFafe=1;
  VideoWriter writerVideoFrame;
  dtToRe.initDetToRec();
  String videoResults;
  String segmentName;
  stringstream intotoStr;
  Mat frameValidation;
  ValidateDetection validateDetection;
  CvRect validationArea;
  Mat ncc(1,1,CV_32F);
  float nccP,nccpSum;

  //int nccPCount;
  CvRect surfBlob=Rect(0,0,0,0);

  CvSize2D32f avgBlob=cvSize2D32f(0,0);
  vector<CvSize> avgBlobs;
  int lastFtameInit;
  int xc,yc;
  CvRect multiresRect=cvRect(0,0,0,0);
  Mat multiResObj;
  int framesFailure;
  int trackNotValid;
  tld=FeaturesTrack();
  tld.read(fs.getFirstTopLevelNode());

  start=clock();

  Mat newFace,newFace15;
  Scalar dummyS,dummyM;
  float matchValue;
  Mat smoothGaussian;
 // Mat erod(cvSize(7,7),CV_8UC1);

  cout << endl << "Face " << totalFrames;
  int totalFramesTracked=0;


  NEXTFACE:
  while(dtToRe.bBoxListIsNotVoid()){

 	  nFrame=&(dtToRe.getNextBBox());
	  box=nFrame->getBBoxCrop();

	  cFrame=nFrame->getFrameId();
	  capture.set(CV_CAP_PROP_POS_FRAMES,cFrame-1);
	 //capture.set(CV_CAP_PROP_POS_FRAMES,100);
	  capture >> frame;
	  cvtColor(frame, last_gray, CV_RGB2GRAY);
	  erode(last_gray,last_gray,Mat());
	  GaussianBlur(last_gray,smoothGaussian,Size(9,9),1.5);
	  equalizeHist(smoothGaussian,last_gray);




	  pbox=cvRect(box.x,box.y,box.width,box.height);

	  avgBlob.height=pbox.height;
	  avgBlob.width=pbox.width;
	  avgBlobs.clear();
	  for(int tmpCo=0;tmpCo<20;++tmpCo)
		  avgBlobs.push_back(cvSize(pbox.width,pbox.height));

	  //frame.copyTo(frameInit);
	  //rectangle(frameInit,pbox,cvScalar(255,0,0),2);
	  //imshow("Init",frameInit);

	  rectangle(frameValidation,pbox,cvScalar(255,0,0),2);
	  validationArea.x=pbox.x-(pbox.width/4) > 0  ? pbox.x-(pbox.width/4) : 0;
	  validationArea.y=pbox.y-(pbox.height/4) > 0 ? pbox.y-(pbox.height/4): 0;
	  validationArea.width=pbox.width +(pbox.width/2);
	  validationArea.width=(validationArea.width + validationArea.x) < frame.cols ? validationArea.width : (frame.cols-validationArea.x);
	  validationArea.height=pbox.height +(pbox.height/2);
	  validationArea.height=(validationArea.height + validationArea.y) < frame.rows ? validationArea.height : (frame.rows-validationArea.y);
	  frame(validationArea).copyTo(frameValidation);

	  /*validateDetection.detectFacialFeatures(frameValidation);
	  validateDetection.detectFacialFeaturesAlt2(frameValidation);
	  validateDetection.detectEyes(frameValidation);
	  validateDetection.detectMouth(frameValidation);
	  validateDetection.detectNose(frameValidation);
	  validateDetection.detectGlass(frameValidation);
	  */
	  cout << endl << "FrameId " << cFrame << endl;
	  if((validateDetection.validateDetection(frameValidation))){
		  videoResults.clear();
		  videoResults.append(videoOut);
		  intotoStr.str("");
		  intotoStr << cFafe;
		  videoResults.append(intotoStr.str());
		  videoResults.append(".avi");
		  writerVideoFrame=VideoWriter(videoResults,CV_FOURCC('X','V','I','D'),capture.get(CV_CAP_PROP_FPS),sizeOriginal);
          last_gray(pbox).copyTo(newFace);
  	  	  tld.InitLastFiveFace(last_gray,pbox,fs.getFirstTopLevelNode());
		  cFafe++;
		  detectionPoints=0;
		  trackerSegmentSize=0;
		  segmentName.clear();
		  segmentName.append("SegmentoN_");
		  segmentName.append(intotoStr.str());
		 // imshow("ValidationFrame",frameValidation);
		 // waitKey(0);
	  }else{
		  dtToRe.setFaceAlreadyCheckedOrInvalid(cFrame,((CvRect&)pbox));
		  //imshow("ValidationFrame",frameValidation);
		  //cvWaitKey(50);
		  goto NEXTFACE;
	  }

	  //if(cFrame==990)
	 	//		  waitKey(0);
	  cout << endl << "FrameID="<<cFrame;
	  cout << endl;


	  last_gray.copyTo(frameFrom);
	  //cvtColor(frame,frameHsv,CV_RGB2HSV);
	  //frameHsv.copyTo(frameHsvHistTrack);

	 // selectionBox.x=box.x-(box.width/4) > 0  ? box.x-(box.width/4) : 0;
	 // selectionBox.y=box.y-(box.height/4) > 0 ? box.y-(box.height/4): 0;
	 // selectionBox.width=box.width +(box.width/2);
	 // selectionBox.width=(selectionBox.width + selectionBox.x) < frame.cols ? selectionBox.width : (frame.cols-selectionBox.x);
	 // selectionBox.height=box.height +(box.height/2);
	 // selectionBox.height=(selectionBox.height + selectionBox.y) < frame.rows ? selectionBox.height : (frame.rows-selectionBox.y);
	 // frameHsv(selectionBox)=cvScalar(0);

	  dtToRe.addFace();
	  dtToRe.addFacePosition(cFrame,((CvRect&)pbox));
	  dtToRe.setFaceAlreadyCheckedOrInvalid(cFrame,((CvRect&)pbox));
	  detections=0;
	  frames=cFrame;
	  lastFtameInit=-11;
	  framesFailure=0;
	  trackNotValid=0;
	  oldObj=true;

	  while(capture.read(frame)){
		  cFrame++;
		 // if(cFrame==266)
		 // waitKey(0);
		 // cvtColor(frame,frameHsv,CV_RGB2HSV);
		  //frameHsv.copyTo(frameHsvHistTrack);
		 // imshow("back hist positive",frameProjection);
		  if(connectedToSocket){
			  newP=(((float)cFrame/totalFrames)*100)+1;
			  if(percentageProgress!=newP){
				  soketTpcConn.writeToSoket(newP);
				  percentageProgress=newP;
			  }
		  }
		  cvtColor(frame, current_gray, CV_RGB2GRAY);
		  erode(current_gray,current_gray,Mat());
		  GaussianBlur(current_gray,smoothGaussian,Size(9,9),1.5);
		  equalizeHist(smoothGaussian,current_gray);


		  matchTemplate(last_gray,current_gray,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
		  nccP=(((float*)ncc.data)[0]+1)*0.5; //rescale interval -1:1 to 0:1
          cout << endl << "nccP: " << nccP;
          cout << endl;

          surfBlob.x=pbox.x;
          surfBlob.y=pbox.y;
          surfBlob.width=pbox.width;
          surfBlob.height=pbox.height;

          if(nccP<0.95){
        	  cout << endl;
        	 status=false;
          } else if(!(dtToRe.ckeckIfTracking(cFrame,((CvRect&)pbox)))){

        	  	  	  if(tld.myTrk(last_gray,current_gray,((CvRect&)pbox))){
        	 			  framesFailure=0;
        	 			  oldObj=true;
        	 			  status=true;
        	 		  }
        	  	  	  /*else{
        	 			 pbox.x=surfBlob.x;
        	 			 pbox.y=surfBlob.y;
        	 			 pbox.width=surfBlob.width;
        	 			 pbox.height=surfBlob.height;
        	 			 if(oldObj)
        	 			      last_gray(pbox).copyTo(objSRF);
        	 			 current_gray.copyTo(sceneSRF);

        	 			 if(tld.SurfExtraction(objSRF,last_gray,sceneSRF,(CvRect&)pbox,oldObj)){
        	 				 framesFailure=0;
        	 				 oldObj=true;
        	 				 status=true;
        	 				 objSRF.release();
        	 				 sceneSRF.release();
        	 			 }*/
        	  	  	    else{
        	 				 oldObj=false;
        	 				 framesFailure++;
        	 				 if(framesFailure>=5)
        	 					 status=false;
        	 				 else
        	 					 status=true;

        	 				 pbox.x=surfBlob.x;
        	 				 pbox.y=surfBlob.y;
        	 				 pbox.width=surfBlob.width;
        	 				 pbox.height=surfBlob.height;
        	 			 }

        	 		  //}



        	  	  	trackerSegmentSize++;
        	  	  	current_gray(pbox).copyTo(newFace);
        	  	  	tld.getPatterns(newFace,newFace15,dummyM,dummyS);
        	  	  	matchValue=tld.getMatching(newFace15);
        	  	  //	cout << endl << "#### matchValue:" << matchValue << " thresholdMatch:" << tld.getMatchMean() << " ####" << endl;
        	  	  	if(matchValue<(tld.getMatchMean())){
        	  	  		trackNotValid++;
        	  	  		if(trackNotValid>=5)
        	  	  			status=false;
        	  	  	//	cout << endl << "frameFailure: " << trackNotValid;
        	  	  	//	cout << endl;
        	  	  		//waitKey(0);

        	  	  	}else{
        	  	  		trackNotValid=0;
        	  	  		tld.learnModel(current_gray,pbox);
        	  	  	}

          }else{
        	 // current_gray(pbox).copyTo(newFace);
        	  tld.learnModel(current_gray,pbox);
        	  status=true;
        	  trackerSegmentSize++;
        	  detectionPoints++;
          }

		  if (status){
			 // tld.showExamples();
			  cout << endl << "avg bb: " << pbox.width << " " << pbox.height << " " << (int)avgBlob.width << " "<< (int)avgBlob.height<< " " << ((int)avgBlob.width/2) << " "<< ((int)avgBlob.height/2) << endl;
			  if((pbox.width<((int)avgBlob.width/2)) ||(pbox.width>((int)avgBlob.width*2)) || (pbox.height<((int)avgBlob.height/2))|| (pbox.height>((int)avgBlob.height*2))){

				  rectangle(frame,pbox,cvScalar(125),2);

				  xc=pbox.x+(pbox.width)/2;
				  yc=pbox.y+(pbox.height)/2;

				  pbox.x=((xc-(surfBlob.width/2))+surfBlob.x)/2;
				  pbox.x=pbox.x>0?pbox.x:0;
				  pbox.y=((yc-(surfBlob.height/2))+surfBlob.y)/2;
				  pbox.y=pbox.y>0?pbox.y:0;


				  pbox.width=(pbox.x+surfBlob.width) < frame.cols? surfBlob.width : (frame.cols-pbox.x);
				  pbox.height=(pbox.y+surfBlob.height) < frame.rows? surfBlob.height : (frame.rows-pbox.y);
				  cout << endl << "ERROE ROI";
				  cout << endl;
				 // waitKey(0);
			  }

			  dtToRe.addFacePosition(cFrame,((CvRect&)pbox));

			  /*
			  selectionBox.x=pbox.x + (pbox.width/4) > 0  ? pbox.x-(pbox.width/4) : 0;
			  selectionBox.y=pbox.y-(pbox.height/4) > 0 ? pbox.y-(pbox.height/4): 0;
			  selectionBox.width=pbox.width +(pbox.width/2);
			  selectionBox.width=(selectionBox.width + selectionBox.x) < frameHsv.cols ? selectionBox.width : (frameHsv.cols-selectionBox.x);
			  selectionBox.height=pbox.height +(pbox.height/2);
			  selectionBox.height=(selectionBox.height + selectionBox.y) < frameHsv.rows ? selectionBox.height : (frameHsv.rows-selectionBox.y);

			  selectionBoxInternal.x=pbox.x-(pbox.width) > 0  ? pbox.x-(pbox.width) : 0;
			  selectionBoxInternal.y=pbox.y-(pbox.height) > 0 ? pbox.y-(pbox.height): 0;
			  selectionBoxInternal.width=pbox.width +(pbox.width*2);
			  selectionBoxInternal.width=(selectionBoxInternal.width + selectionBoxInternal.x) < frameHsv.cols ? selectionBoxInternal.width : (frameHsv.cols-selectionBoxInternal.x);
			  selectionBoxInternal.height=pbox.height +(pbox.height*2);
			  selectionBoxInternal.height=(selectionBoxInternal.height + selectionBoxInternal.y) < frameHsv.rows ? selectionBoxInternal.height : (frameHsv.rows-selectionBoxInternal.y);

			  frameHsv(selectionBox)=cvScalar(0);
			  */

			//  imshow("SelectionColor",frameHsv);


			  drawPoints(frame,pts1,Scalar(255,0,0));
			  drawPoints(frame,pts2,Scalar(0,255,0));
			  drawBox(frame,pbox);
			  detections++;
			  avgBlob.width=((((20*avgBlob.width)-avgBlobs.back().width)+pbox.width)/20);
			  avgBlob.height=((((20*avgBlob.height)-avgBlobs.back().height)+pbox.height)/20);
			  avgBlobs.pop_back();
			  avgBlobs.insert(avgBlobs.begin(),cvSize(pbox.width,pbox.height));



		  } else {
			  	dtToRe.setFaceAlreadyCheckedOrInvalid(cFrame,((CvRect&)pbox));
			    dtToRe.closeFace();
			    metrics << segmentName << "{:" << "Tracked Frames Nr" << trackerSegmentSize << "Detected Faces Nr" << detectionPoints << "}";
			    metrics << "segmentName" << segmentName << "Sequence Size" << trackerSegmentSize << "Valid Detection" << detectionPoints;
			    cout << "RES: " << "segmentName" << segmentName << "Sequence Size" << trackerSegmentSize << "Valid Detection" << detectionPoints;
			    totalFramesTracked+=trackerSegmentSize;
			    goto NEXTFACE;
		  }



		  writerVideoFrame.write(frame);
		  imshow("TLD", frame);
		  //imshow("backProjection",backProjection);

		  current_gray.copyTo(last_gray);
		  frameTo.copyTo(frameFrom);
		  pts1.clear();
		  pts2.clear();
		  frames++;
		  printf("Detection rate: \n%d/%d/%d\n",detections,frames,cFrame);
		  cvWaitKey(5);

	  }

  }

  end=clock();
  time=((double)(end-start))/CLOCKS_PER_SEC ;
 // time=time/60;
  if(connectedToSocket)
	  soketTpcConn.release();

  if(trackerSegmentSize>=1){
	  metrics << segmentName << "{:" << "Tracked Frames Nr" << trackerSegmentSize << "Detected Faces Nr" << detectionPoints << "}";
	  //metrics << "segmentName" << segmentName << "Sequence Size" << trackerSegmentSize << "Valid Detection" << detectionPoints;
      cout << "RES: " << "segmentName:" << segmentName << ", Sequence Size:" << trackerSegmentSize << ", Valid Detection" << detectionPoints;
  }
  cout << endl << "Tempo esecuzione: " << time;


  metrics << "Tracked Faces Nr" << totalFramesTracked;
  metrics << "Tracker Execution time in Seconds" << ((int)time);


  dtToRe.closeDetToRec();
  fclose(bb_file);
  return 0;
}



