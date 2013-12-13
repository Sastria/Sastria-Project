/*
 * CalcHistPositive.cpp
 *
 *  Created on: 17/gen/2013
 *      Author: Alessandro Romanino
 */

#include "CalcHistPositive.h"
#include <list>

#define MINWS 15


CalcHistPositive::CalcHistPositive(int channelA,int channelB,int w, int h) {
	// TODO Auto-generated constructor stub
	histPositive=Mat(cvSize(channelA,channelB),CV_32FC1,cvScalar(0));
	histPNorm=Mat(cvSize(channelA,channelB),CV_32FC1,cvScalar(0));
	histNNorm=Mat(cvSize(channelA,channelB),CV_32FC1,cvScalar(0));
	fakeProjection=Mat(cvSize(w,h),CV_8UC1,cvScalar(0));


}

void CalcHistPositive::clear(){
	int channelA=histPositive.rows;
	int channelB=histPositive.cols;
	int w=fakeProjection.rows;
	int h=fakeProjection.cols;

	histPositive.release();
	histPNorm.release();
	histNNorm.release();
	fakeProjection.release();
	histPositive=Mat(cvSize(channelA,channelB),CV_32FC1,cvScalar(0));
	histPNorm=Mat(cvSize(channelA,channelB),CV_32FC1,cvScalar(0));
	histNNorm=Mat(cvSize(channelA,channelB),CV_32FC1,cvScalar(0));
	fakeProjection=Mat(cvSize(w,h),CV_8UC1,cvScalar(0));
}

float CalcHistPositive::bbOverlapLastBB(CvRect& box1,CvRect& box2){
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

bool CalcHistPositive::getIfOverlap(CvRect& boxE,CvRect& boxI){
	bool returnV=false;

	if((boxI.x>boxE.x) && (boxI.x<(boxE.x+boxE.width)))
		 returnV=true;
	else if(((boxI.x+boxI.width)>boxE.x) && ((boxI.x+boxI.width)<(boxE.x+boxE.width)))
		 returnV=true;
	else if((boxI.y>boxE.y) && (boxI.y<(boxE.y+boxE.height)))
		 returnV=true;
	else if(((boxI.y+boxI.height)>boxE.y) && ((boxI.y+boxI.height)<(boxE.y+boxE.height)))
		 returnV=true;
	else returnV=false;

	return(returnV);
}


CvRect CalcHistPositive::getNewBox(CvRect& boxE,CvRect& boxI){
	int xMin= boxE.x < boxI.x ? boxE.x : boxI.x;
	int yMin= boxE.y < boxI.y ? boxE.y : boxI.y;
	int xMax= (boxE.x + boxE.width) > (boxI.x + boxI.width) ? (boxE.x +boxE.width): (boxI.x + boxI.width);
	int yMax= (boxE.y + boxE.height) > (boxI.y + boxI.height) ? (boxE.y +boxE.height): (boxI.y + boxI.height);
	return(cvRect(xMin,yMin,xMax-xMin,yMax-yMin));

}

/*
float percentage=0.1;
	  	int wMin=bbnext.width - bbnext.width*percentage;
	  	int wMax=bbnext.width + bbnext.width*percentage;
	  	int hMin=bbnext.height - bbnext.height*percentage;
	  	int hMax=bbnext.height + bbnext.height*percentage;

	  	if((histogramDetectionBlob.width>wMax) || (histogramDetectionBlob.width<wMin) || (histogramDetectionBlob.height>hMax) || (histogramDetectionBlob.height<hMin)){
	  	  Mat imageBig;
	  	  img2.copyTo(imageBig);
	  	  rectangle(imageBig,histogramDetectionBlob,cvScalar(0),2);
	  	  Mat patchMoment(frameTo(histogramDetectionBlob));
	  	  Moments moM=moments(patchMoment);
	  	  int xcc=(moM.m10/moM.m00)+histogramDetectionBlob.x;

	  	  int ycc=(moM.m01/moM.m00)+histogramDetectionBlob.y;


	  	  histogramDetectionBlob.width=histogramDetectionBlob.width > wMax ? wMax :histogramDetectionBlob.width;
	  	  histogramDetectionBlob.width=histogramDetectionBlob.width < wMin ? wMin :histogramDetectionBlob.width;
	  	  histogramDetectionBlob.height=histogramDetectionBlob.height > hMax ? hMax :histogramDetectionBlob.height;
	  	  histogramDetectionBlob.height=histogramDetectionBlob.height < hMin ? hMin :histogramDetectionBlob.height;

	  	  histogramDetectionBlob.x=xcc-(histogramDetectionBlob.width/2);
	  	  histogramDetectionBlob.x=histogramDetectionBlob.x>0 ?   histogramDetectionBlob.x:0;
	  	  histogramDetectionBlob.y=ycc-(histogramDetectionBlob.height/2);
	  	  histogramDetectionBlob.y=histogramDetectionBlob.y>0 ?   histogramDetectionBlob.y:0;
	      rectangle(imageBig,histogramDetectionBlob,cvScalar(255),2);
	      rectangle(imageBig,bbnext,cvScalar(255),1);
	      imshow("cambioDim",imageBig);
	  	 // cvWaitKey(0);

	  	}
*/


/*
 * int pixXMin,pixXMax,pixYMin,pixYMax;
	int rStart,cStart,rEnd,cEnd;
	Vec3b pixel;
	int colorSize=1;



	backProjectionCopy.at<uchar>(row,col)=0;


	if(backProjection.at<uchar>(row,col)!=0){
		realEliminaterFrame.at<uchar>(row,col)=100;
		pixel=frameHSV.at<Vec3b>(row,col);
		pixXMin=pixel[0]-colorSize > 0 ? pixel[0]-colorSize : 0;
		pixXMax=pixel[0]+colorSize < histPNorm.cols ? pixel[0]+colorSize : (histPNorm.cols-1);
		pixYMin=pixel[1]-colorSize > 0 ? pixel[1]-colorSize : 0;
		pixYMax=pixel[1]+colorSize < histPNorm.rows ? pixel[1]+colorSize : (histPNorm.rows-1);
		histPNorm(cvRect(pixXMin,pixYMin,pixXMax-pixXMin,pixYMax-pixYMin))=cvScalar(254);
	}

	rStart=(row-1) >0 ? (row-1):0;
	cStart=(col-1) >0 ? (col-1):0;
	rEnd=(row+1) < (backProjectionCopy.rows-1) ? (row+1) : row;
	cEnd=(col+1) < (backProjectionCopy.cols-1) ? (col+1) : col;
	for(int r=rStart;r<=rEnd;++r)
		for(int c=cStart;c<=cEnd;++c){
			if(backProjectionCopy.at<uchar>(r,c)!=0)
				removeFakeColorFromRealFrame(frameHSV,backProjection,backProjectionCopy,realEliminaterFrame,r,c);
		}
 *
 */


void CalcHistPositive::removeFakeColorFromRealFrame(cv::Mat& backProjection,CvRect bBox){

 int rows=backProjection.rows;
 int cols=backProjection.cols;
 int val;
 fakeProjection=cvScalar(0);
 backProjection(bBox)=cvScalar(0);

  for(int row=0; row<rows ; ++row)
	  for(int col=0; col<cols ; ++col){
		  //val=fakeProjection.at<uchar>(row,col);
		  if(backProjection.at<uchar>(row,col)!=0)
			  fakeProjection.at<uchar>(row,col)=254;
	  }

 // imshow("frame projection fake",fakeProjection);

}



void CalcHistPositive::getRealFrameProjection(cv::Mat& frameHSV,cv::Mat& backProjection,CvRect lastboxTrack){

	float percentageInternal=0.2;//0.35; //0.15
	float percentageExternal=0.5;//0.65; //0.15
	CvRect InternalBox=cvRect(0,0,0,0);
	CvRect ExternalBox=cvRect(0,0,0,0);
	Mat removeColor,backProjectionCopy;
	int wPercentage=lastboxTrack.width*percentageInternal;
	int hPercentage=lastboxTrack.height*percentageInternal;
	Mat colorFilter,selectionFilter;


	InternalBox.x=lastboxTrack.x+(wPercentage);
	InternalBox.y=lastboxTrack.y+(hPercentage);
	InternalBox.width=lastboxTrack.width -(2*wPercentage);
	InternalBox.height=lastboxTrack.height -(2*hPercentage);


		wPercentage=lastboxTrack.width*percentageExternal;
		hPercentage=lastboxTrack.height*percentageExternal;

		ExternalBox.x=lastboxTrack.x-(wPercentage) > 0  ? lastboxTrack.x-(wPercentage) : 0;
		ExternalBox.y=lastboxTrack.y-(hPercentage) > 0 ? lastboxTrack.y-(hPercentage): 0;
		ExternalBox.width=lastboxTrack.width +(2*wPercentage);
		ExternalBox.width=(ExternalBox.width + ExternalBox.x) < frameHSV.cols ? ExternalBox.width : (frameHSV.cols-ExternalBox.x-1);
		ExternalBox.height=lastboxTrack.height +(2*hPercentage);
		ExternalBox.height=(ExternalBox.height + ExternalBox.y) < frameHSV.rows ? ExternalBox.height : (frameHSV.rows-ExternalBox.y-1);

/*
	Mat bckP;
	backProjection.copyTo(bckP);
	dilate(bckP,bckP,Mat());
	erode(bckP,bckP,Mat());
*/



/*
	std::vector<std::vector<cv::Point> > contours;
	findContours(backProjection,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	backProjection=cvScalar(0);
	drawContours(backProjection,contours,-1,cv::Scalar(254),CV_FILLED);

*/

	backProjection.copyTo(colorFilter);
	backProjection.copyTo(selectionFilter);

	//colorFilter=cvScalar(0);

	Mat patchMom(backProjection(ExternalBox));
	Moments mom=moments(patchMom);
	int xc=(mom.m10/mom.m00);
	xc=xc+ExternalBox.x;
	int yc=(mom.m01/mom.m00);
	yc =yc + ExternalBox.y;


	int lbttXMin=lastboxTrack.x;
	int lbttXMax=lastboxTrack.x+lastboxTrack.width;
	int lbttYMin=lastboxTrack.y;
	int lbttYMax=lastboxTrack.y+lastboxTrack.height;
	int row= yc - lastboxTrack.height/4 ;
	int col;
	int count=0;
	int lSegment;
	int rMin=row;
	int steep=2;


	lSegment=lastboxTrack.width;
	int colStart= xc - lSegment/2;
	int rminValue=lSegment;
	do {
		count=0;
		 for(col=0;col<lSegment;++col){
			 if(backProjection.at<uchar>(row,colStart+col)>0)
				 ++count;
		 }
		 count+=1.5*(abs(row-lbttYMin));
		 if((count<rminValue)){
			 rMin=row;
			 rminValue=count;
		 }
		 row-=steep;
		 line(selectionFilter,cvPoint(colStart,row),cvPoint(colStart+lSegment,row),cvScalar(255),1);
		 cout << endl << "1:" << row+steep;
	} while ((row>=ExternalBox.y));

	//line(selectionFilter,cvPoint(InternalBox.x,rMin),cvPoint(InternalBox.x+InternalBox.width,rMin),cvScalar(255),3);
	//imshow("selectionFilter",selectionFilter);
	//cvWaitKey(0);
	InternalBox.y=rMin;

	col=xc-lastboxTrack.width/4;
	lSegment=lastboxTrack.height;
	int cMin=col;
	int cMinValue=lSegment;
	int rowStart=yc -lSegment/2;
	 do {
		 count=0;
		 for(row=0;row<lSegment;++row){
			 if(backProjection.at<uchar>(rowStart+row,col)>0)
				 ++count;
		 }
		 count+=1.5*(abs(col-lbttXMin));
		 if((count<cMinValue)){
		 	cMin=col;
		 	cMinValue=count;
		 }
		 col-=steep;
		 line(selectionFilter,cvPoint(col,rowStart),cvPoint(col,rowStart+lSegment),cvScalar(255),1);
		 cout << endl << "2:" << col+steep;
	} while ((col>=ExternalBox.x));

	// line(selectionFilter,cvPoint(cMin,InternalBox.y),cvPoint(cMin,InternalBox.y+InternalBox.height),cvScalar(255),3);
	// imshow("selectionFilter",selectionFilter);
	// cvWaitKey(0);
	InternalBox.x=cMin;


	col=xc + (lastboxTrack.width)/4;
	lSegment=lastboxTrack.height;
	cMin=col;
	cMinValue=lSegment;
	rowStart=rowStart= yc -lSegment/2;
	 do {
		 count=0;
		 for(row=0;row<lSegment;++row){
			 if(backProjection.at<uchar>(rowStart+row,col)>0)
				 ++count;
		 }

		 count+=1.5*(abs(col-lbttXMax));
		 if((count<cMinValue)){
		 	cMin=col;
		 	cMinValue=count;
		 }

		 col+=steep;
		 line(selectionFilter,cvPoint(col,rowStart),cvPoint(col,rowStart+lSegment),cvScalar(255),1);
		 cout << endl << "3:" << col-steep;
	} while ((col<(ExternalBox.x+ExternalBox.width)));

	// line(selectionFilter,cvPoint(cMin,InternalBox.y),cvPoint(cMin,InternalBox.y+InternalBox.height),cvScalar(255),3);
	// imshow("selectionFilter",selectionFilter);
	// cvWaitKey(0);
	InternalBox.width=cMin-InternalBox.x;

	row=yc + lastboxTrack.height/4;
	lSegment=lastboxTrack.width;
	rMin=row;
	rminValue=lSegment;
	colStart=xc-lSegment/2;
	 do {
		 count=0;
		 for(col=0;col<lSegment;++col){
			 if(backProjection.at<uchar>(row,colStart+col)>0)
				 ++count;
		 }

		 count+=1.5*(abs(row-lbttYMax));
		 if((count<rminValue)){
			rMin=row;
			rminValue=count;
		}
		 row+=steep;
		 line(selectionFilter,cvPoint(colStart,row),cvPoint(colStart+col,row),cvScalar(255),1);
		 cout << endl << "4:" << row-steep;
	} while ((row<(ExternalBox.y+ExternalBox.height)));


	 //line(selectionFilter,cvPoint(InternalBox.x,rMin),cvPoint(InternalBox.x+InternalBox.width,rMin),cvScalar(255),3);


	 InternalBox.height=rMin-InternalBox.y;
	 rectangle(selectionFilter,InternalBox,cvScalar(255),3);
	// imshow("selectionFilter",selectionFilter);



	 backProjectionCopy.release();
	 backProjection.copyTo(backProjectionCopy);

	 backProjection=cvScalar(0);
	 Mat fromCopy(backProjectionCopy(InternalBox)),toCopy(backProjection(InternalBox));
	 fromCopy.copyTo(toCopy);


	// imshow("prima",backProjection);
	// myErodeDilate(toCopy,5,5);
	// imshow("dopo",backProjection);
	// dilate(backProjection,backProjection,Mat());
	// erode(backProjection,backProjection,er);





	//cvWaitKey(0);backProjection

/*

	 percentageExternal=0.3;

	 wPercentage=InternalBox.width*percentageExternal;
	 hPercentage=InternalBox.height*percentageExternal;

	 ExternalBox.x=InternalBox.x-(wPercentage) > 0  ? InternalBox.x-(wPercentage) : 0;
	 ExternalBox.y=InternalBox.y-(hPercentage) > 0 ? InternalBox.y-(hPercentage): 0;
	 ExternalBox.width=InternalBox.width +(2*wPercentage);
	 ExternalBox.width=(ExternalBox.width + ExternalBox.x) < frameHSV.cols ? ExternalBox.width : (frameHSV.cols-ExternalBox.x);
	 ExternalBox.height=InternalBox.height +(2*hPercentage);
	 ExternalBox.height=(ExternalBox.height + ExternalBox.y) < frameHSV.rows ? ExternalBox.height : (frameHSV.rows-ExternalBox.y);

	 backProjection.copyTo(backProjectionCopy);
	 backProjectionCopy(InternalBox)=cvScalar(0);
	 rectangle(backProjectionCopy,InternalBox,cvScalar(254),1);
	 rectangle(backProjectionCopy,ExternalBox,cvScalar(0),1);

	 //frameHSV.copyTo(removeColor,backProjection);



	 Vec3b pixel;
	 histPNorm=cvScalar(0);
	// imshow("histPNormPrima",histPNorm);
	 Mat realEliminaterFrame;
	 backProjection.copyTo(realEliminaterFrame);
	 cout << endl << "Remove FAKEP";
	 removeFakeColorFromRealFrame(frameHSV,backProjection,backProjectionCopy,realEliminaterFrame,InternalBox.y,InternalBox.x);
	 cout << endl << "Remove FAKED";
	 imshow("pixeleliminati",realEliminaterFrame);


	 float val;
	 colorFilter=cvScalar(0);
	 backProjection.copyTo(colorFilter);
	 int examined=0,eliminated=0;
	 backProjectionCopy=cvScalar(0);
	 backProjection.copyTo(backProjectionCopy);
	// backProjection.release();
	// backProjection=Mat::zeros(cvSize(image.cols,image.rows),CV_8UC1);
	 for(int col=ExternalBox.x;col<=(ExternalBox.x+ExternalBox.width);++col){
	 	for(int row=ExternalBox.y;row<=(ExternalBox.y+ExternalBox.height); ++row){
	 		if(backProjection.at<uchar>(row,col)!=0){
	 			examined++;
	 			pixel=frameHSV.at<Vec3b>(row,col);
	 			val=histPNorm.at<float>(pixel[1],pixel[0]);
	 			if(val > 0){
	 				colorFilter.at<uchar>(row,col)=150;
	 				backProjection.at<uchar>(row,col)=0;
	 				eliminated++;
	 			}
	 		}
	 	}
	 }


	 if(eliminated>(examined)){
		 backProjection=cvScalar(0);
		 backProjectionCopy.copyTo(backProjection);
	 }else{

		 std::vector<std::vector<cv::Point> > contours;
		 imshow("BackP prima di contorno",backProjection);
		 findContours(backProjection,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		 backProjection=cvScalar(0);
		 drawContours(backProjection,contours,-1,cv::Scalar(254),CV_FILLED);
		 dilate(backProjection,backProjection,Mat());

	 }

	// cvWaitKey(0);
*/


	// imshow("histPNormDopo",histPNorm);
	 //rectangle(colorFilter,ExternalBox,cvScalar(254),1);
	// rectangle(colorFilter,InternalBox,cvScalar(254),1);
	// imshow("removeColor",removeColor);
	// imshow("Colorilter",colorFilter);




}



/*
void CalcHistPositive::getRealFrameProjection(cv::Mat& frameHSV,cv::Mat& backProjection,CvRect lastboxTrack){

	float percentageInternal=0.1;//0.35; //0.15
	float percentageExternal=0.4;//0.65; //0.15
	CvRect InternalBox=cvRect(0,0,0,0);
	CvRect ExternalBox=cvRect(0,0,0,0);
	Mat removeColor,backProjectionCopy;
	int wPercentage=lastboxTrack.width*percentageInternal;
	int hPercentage=lastboxTrack.height*percentageInternal;
	Mat colorFilter,selectionFilter;


	backProjection.copyTo(colorFilter);
	backProjection.copyTo(selectionFilter);
	backProjection.copyTo(backProjectionCopy);

	//colorFilter=cvScalar(0);

	InternalBox.x=lastboxTrack.x-(wPercentage) > 0  ? lastboxTrack.x-(wPercentage) : 0;
	InternalBox.y=lastboxTrack.y-(hPercentage) > 0 ? lastboxTrack.y-(hPercentage): 0;
	InternalBox.width=lastboxTrack.width +(2*wPercentage);
	InternalBox.width=(InternalBox.width + InternalBox.x) < frameHSV.cols ? InternalBox.width : (frameHSV.cols-InternalBox.x);
	InternalBox.height=lastboxTrack.height +(2*hPercentage);
	InternalBox.height=(InternalBox.height + InternalBox.y) < frameHSV.rows ? InternalBox.height : (frameHSV.rows-InternalBox.y);

	wPercentage=InternalBox.width*percentageExternal;
	hPercentage=InternalBox.height*percentageExternal;

	ExternalBox.x=InternalBox.x-(wPercentage) > 0  ? InternalBox.x-(wPercentage) : 0;
	ExternalBox.y=InternalBox.y-(hPercentage) > 0 ? InternalBox.y-(hPercentage): 0;
	ExternalBox.width=InternalBox.width +(2*wPercentage);
	ExternalBox.width=(ExternalBox.width + ExternalBox.x) < frameHSV.cols ? ExternalBox.width : (frameHSV.cols-ExternalBox.x-1);
	ExternalBox.height=InternalBox.height +(2*hPercentage);
	ExternalBox.height=(ExternalBox.height + ExternalBox.y) < frameHSV.rows ? ExternalBox.height : (frameHSV.rows-ExternalBox.y-1);



	int row=InternalBox.y;
	int col;
	int count=0;
	int lSegment;
	int rMin=row;
	int steep=2;

	lSegment=InternalBox.x+InternalBox.width;
	int rminValue=lSegment;
	do {
		count=0;
		 for(col=InternalBox.x;col<lSegment;++col){
			 if(backProjection.at<uchar>(row,col)>0)
				 ++count;
		 }
		 if(count<rminValue){
			 rMin=row;
			 rminValue=count;
		 }
		 row-=steep;
		 line(selectionFilter,cvPoint(InternalBox.x,row),cvPoint(InternalBox.x+InternalBox.width,row),cvScalar(255),1);
		 cout << endl << "1:" << row+steep;
	} while ((row>=ExternalBox.y));

	//line(selectionFilter,cvPoint(InternalBox.x,rMin),cvPoint(InternalBox.x+InternalBox.width,rMin),cvScalar(255),3);
	//imshow("selectionFilter",selectionFilter);
	//cvWaitKey(0);
	InternalBox.y=rMin;

	col=InternalBox.x;
	lSegment=InternalBox.y+InternalBox.height;
	int cMin=col;
	int cMinValue=lSegment;
	 do {
		 count=0;
		 for(row=InternalBox.y;row<lSegment;++row){
			 if(backProjection.at<uchar>(row,col)>0)
				 ++count;
		 }

		 if(count<cMinValue){
		 	cMin=col;
		 	cMinValue=count;
		 }
		 col-=steep;
		 line(selectionFilter,cvPoint(col,InternalBox.y),cvPoint(col,InternalBox.y+InternalBox.height),cvScalar(255),1);
		 cout << endl << "2:" << col+steep;
	} while ((col>=ExternalBox.x));

	// line(selectionFilter,cvPoint(cMin,InternalBox.y),cvPoint(cMin,InternalBox.y+InternalBox.height),cvScalar(255),3);
	// imshow("selectionFilter",selectionFilter);
	// cvWaitKey(0);
	InternalBox.x=cMin;


	col=(InternalBox.x+InternalBox.width);
	lSegment=InternalBox.y+InternalBox.height;
	cMin=col;
	cMinValue=lSegment;
	 do {
		 count=0;
		 for(row=InternalBox.y;row<lSegment;++row){
			 if(backProjection.at<uchar>(row,col)>0)
				 ++count;
		 }

		 if(count<cMinValue){
		 	cMin=col;
		 	cMinValue=count;
		 }

		 col+=steep;
		 line(selectionFilter,cvPoint(col,InternalBox.y),cvPoint(col,InternalBox.y+InternalBox.height),cvScalar(255),1);
		 cout << endl << "3:" << col-steep;
	} while ((col<(ExternalBox.x+ExternalBox.width)));

	// line(selectionFilter,cvPoint(cMin,InternalBox.y),cvPoint(cMin,InternalBox.y+InternalBox.height),cvScalar(255),3);
	// imshow("selectionFilter",selectionFilter);
	// cvWaitKey(0);
	InternalBox.width=cMin-InternalBox.x;

	row=(InternalBox.y+InternalBox.height);
	lSegment=InternalBox.x+InternalBox.width;
	rMin=row;
	rminValue=lSegment;
	 do {
		 count=0;
		 for(col=InternalBox.x;col<lSegment;++col){
			 if(backProjection.at<uchar>(row,col)>0)
				 ++count;
		 }
		 if(count<rminValue){
			rMin=row;
			rminValue=count;
		}
		 row+=steep;
		 line(selectionFilter,cvPoint(InternalBox.x,row),cvPoint(InternalBox.x+InternalBox.width,row),cvScalar(255),1);
		 cout << endl << "4:" << row-steep;
	} while ((row<(ExternalBox.y+ExternalBox.height)));


	 //line(selectionFilter,cvPoint(InternalBox.x,rMin),cvPoint(InternalBox.x+InternalBox.width,rMin),cvScalar(255),3);


	 InternalBox.height=rMin-InternalBox.y;
	 rectangle(selectionFilter,InternalBox,cvScalar(255),3);
	 imshow("selectionFilter",selectionFilter);
	 backProjectionCopy(InternalBox)=cvScalar(0);

	//cvWaitKey(0);



	 percentageExternal=0.4;

	 wPercentage=InternalBox.width*percentageExternal;
	 hPercentage=InternalBox.height*percentageExternal;

	 ExternalBox.x=InternalBox.x-(wPercentage) > 0  ? InternalBox.x-(wPercentage) : 0;
	 ExternalBox.y=InternalBox.y-(hPercentage) > 0 ? InternalBox.y-(hPercentage): 0;
	 ExternalBox.width=InternalBox.width +(2*wPercentage);
	 ExternalBox.width=(ExternalBox.width + ExternalBox.x) < frameHSV.cols ? ExternalBox.width : (frameHSV.cols-ExternalBox.x);
	 ExternalBox.height=InternalBox.height +(2*hPercentage);
	 ExternalBox.height=(ExternalBox.height + ExternalBox.y) < frameHSV.rows ? ExternalBox.height : (frameHSV.rows-ExternalBox.y);


	 rectangle(backProjectionCopy,InternalBox,cvScalar(254),1);
	 rectangle(backProjectionCopy,ExternalBox,cvScalar(0),1);

	 frameHSV.copyTo(removeColor,backProjection);

	 //TOP of ROI
		 /*   ---------------------------
		  *   -*************************-
		  *   -   -------------------	-
		  *   -	  -                 -   -
		  *   -	  -	                -   -
		  *   -   -	                -   -
		  */
/*
		 for(int row=ExternalBox.y;row < InternalBox.y;++row)
			 for(int col=ExternalBox.x;col < (ExternalBox.x+ExternalBox.width);++col) {
				 backProjection.at<uchar>(row,col)=0;
		 }
*/

		 //LEFT of ROI
		 /*   ---------------------------
		 *    - *                       -
		 *    - * -------------------	-
		 *    -	* -                 -   -
		 *    -	* -	                -   -
		 *    - * -	                -   -
		*/
/*
		 for(int row=InternalBox.y;row < (ExternalBox.height+ExternalBox.y);++row)
			 for(int col=ExternalBox.x;col <= (InternalBox.x);++col) {
					backProjection.at<uchar>(row,col)=0;

		 }
*/
		 //Right of ROI
		 /*   ---------------------------
		 *    -                       * -
		 *    -   ------------------- *	-
		 *    -	  -                 - * -
		 *    -	  -	                - * -
		 *    -   -	                - * -
		*/
/*
		 for(int row=InternalBox.y;row < (ExternalBox.height+ExternalBox.y);++row)
			 for(int col=(InternalBox.x+InternalBox.width);col <= (ExternalBox.x+ExternalBox.width);++col) {
				backProjection.at<uchar>(row,col)=0;
		 }
*/

		 //Bottom of ROI
		 /*
		  *   -	  -                 -   -
		  *   -	  -                 -   -
		  *   -	  -                 -   -
		  *   -	  -                 -   -
		  *   -   -------------------   -
		  *   -*************************-
		  *   ---------------------------
		  */
/*
		 for(int row=(InternalBox.y+InternalBox.height);row < (ExternalBox.height+ExternalBox.y);++row)
			 for(int col=ExternalBox.x;col < (ExternalBox.x+ExternalBox.width);++col) {
				 backProjection.at<uchar>(row,col)=0;
		     }


	 rectangle(colorFilter,ExternalBox,cvScalar(254),1);
	 rectangle(colorFilter,InternalBox,cvScalar(254),1);
	 imshow("removeColor",removeColor);
	 imshow("Colorilter",colorFilter);
	 imshow("backprojectionRealFrame",backProjection);



}
*/



int CalcHistPositive::calcDensity(cv::Mat detection,cv::Mat& debug){
	 int contValues=0;

	 for(int row=0;row <detection.rows;++row)
		 for(int col=0;col <detection.cols;++col){
			 if(detection.at<uchar>(row,col)>0){
				 ++contValues;
				 debug.at<uchar>(row,col)=254;
			 }else
				 debug.at<uchar>(row,col)=125;
		 }

		 return(contValues);
}


//void CalcHistPositive::getMinMaxXYPointR(cv::Mat& image,cv::Mat& colorImage,CvRect lastboxTrack,int& xLeft,int& xRight, int& yTop, int& yBottom){
void CalcHistPositive::getDetectionByHistogram(cv::Mat& frameHSV,cv::Mat& image,cv::Mat& backProjection,cv::Mat& detection,cv::Mat& negativeDetection,CvRect lastboxTrack,CvRect& blobDetection){
	/*
	xLeft=image.cols;
	xRight=0;
	yTop=image.rows;
	yBottom=0;
	*/

	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point> > contoursUnified;
	Mat allBackP;
	detection.release();
	negativeDetection.release();


	 CvRect nextBox=cvRect(0,0,0,0);
	 float percentage=0.25; //0.15
	 int wPercentage=lastboxTrack.width*percentage;
	 int hPercentage=lastboxTrack.height*percentage;

	 nextBox.x=lastboxTrack.x-(wPercentage) > 0  ? lastboxTrack.x-(wPercentage) : 0;
	 nextBox.y=lastboxTrack.y-(hPercentage) > 0 ? lastboxTrack.y-(hPercentage): 0;
	 nextBox.width=lastboxTrack.width +(2*wPercentage);
	 nextBox.width=(nextBox.width + nextBox.x) < image.cols ? nextBox.width : (image.cols-nextBox.x);
	 nextBox.height=lastboxTrack.height +(2*hPercentage);
	 nextBox.height=(nextBox.height + nextBox.y) < image.rows ? nextBox.height : (image.rows-nextBox.y);


 	 image.copyTo(allBackP);
     allBackP=cvScalar(0);
     allBackP(nextBox)=cvScalar(255);
	 image.copyTo(detection,allBackP);


	 int contValues=0;

	 for(int row=nextBox.y;row <(nextBox.y+nextBox.height);++row)
		 for(int col=nextBox.x;col <(nextBox.x+nextBox.width);++col){
			 if(detection.at<uchar>(row,col)>0){
				 ++contValues;
			 }
		 }

	 if(contValues<((nextBox.height*nextBox.width)/5)){ //5
		 dilate(detection,detection,Mat());
	 }

	allBackP=cvScalar(0);
	findContours(detection,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	CvRect bbox;
	blobDetection=cvRect(0,0,0,0);
	float bestOverlap=0,overLap;
	int bestC=0;
	vector<int> goodBox;
    int xMin=image.cols,yMin=image.rows,yMax=0,xMax=0;
    bool visited=false;


	for (int i=0; i< contours.size();++i){
      	 bbox=boundingRect(contours[i]);
      	 overLap=bbOverlapLastBB(lastboxTrack,bbox);

      	 if(overLap>=bestOverlap){
      	   	 bestOverlap=overLap;
      	   	 blobDetection=bbox;
      	   	 bestC=i;
      	 }

         if(overLap>=0.1){
        	 goodBox.push_back(i);
        	 xMin= xMin < bbox.x ? xMin :bbox.x ;
        	 yMin= yMin < bbox.y ? yMin :bbox.y ;
        	 xMax= xMax > (bbox.x+bbox.width) ? xMax :(bbox.x+bbox.width) ;
        	 yMax= yMax > (bbox.y+bbox.height) ? yMax :(bbox.y+bbox.height) ;
        	 visited=true;
         }
       	 drawContours(allBackP,contours,i,cv::Scalar(255),CV_FILLED);
        // rectangle(unifiedContours,bbox,cvScalar(255),CV_FILLED);
	}

	if(visited){
		blobDetection.x=xMin;
		blobDetection.y=yMin;
		blobDetection.width=xMax-xMin;
		blobDetection.height=yMax-yMin;
	}

	detection=cvScalar(0);
	if((blobDetection.width<=MINWS) || (blobDetection.height<=MINWS)){

		int xc=blobDetection.x+(blobDetection.width/2);
		int yc=blobDetection.y+(blobDetection.height/2);

		float aRactio=(float)nextBox.width/(float)nextBox.height;

		if(aRactio>1){
			blobDetection.height=MINWS;
			blobDetection.width=MINWS*aRactio;
			//cout << endl << "w > h ractio=" << aRactio;
		}
		else{
			blobDetection.width=MINWS;
			blobDetection.height=(MINWS/aRactio);
			//cout << endl << "h > w ractio=" << aRactio;
		}

		blobDetection.x=xc-(blobDetection.width/2);
		blobDetection.x=blobDetection.x >0 ? blobDetection.x : 0;
		blobDetection.y=yc-(blobDetection.height/2);
		blobDetection.y=blobDetection.y >0 ? blobDetection.y : 0;

		detection(blobDetection)=cvScalar(254);

	} else{
		drawContours(detection,contours,bestC,cv::Scalar(255),CV_FILLED);
		for (int i=0; i< goodBox.size();++i){
			drawContours(detection,contours,goodBox[i],cv::Scalar(255),CV_FILLED);
		}

		}

	detection.copyTo(backProjection);
	allBackP.copyTo(detection);
	detection(blobDetection)=cvScalar(254);

	negativeDetection=Mat::zeros(cvSize(detection.cols,detection.rows),CV_8UC1);
	allBackP=cvScalar(0);
	backProjection.copyTo(allBackP);
	allBackP(cvRect(((blobDetection.width/4)+blobDetection.x),((blobDetection.height/4)+blobDetection.y),(blobDetection.width/2),(blobDetection.height/2)))=cvScalar(255);
	getBackProjectionNegative(allBackP,negativeDetection);
}

void CalcHistPositive::calcNegative(Mat& image, Mat& weightPixel){
	Vec3b pixel;
	int weight;

	Mat printingVal;

	image.copyTo(printingVal);
	printingVal=cvScalar(0);
	histNNorm=cvScalar(0);

	for(int r=0;r<image.rows;r++){
		//cout << endl;
		for(int c=0; c< image.cols; ++c){
			pixel=image.at<Vec3b>(r,c);
			//cout << endl << "R:" <<(int)pixel[0] << " G:" <<(int) pixel[1]<< " B:" << (int)pixel[2];
			if((pixel[0]!=1) || (pixel[1]!=2) || (pixel[2]!=3) ){
				weight=weightPixel.at<uchar>(r,c);
				//cout << "- " << weight;
				histNNorm.at<float>(pixel[1],pixel[0])+=weight;
				pixel[0]=weight*20;
				pixel[1]=0;
				pixel[2]=0;
				printingVal.at<Vec3b>(r,c)=pixel;
			} else {
				//cout << "- B";
				pixel[0]=0;
				pixel[1]=0;
				pixel[2]=255;
				printingVal.at<Vec3b>(r,c)=pixel;
			}
		}
	}
	normalize(histNNorm,histNNorm,100,0,NORM_L1);

	//imshow("TestColor",printingVal);
	//cvWaitKey(0);
}

void CalcHistPositive::calcNegative(Mat& image, Mat& weightPixel, Mat& detection){
	Vec3b pixel;
	int weight;

	histNNorm=cvScalar(0);

//	int zero=0,uno=0,due=0;

	for(int r=0;r<image.rows;r++){
		//cout << endl;
		for(int c=0; c< image.cols; ++c){
			weight=weightPixel.at<uchar>(r,c);
			if(detection.at<uchar>(r,c)>0){
				pixel=image.at<Vec3b>(r,c);
				histNNorm.at<float>(pixel[1],pixel[0])+=weight;
			}
		}
	}
	normalize(histNNorm,histNNorm,100,0,NORM_L1);
}



void CalcHistPositive::calcPositive(Mat& image, Mat& weightPixel){
	Vec3b pixel;
	//int maxWeight=10;
	int weight;
	CvPoint centre=cvPoint(image.cols/2,image.rows/2);
	//int maxDist=sqrt(pow(centre.x,2)+pow(centre.y,2))/maxWeight;


	histPNorm=cvScalar(0);
	for(int r=0;r<image.rows;r++){
		//cout << endl;
		for(int c=0; c< image.cols; ++c){
			pixel=image.at<Vec3b>(r,c);
			//cout << endl << "R:" <<(int)pixel[0] << " G:" <<(int) pixel[1]<< " B:" << (int)pixel[2];
			if((pixel[0]!=1) || (pixel[1]!=2)|| (pixel[2]!=3) ){
				weight=weightPixel.at<uchar>(r,c);
				cout << "- " << weight;
				histPNorm.at<float>(pixel[1],pixel[0])+=weight;
			}else cout << "- B";
		}
	}

	normalize(histPNorm,histPNorm,100,0,NORM_L1);
	//cvWaitKey(0);
}

void CalcHistPositive::calcPositive(Mat& image, Mat& weightPixel, Mat detection){
	Vec3b pixel;
	//int maxWeight=10;
	int weight;
	histPNorm=cvScalar(0);
	for(int r=0;r<image.rows;r++){
		//cout << endl;
		for(int c=0; c< image.cols; ++c){
			weight=weightPixel.at<uchar>(r,c);
			if(detection.at<uchar>(r,c)>0){
				pixel=image.at<Vec3b>(r,c);
				histPNorm.at<float>(pixel[1],pixel[0])+=weight;
			}
		}
	}

	normalize(histPNorm,histPNorm,100,0,NORM_L1);
}





void CalcHistPositive::updadeHistPositive(Mat& image,Mat& histogramDetection,Mat& negativeHistogramDetection,CvRect selectionBoxPositive){
	//histPositive=Mat(cvSize(histP.cols,histP.rows),histP.type(),cvScalar(0));
		//histPNorm=Mat(cvSize(histP.cols,histP.rows),histP.type(),cvScalar(0));
		//histNNorm=Mat(cvSize(histP.cols,histP.rows),histP.type(),cvScalar(0));

		Mat positiveWeight=Mat(cvSize(selectionBoxPositive.width,selectionBoxPositive.height),CV_8UC1,cvScalar(0));
		Mat negativeWeight=Mat(cvSize(image.cols,image.rows),CV_8UC1,cvScalar(0));
		//Mat printW=Mat(cvSize(selectionBoxPositive.width,selectionBoxPositive.height),CV_8UC1,cvScalar(0));
		int numWeight=10;
		int yOffset=(selectionBoxPositive.height/2)/numWeight;
		int xOffset=(selectionBoxPositive.width/2)/numWeight;

		CvRect tmpW=cvRect(0,0,selectionBoxPositive.width,selectionBoxPositive.height);
		positiveWeight=cvScalar(100);
		//rectangle(printW,tmpW,cvScalar(255),1);
		for(int count=1;count <=numWeight;++count){
	       positiveWeight(tmpW)=cvScalar(count);
	       tmpW.x+=xOffset;
	       tmpW.y+=yOffset;
	       tmpW.width-=2*xOffset;
	       tmpW.height-=2*yOffset;
	      // rectangle(printW,tmpW,cvScalar(255),1);
		}


		//imshow("printw",printW);
		//imshow("pesi",positiveWeight);

		CvRect selectionBoxNegative=cvRect(selectionBoxPositive.x,selectionBoxPositive.y,selectionBoxPositive.width,selectionBoxPositive.height);

		selectionBoxNegative.x=selectionBoxPositive.x+(selectionBoxPositive.width/4);
		selectionBoxNegative.y=selectionBoxPositive.y+(selectionBoxPositive.height/4);
		selectionBoxNegative.width=selectionBoxPositive.width - (selectionBoxPositive.width/2);
		selectionBoxNegative.height=selectionBoxPositive.height -(selectionBoxPositive.height/2);


		/*
		 * selectionBoxNegative.x=selectionBoxNegative.x-(selectionBoxNegative.width/4) > 0  ? selectionBoxNegative.x-(selectionBoxNegative.width/4) : 0;
		selectionBoxNegative.y=selectionBoxNegative.y-(selectionBoxNegative.height/4) > 0 ? selectionBoxNegative.y-(selectionBoxNegative.height/4): 0;
		selectionBoxNegative.width=selectionBoxNegative.width +(selectionBoxNegative.width/2);
		selectionBoxNegative.width=(selectionBoxNegative.width + selectionBoxNegative.x) < image.cols ? selectionBoxNegative.width : (image.cols-selectionBoxNegative.x);
		selectionBoxNegative.height=selectionBoxNegative.height +(selectionBoxNegative.height/2);
		selectionBoxNegative.height=(selectionBoxNegative.height + selectionBoxNegative.y) < image.rows ? selectionBoxNegative.height : (image.rows-selectionBoxNegative.y);
		 *
		 */

		tmpW=cvRect(0,0,image.cols,image.rows);
		xOffset=selectionBoxNegative.x > (tmpW.width-(selectionBoxNegative.x+selectionBoxNegative.width)) ? selectionBoxNegative.x :(tmpW.width-(selectionBoxNegative.x+selectionBoxNegative.width));
		yOffset=selectionBoxNegative.y > (tmpW.height-(selectionBoxNegative.y+selectionBoxNegative.height)) ? selectionBoxNegative.y :(tmpW.height-(selectionBoxNegative.y+selectionBoxNegative.height));
		//numWeight/=2;
		xOffset/=(numWeight);
		yOffset/=(numWeight);


		vector<CvRect> boxesW;
		int xboxesW,yboxesW,cWidth,cHeight;
		int xS=selectionBoxNegative.x;
		int yS=selectionBoxNegative.y;

		int xE=xS + selectionBoxNegative.width;
		int yE=yS + selectionBoxNegative.height;




		for(int count=1;count <numWeight;++count){
			xS=(xS-xOffset) > 0 ? (xS-xOffset) : 0 ;
			xE=(xE + xOffset) < image.cols ? (xE + xOffset) : (image.cols);
			yS=(yS-yOffset) > 0 ? (yS-yOffset) : 0 ;
			yE=(yE + yOffset) < image.rows ? (yE + yOffset) : (image.rows);
			boxesW.push_back(cvRect(xS,yS,xE-xS,yE-yS));

		}
		boxesW.push_back(cvRect(tmpW.x,tmpW.y,tmpW.width,tmpW.height));


		//Mat printWN=Mat(cvSize(image.cols,image.rows),CV_8UC1,cvScalar(0));
		//negativeWeight=cvScalar(1);

		//rectangle(printWN,tmpW,cvScalar(255),1);
		//rectangle(printWN,selectionBoxPositive,cvScalar(120),3);
		for(int count=1; boxesW.size() > 0; ++count){
			//negativeWeight(tmpW)=cvScalar((numWeight+1)-count);
			tmpW=boxesW.back();
			negativeWeight(tmpW)=cvScalar(count);
			boxesW.pop_back();
			// rectangle(printWN,tmpW,cvScalar(255),1);
		}


		Mat positiveImage=image(selectionBoxPositive);
		calcPositive(positiveImage,positiveWeight,histogramDetection(selectionBoxPositive));
		//positiveImage=image(selectionBoxNegative);
		//positiveImage=cvScalar(1,2,3);
		calcNegative(image,negativeWeight,negativeHistogramDetection);


		float v1=0,v2=0,vOld,VNew;
        int pixXMin,pixXMax,pixYMin,pixYMax;
        int colorSize=1;

		for(int r=0;r<histPNorm.rows;++r){
			for(int c=0;c<histPNorm.cols;++c){
				v1=histPNorm.at<float>(r,c);
				v2=histNNorm.at<float>(r,c);
				if(!((v1==0) && (v2==0))){
					vOld=histPositive.at<float>(r,c);
					VNew=vOld+ (((v1-v2)>0)? 1:-1);
					if((VNew>vOld) && (VNew>15)){
						vOld=VNew;
					}
					else{
						vOld=VNew;
					}
					histPositive.at<float>(r,c)=vOld;
				}
			}
		}

}

void CalcHistPositive::updadeHistPositive(Mat& image,CvRect selectionBoxPositive){
	Mat positiveWeight=Mat(cvSize(selectionBoxPositive.width,selectionBoxPositive.height),CV_8UC1,cvScalar(0));
	Mat negativeWeight=Mat(cvSize(image.cols,image.rows),CV_8UC1,cvScalar(0));

	int numWeight=10;
	int yOffset=(selectionBoxPositive.height/2)/numWeight;
	int xOffset=(selectionBoxPositive.width/2)/numWeight;

	CvRect tmpW=cvRect(0,0,selectionBoxPositive.width,selectionBoxPositive.height);
	positiveWeight=cvScalar(1);

	for(int count=1;count <=numWeight;++count){
       tmpW.x+=xOffset;
       tmpW.y+=yOffset;
       tmpW.width-=2*xOffset;
       tmpW.height-=2*yOffset;
       positiveWeight(tmpW)=cvScalar(count);
	}

	CvRect selectionBoxNegative=cvRect(selectionBoxPositive.x,selectionBoxPositive.y,selectionBoxPositive.width,selectionBoxPositive.height);

	xOffset=selectionBoxNegative.x > (tmpW.width-(selectionBoxNegative.x+selectionBoxNegative.width)) ? selectionBoxNegative.x :(tmpW.width-(selectionBoxNegative.x+selectionBoxNegative.width));
	yOffset=selectionBoxNegative.y > (tmpW.height-(selectionBoxNegative.y+selectionBoxNegative.height)) ? selectionBoxNegative.y :(tmpW.height-(selectionBoxNegative.y+selectionBoxNegative.height));
	xOffset/=(numWeight);
	yOffset/=(numWeight);

	int wS=(selectionBoxNegative.width ) + (2*xOffset)*numWeight;
	int hS=(selectionBoxNegative.height) + (2*yOffset)*numWeight;

	tmpW=cvRect(0,0,image.cols,image.rows);

	for(int count=1;count <=numWeight;++count){

		negativeWeight(tmpW)=cvScalar(count);
	       tmpW.x+=xOffset;
	       tmpW.y+=yOffset;
	       wS-=2*xOffset;
	       tmpW.width= (tmpW.x+wS) < image.cols ? wS : (image.cols-tmpW.x);
	       hS-=2*yOffset;
	       tmpW.height= (tmpW.y+hS) < image.rows ? hS : (image.rows-tmpW.y);
	}
	Mat positiveImage=image(selectionBoxPositive);
	calcPositive(positiveImage,positiveWeight);
	positiveImage=image(selectionBoxNegative);
	positiveImage=cvScalar(1,2,3);
	calcNegative(image,negativeWeight);

	float v1=0,v2=0,vOld;

	for(int r=0;r<histPNorm.rows;++r){
		for(int c=0;c<histPNorm.cols;++c){
			v1=histPNorm.at<float>(r,c);
			v2=histNNorm.at<float>(r,c);
			vOld=histPositive.at<float>(r,c);
			vOld+=((v1-v2)>0)? 1:-1;
			histPositive.at<float>(r,c)=vOld;
		}
	}
}


void CalcHistPositive::calcBackProjection(Mat& image, Mat& backProjection, Mat& backProjectionFake){
	Vec3b pixel;
	float val;
	int fake;
	backProjection.release();
	backProjection=Mat::zeros(cvSize(image.cols,image.rows),CV_8UC1);
	backProjection.copyTo(backProjectionFake);

	for(int r=0;r<image.rows;r++){
		for(int c=0; c< image.cols; ++c){
				pixel=image.at<Vec3b>(r,c);
				val=histPositive.at<float>(pixel[1],pixel[0]);
				fake=fakeProjection.at<uchar>(r,c);
				if((val > 0)){
					backProjectionFake.at<uchar>(r,c)=254;
					if (fake==0)
						backProjection.at<uchar>(r,c)=254;
				}
		}
	}
}


void CalcHistPositive::calcBackProjection(Mat& image, Mat& backProjection){
	Vec3b pixel;
	float val;
	backProjection.release();
	backProjection=Mat::zeros(cvSize(image.cols,image.rows),CV_8UC1);
	for(int r=0;r<image.rows;r++){
		for(int c=0; c< image.cols; ++c){
				pixel=image.at<Vec3b>(r,c);
				val=histPositive.at<float>(pixel[1],pixel[0]);
				if(val > 0)
					backProjection.at<uchar>(r,c)=254;
		}
	}
}



void CalcHistPositive::getBackProjectionNegative(Mat& image, Mat& imageNegative){
	  threshold(image,imageNegative,0,255,CV_THRESH_BINARY_INV);
}

Mat& CalcHistPositive::getHistogram(){
	return(histPositive);
}

CalcHistPositive::~CalcHistPositive() {
	// TODO Auto-generated destructor stub
}
