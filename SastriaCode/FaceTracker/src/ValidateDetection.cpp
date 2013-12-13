/*
 * ValidateDetection.cpp
 *
 *  Created on: 04/giu/2013
 *      Author: Alessandro Romanino
 */

#include "ValidateDetection.h"

ValidateDetection::ValidateDetection() {

	cascade.load("./Parameters/haar/haarcascade_frontalface_alt.xml");
	cascadeAlt2.load("./Parameters/haar/haarcascade_frontalface_alt2.xml");
	cascade_eye.load("./Parameters/haar/haarcascade_mcs_eyepair_big.xml");
	cascade_nose.load("./Parameters/haar/haarcascade_mcs_nose.xml");
	cascade_mouth.load("./Parameters/haar/haarcascade_mcs_mouth.xml");
	cascade_glass.load("./Parameters/haar/haarcascade_eye_tree_eyeglasses.xml");
	cascade_eyeBig.load("./Parameters/haar/haarcascade_mcs_eyepair_big.xml");
	cascade_eyeSmall.load("./Parameters/haar/haarcascade_mcs_eyepair_small.xml");
	cascade_profFace.load("./Parameters/haar/haarcascade_profileface.xml");
}

bool ValidateDetection::detectMouth( Mat& img)
{
	vector<Rect> mouth;
	bool returnV=false;

	cascade_mouth.detectMultiScale(img,mouth);

    for( int i = 0; i < mouth.size(); i++ ){
       rectangle(img,mouth[i],cvScalar(255,0,0),1);
    }
     //end mouth detection

    if(mouth.size()>0) returnV=true;

    return(returnV);

}

/*Eyes detection*/
bool ValidateDetection::detectGlass( Mat& img)
{
	vector<Rect> eyes;
	bool returnV=false;

		cascade_glass.detectMultiScale(img,eyes);

	    for( int i = 0; i < eyes.size(); i++ ){
	       rectangle(img,eyes[i],cvScalar(255,255,255),1);
   	    }

	if(eyes.size()>0) returnV=true;

	return(returnV);
}

/*Eyes detection*/
bool ValidateDetection::detectEyes( Mat& img)
{
	vector<Rect> eyes;
	bool returnV=false;

		cascade_eye.detectMultiScale(img,eyes);

	    for( int i = 0; i < eyes.size(); i++ ){
	       rectangle(img,eyes[i],cvScalar(0,255,0),1);
   	    }

    if(eyes.size()>0) returnV=true;

    return(returnV);
}

/*Nose detection*/
bool ValidateDetection::detectNose( Mat& img)
{
	vector<Rect> noses;
	bool returnV=false;

	cascade_nose.detectMultiScale(img,noses);

	for( int i = 0; i < noses.size(); i++ ){
	      rectangle(img,noses[i],cvScalar(0,0,255),1);
	}

	if(noses.size()>0) returnV=true;

	return(returnV);
}


/*ear detection*/
bool ValidateDetection::detectEyeBig( Mat& img)
{
	vector<Rect> eyeB;
	bool returnV=false;

	cascade_eyeBig.detectMultiScale(img,eyeB);

	for( int i = 0; i < eyeB.size(); i++ ){
	      rectangle(img,eyeB[i],cvScalar(0,120,255),1);
	}

	if(eyeB.size()>0) returnV=true;

	return(returnV);
}

/*ear detection*/
bool ValidateDetection::detectEyeSmall( Mat& img)
{
	vector<Rect> eyeS;
	bool returnV=false;

	cascade_eyeSmall.detectMultiScale(img,eyeS);

	for( int i = 0; i < eyeS.size(); i++ ){
	      rectangle(img,eyeS[i],cvScalar(0,120,255),1);
	}

	if(eyeS.size()>0) returnV=true;

	return(returnV);
}


bool ValidateDetection::detectFacialFeatures(Mat& img)
{
	vector<Rect> faces;
	bool returnV=false;


	cascade.detectMultiScale(img, faces);

		for( int i = 0; i < faces.size(); i++ ){
		      rectangle(img,faces[i],cvScalar(255,255,0),1);
		}

    if(faces.size()>0) returnV=true;

    return(returnV);
}


bool ValidateDetection::detectFacialFeaturesAlt2(Mat& img)
{
	vector<Rect> faces;
	bool returnV=false;


	cascadeAlt2.detectMultiScale(img, faces);

		for( int i = 0; i < faces.size(); i++ ){
		      rectangle(img,faces[i],cvScalar(0,255,255),1);
		}
	if(faces.size()>0) returnV=true;

	return(returnV);
}


bool ValidateDetection::detectProfFace(Mat& img)
{
	vector<Rect> faces;
	bool returnV=false;


	cascade_profFace.detectMultiScale(img, faces);

		for( int i = 0; i < faces.size(); i++ ){
		      rectangle(img,faces[i],cvScalar(0,255,255),1);
		}
	if(faces.size()>0) returnV=true;

	return(returnV);
}

bool ValidateDetection::validateDetection(Mat& img)
{
	vector<Rect> faces;
	bool returnV=false;
	int points=0;
	CvRect validationArea;
	CvRect findArea;
	//Mat validFrame;

	cascadeAlt2.detectMultiScale(img, faces);

		for( int i = 0; i < faces.size(); i++ ){
		      rectangle(img,faces[i],cvScalar(0,255,255),1);
		}

	if(faces.size()==0){
		cascade.detectMultiScale(img, faces);
		for( int i = 0; i < faces.size(); i++ ){
	      rectangle(img,faces[i],cvScalar(0,255,255),1);
		}
	}

	if(faces.size()>0){
		findArea=faces[0];
		validationArea.x=findArea.x+(findArea.width/8) ;
		validationArea.y=findArea.y+(findArea.height/8);
		validationArea.width=findArea.width -(findArea.width/4);
		validationArea.height=findArea.height -(findArea.height/4);
		Mat validFrame(img(validationArea));
		rectangle(img,validationArea,cvScalar(50,100,50),1);
		points++;
		if(detectEyes(validFrame)) ++points;
		if(detectGlass(validFrame)) ++points;
		if(detectMouth(validFrame)) ++points;
		if(detectNose(validFrame)) ++points;
		if(detectEyeSmall(validFrame)) ++points;
		if(detectEyeBig(validFrame)) ++points;
		if(detectProfFace(validFrame)) ++points;
	}

	if(points>=3)
		returnV=true;

	cout << endl << "Punteggio:" << points;
	cout << endl;
	return(returnV);
}


float ValidateDetection::bbOverlap(const CvRect& box1,const CvRect& box2){
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


bool ValidateDetection::checkFaceTrak(Mat& img,CvRect& pbox)
{
	vector<Rect> faces;
	bool returnV=false;
	int points=0;
	CvRect validationArea;
	CvRect findArea;
	float bestOverlap=0;
	float currentOverlap;
	//Mat validFrame;



	cascadeAlt2.detectMultiScale(img, faces);
	for( int i = 0; i < faces.size(); i++ ){
			currentOverlap=bbOverlap(faces[i],pbox);
			if(currentOverlap>bestOverlap){
				bestOverlap=currentOverlap;
				validationArea=faces[i];
			}
	}

	if(!(faces.size()>0)){
			cascade.detectMultiScale(img, faces);
			for( int i = 0; i < faces.size(); i++ ){
				currentOverlap=bbOverlap(faces[i],pbox);
				if(currentOverlap>bestOverlap){
					bestOverlap=currentOverlap;
					validationArea=faces[i];
				}
			}
			if(!(faces.size()>0)){
				cascade_profFace.detectMultiScale(img, faces);
				for( int i = 0; i < faces.size(); i++ ){
							currentOverlap=bbOverlap(faces[i],pbox);
							if(currentOverlap>bestOverlap){
								bestOverlap=currentOverlap;
								validationArea=faces[i];
							}
				}
				if((faces.size()>0)){
					rectangle(img,pbox,cvScalar(0),5);
					pbox.x=validationArea.x;
					pbox.y=validationArea.y;
					pbox.width=validationArea.width;
					pbox.height=validationArea.height;
					returnV=true;
					//cout << "profile";
					//cout<< endl;
					//waitKey(0);
				}
			} else{
				rectangle(img,pbox,cvScalar(0),5);
				pbox.x =validationArea.x;
				pbox.y=validationArea.y;
				pbox.width=validationArea.width;
				pbox.height=validationArea.height;
				returnV=true;
				//cout << "face 2";
				//cout<< endl;
				//waitKey(0);
			}
	}else{
		rectangle(img,pbox,cvScalar(0),5);
		pbox.x=validationArea.x;
		pbox.y=validationArea.y;
		pbox.width=validationArea.width;
		pbox.height=validationArea.height;
		returnV=true;

	}


	/*
	if(faces.size()>0){
		findArea=faces[0];
		validationArea.x=findArea.x+(findArea.width/8) ;
		validationArea.y=findArea.y+(findArea.height/8);
		validationArea.width=findArea.width -(findArea.width/4);
		validationArea.height=findArea.height -(findArea.height/4);
		Mat validFrame(img(validationArea));
		rectangle(img,validationArea,cvScalar(50,100,50),1);
		points++;
		if(detectEyes(validFrame)) ++points;
		if(detectGlass(validFrame)) ++points;
		if(detectMouth(validFrame)) ++points;
		if(detectNose(validFrame)) ++points;
		if(detectEyeSmall(validFrame)) ++points;
		if(detectEyeBig(validFrame)) ++points;
		if(detectProfFace(validFrame)) ++points;
	}

	if(points>=3)
		returnV=true;*/

	if(returnV){
		rectangle(img,pbox,cvScalar(250),3);
		rectangle(img,validationArea,cvScalar(100),2);
		imshow("realign face",img);
	}
	return(returnV);
}



ValidateDetection::~ValidateDetection() {
	// TODO Auto-generated destructor stub
}
