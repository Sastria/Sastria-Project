/*
 * myClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 *      review: Alessandro Romanino
 */

#include "myClassifier.h"
#define MINWINLBPSIZE 4

void myClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];
  structSize = (int)file["num_features"];
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
  best_calculated_thr_nn=0.5;
  pEx.clear();
  nEx.clear();
  pFx.clear();
  nFx.clear();
  for (int i=0;i<512;++i){
	  pFx.push_back(0);
	  nFx.push_back(0);
  }
}


void myClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  int WinLBPW,WinLBPH;

  for (int i=0;i<totalFeatures;i++){
      x1f = (((float)rng));
      y1f = (((float)rng));
      x2f = (((float)rng));
      y2f = (((float)rng));
      for (int s=0;s<((int)scales.size());s++){
          WinLBPW = floor(x1f * (scales[s].width )) > MINWINLBPSIZE ? floor(x1f * (scales[s].width )) : MINWINLBPSIZE;
          WinLBPH = floor(y1f * (scales[s].height)) > MINWINLBPSIZE ? floor(y1f * (scales[s].height)) : MINWINLBPSIZE;
          x1 = floor(x2f * (scales[s].width - WinLBPW));
          y1 = floor(y2f * (scales[s].height - WinLBPH));
          features[s][i] = Feature(x1, y1, WinLBPW, WinLBPH);
      }
  }

 // cvWaitKey(0);
  //Thresholds
  thrN = 0.5*nstructs;

  cout << endl << "Init posteriors!"<< endl;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow((double) 4,(double)structSize), 0));
      pCounter.push_back(vector<int>(pow((double)4,(double)structSize), 0));
      nCounter.push_back(vector<int>(pow((double)4,(double)structSize), 0));
  }

  cout << endl << "End init posteriors!"<< endl;
}



/*
void myClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  double distance=(1.0/(2*(structSize+1)));
  vector<float> distances;
  float currentDistance;
  currentDistance=distance;


  for(int j=0;j<structSize+1;++j){
	  distances.push_back(currentDistance);
	  currentDistance+=distance;
  }

  for (int i=0;i<totalFeatures;i++){
      for (int s=0;s<((int)scales.size());s++){
          x1 = distances.at(i % structSize) * scales[s].width ;
          y1 = distances.at(i % structSize) * scales[s].height;
          x2 = ( distances.at((i % structSize)+1) * scales[s].width );
          y2 = (distances.at((i % structSize)+1) * scales[s].height) ;

          if((x1==0) && (y1==0)){
                  	  x1= (x1+1);
                  	  y1= (y1+1);
          }
          features[s][i] = Feature(x1, y1, x2, y2);
      }
   }

  for(int i=0;i<totalFeatures;++i){
	  cout << i << ":" <<(float) features[0][i].x1 << "-" << (float)features[0][i].y1 << "-" <<(float) features[0][i].x2 << "-" << (float)features[0][i].y2 << " ### ";
	  cout <<(float) features[1][i].x1 << "-" << (float)features[1][i].y1 << "-" <<(float) features[1][i].x2 << "-" << (float)features[1][i].y2 << endl;
  }


  //cvWaitKey(0);
  //Thresholds
  thrN = 0.5*nstructs;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}
*/

void myClassifier::getFeatures(const cv::Mat& image,IntegralImage& integralFrame,const int& scale_idx, vector<int>& fern){
  int leaf;
  for (int t=0;t<nstructs;t++){
      leaf=0;
      //imshow("Feature Extraction",integralFrame.getIntegralImageMat());
      for (int f=0; f<structSize; f++){
    	 // cout << "features[scale_idx][t*nstructs+f](image): " << features[scale_idx][t*nstructs+f](image) << endl;
          leaf = (leaf << 2) + features[scale_idx][t*structSize+f](image,integralFrame);
          //cout << t*structSize << " + " << f << " = " << t*structSize+f << ":" <<(float) features[scale_idx][t*nstructs+f].x1 << "-" << (float)features[scale_idx][t*nstructs+f].y1 << "-" <<(float) features[scale_idx][t*nstructs+f].x2 << "-" << (float)features[scale_idx][t*nstructs+f].y2 << endl;

         // cout << "leaf: " << leaf << endl;
      }
      fern[t]=leaf;
    //  cout << endl << "ScaleID:" << scale_idx << " - " << t << ": featureValue:" << leaf;
      //cvWaitKey(0);
  }
}


float myClassifier::measure_forest(vector<int> fern) {
  float votes = 0;

//  cout << endl << "Posterior Votes ";
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];
    //  cout << posteriors[i][fern[i]] << "-" ;
  }
  //cout << endl;
 //cvWaitKey(0);
  return votes;
}

void myClassifier::forcedUpdate(const vector<int>& fern) {
  int idx;
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      pCounter[i][idx] = 1;
      nCounter[i][idx] = 0;
      posteriors[i][idx] = 1;
  }
}


void myClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}


void myClassifier::forcedTrainFern(vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;

 //random_shuffle(ferns.begin(),ferns.end());

 // random_shuffle(ferns.begin(),ferns.end());
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){                           //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                forcedUpdate(ferns[i].first);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
            	  forcedUpdate(ferns[i].first);                 //             update(x,0,1);
          }

         // for(int i=0; i< posteriors.size();++i)
        	//  cout << endl <<
      }

/*
           cout << endl << "Posterior pcounter ncounter";
           for(int i=0; i< posteriors[0].size();++i){
         	  cout << endl << i << ": " << posteriors[0][i] << " / " << pCounter[0][i] << " / " << nCounter[0][i];
           }
           cout << endl;
           cvWaitKey(0);
           */

  //}
}


void myClassifier::trainFern(vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;

 //random_shuffle(ferns.begin(),ferns.end());

 // random_shuffle(ferns.begin(),ferns.end());
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){                           //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }

         // for(int i=0; i< posteriors.size();++i)
        	//  cout << endl <<
      }

/*
           cout << endl << "Posterior pcounter ncounter";
           for(int i=0; i< posteriors[0].size();++i){
         	  cout << endl << i << ": " << posteriors[0][i] << " / " << pCounter[0][i] << " / " << nCounter[0][i];
           }
           cout << endl;
           cvWaitKey(0);
           */

  //}
}

void myClassifier::trainNNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;

  if(pEx.size()>20)
	  pEx.erase(pEx.begin(),pEx.begin()+1);

  if(nEx.size()>40)
  	  nEx.erase(nEx.begin(),nEx.begin()+1);

  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){                          //  For each example
      NNConf(nn_examples[i],isin,conf,dummy);                      //  Measure Relative similarity
      if (i==0 && conf<=thr_nn){                                 //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          cout << endl << "conf:"<< conf << " thrnn:" << thr_nn;
    	  pEx.push_back(nn_examples[i]);
         // cvWaitKey(0);
      } else if(i>0 && conf>0.5)          //0.5                            //  if y(i) == 0 && conf1 > 0.5
    	  nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                                 //  end
  acum++;
  printf("%d. \nTrained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end



//  Measure Relative similarity
void myClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
  isin=vector<int>(3,-1);
  //IF positive examples in the model are not defined THEN everything is negative so the probability
  // to classify incorrectly a negative example as positive example is 0
  if (pEx.empty()){
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
     // cout << endl << "Stop PEX VOID" << endl;
     // cvWaitKey(0);
     return;
  }

  //IF negative examples in the model are not defined THEN everything is positive so the probability
  // to classify incorrectly a negative example as positive example is 1
  if (nEx.empty()){ //if isempty(tld.nex) %
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      //cout << endl << "Stop NEX VOID" << endl;
      //cvWaitKey(0);
      return;
  }

  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);
  float nccN, maxN=0;
  bool anyN=false;

/*
if(pEx.size()>=2){

  matchTemplate(pEx[0],pEx[0],ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
  nccP=(((float*)ncc.data)[0]+1)*0.5;
  cout << endl << "100/100: ncc=" << (((float*)ncc.data)[0])<< "  nccP=" << nccP;

  matchTemplate(pEx[0],pEx[1],ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
  nccP=(((float*)ncc.data)[0]+1)*0.5;
  cout << endl << "100/200: ncc=" << (((float*)ncc.data)[0])<< "  nccP=" << nccP;

  matchTemplate(pEx[1],pEx[0],ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
  nccP=(((float*)ncc.data)[0]+1)*0.5;
  cout << endl << "200/100: ncc=" << (((float*)ncc.data)[0])<< "  nccP=" << nccP;

  cvWaitKey(0);
}*/

  //imshow("example",example);
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5; //rescale interval -1:1 to 0:1

      /*
      cout << endl << "ncc:" << ncc << " nccP:" << nccP;
      imshow("pex",pEx[i]);
      cvWaitKey(0);
      */
      if (nccP>ncc_thesame){
        anyP=true;
       // cout << endl << "anyP=true; " << nccP;
       // cvWaitKey(0);
      }
      //cout << endl << i << ": pex.size=" <<pEx.<<" ncc=" << (((float*)ncc.data)[0])<< "  nccP=" << nccP;
      if(nccP > maxP){
          maxP=nccP;
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  //cvWaitKey(0);

  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5; //rescale interval -1:1 to 0:1
      //cout << endl << "ncc:" << ncc << " nccN:" << nccN << " nccN.data:" <<(((float*)ncc.data)[0]);
      if (nccN>ncc_thesame){
        anyN=true;
        //cout << endl << "anyN=true; " << nccN;
        //cvWaitKey(0);
      }
      if(nccN > maxN)
        maxN=nccN;
  }

  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them

  //cout << endl << "MAXN: " << maxN << "MaxP: " << maxP;
  //cvWaitKey(0);
  //Measure Relative Similarity
  float dN=1-maxN; //MaxN probability to be negative. dN probability to have error and classify patch incorrectly as positive
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP); // error confidence to classify negative patch as positive(DN)
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}

void myClassifier::evaluateThV(const vector<cv::Mat>& nExT){

  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      // if the probability to evaluate a negative patch as positive is bigger than the current
      // threshold value than update it
      if (conf>thr_nn)
        thr_nn=conf;

      if(conf>best_calculated_thr_nn)
    	  best_calculated_thr_nn=conf;
  }
  if (thr_nn>thr_nn_valid)
    thr_nn_valid = thr_nn;
}


void myClassifier::show(){

int rows=pEx.size()>30 ? 30:pEx.size();
int cols=1+(int)pEx.size()/30;
//cols=cols>1?cols:1;

if(pEx.size()>0){
	Mat examples(rows*pEx[0].rows,cols*pEx[0].cols,CV_8U);
	double minval;
	Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
	for (int i=0;i<pEx.size();i++){
		minMaxLoc(pEx[i],&minval);
		pEx[i].copyTo(ex);
		ex = ex-minval;
		cout<< endl << "rows:" << (i%rows) << "rowsE:" << (i%rows)+1;
		cout << endl;
		Mat tmp1 = examples.rowRange(Range((i%rows)*pEx[i].rows,((i%rows)+1)*pEx[i].rows));
		Mat tmp= tmp1.colRange(Range((i/rows)*pEx[i].cols,((i/rows)+1)*pEx[i].cols));
		ex.convertTo(tmp,CV_8U);
	}
	imshow("Examples",examples);
}

  //negative
if(nEx.size()>0){
  Mat examplesNN((int)nEx.size()*nEx[0].rows,nEx[0].cols,CV_8U);
  double minvalNN;
  Mat exNN(nEx[0].rows,nEx[0].cols,nEx[0].type());
  for (int i=0;i<nEx.size();i++){
    minMaxLoc(nEx[i],&minvalNN);
    nEx[i].copyTo(exNN);
    exNN = exNN-minvalNN;
    Mat tmpNN = examplesNN.rowRange(Range(i*nEx[i].rows,(i+1)*nEx[i].rows));
    exNN.convertTo(tmpNN,CV_8U);
  }
  imshow("ExamplesNN",examplesNN);
}


}
