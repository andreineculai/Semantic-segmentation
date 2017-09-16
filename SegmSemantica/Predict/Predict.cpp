#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../RandomForest/rf.h"
#include "../Utils/utils.h"
#include "../Utils/gSlicUtils.h"
#include "../GCO/GCoptimization.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "../gSLICr/NVTimer.h"
#include "../Utils/pointcloud.h"


using namespace cv;
using namespace std;

#define N 377
#define scaleFactor 10000
#define inf 10000000

int MAP[11][3] = {
	{ 0, 0, 0 },
	{ 0, 0, 192 },
	{ 64, 0, 128 },
	{ 128, 128, 128 },
	{ 0, 64, 64 },
	{ 128, 128, 0 },
	{ 64, 64, 0 },
	{ 192, 0, 64 },
	{ 64, 192, 0 },
	{ 128, 0, 0 },
	{ 128, 64, 128 }
};

//int MAP[4][3] = {
//	{ 128, 128, 128 },
//	{ 255, 69, 0 },
//	{ 128, 0, 128 },
//	{ 0, 0, 0 }
//};
//
int MAPVirtualSet[37][3] = {
	{ 64, 0, 192 },
	{ 229, 145, 115 },
	{ 179, 146, 134 },
	{ 115, 63, 29 },
	{ 204, 109, 0 },
	{ 242, 214, 182 },
	{ 242, 194, 0 },
	{ 76, 69, 38 },
	{ 127, 121, 32 },
	{ 153, 191, 0 },
	{ 222, 242, 182 },
	{ 117, 128, 96 },
	{ 32, 242, 0 },
	{ 15, 115, 0 },
	{ 64, 255, 166 },
	{ 16, 64, 41 },
	{ 83, 166, 149 },
	{ 64, 242, 255 },
	{ 0, 71, 89 },
	{ 102, 170, 204 },
	{ 255, 68, 0 },
	{ 38, 45, 51 },
	{ 128, 179, 255 },
	{ 64, 115, 255 },
	{ 34, 0, 255 },
	{ 14, 0, 102 },
	{ 87, 77, 153 },
	{ 124, 48, 191 },
	{ 230, 182, 242 },
	{ 48, 0, 51 },
	{ 255, 0, 238 },
	{ 89, 45, 80 },
	{ 191, 0, 128 },
	{ 242, 61, 109 },
	{ 178, 0, 24 },
	{ 89, 22, 31 },
	{ 204, 153, 160 }
};

float B, f, c_u, c_v;

double GP[N][4];
bool withFusion, withGCO;
PointCloud globalPC;
string  dirDisparity, dirLeft, dirRight, dirGP, dirLabels, groundPlaneEqFileName, baseFilePath, modelFileName, dataset, saveDirectory;
void setConstants() {
	//B = 0.15;
	//f = 346.41;
	//c_u = 400;
	//c_v = 200;
	
	//B = 0.149515;
	//f = 702.012;
	//c_u = 340.433;
	//c_v = 251.25;
	f = 645.2;
	B = 0.571;
	c_u = 635.9;
	c_v = 194.1;
	baseFilePath = "..\\..\\";
	dataset = "Real";
	dirDisparity = baseFilePath + "TestData\\" + dataset + "\\disparity\\";
	dirLeft = baseFilePath + "TestData\\" + dataset + "\\left\\";
	dirRight = baseFilePath + "TestData\\" + dataset + "\\right\\";
	dirGP = baseFilePath + "TestData\\" + dataset + "\\gp\\";
	dirLabels = baseFilePath + "TestData\\" + dataset + "\\gt\\";
	groundPlaneEqFileName = "gp" + dataset + ".txt";
	withFusion = false;
	withGCO = true;
	modelFileName = "modelRealPrezentare.binary";
	saveDirectory = "ResultsRealPrezentare/";

}
int main(){
	setConstants();
	ifstream groundPlaneEqFile(baseFilePath + groundPlaneEqFileName);
	FILE* model;
	fopen_s(&model, (baseFilePath + modelFileName).c_str(), "rb");
	Matrix pose = Matrix::eye(4);
	VisualOdometryStereo::parameters param;
	param.calib.f = f; // focal length in pixels
	param.calib.cu = c_u; // principal point (u-coordinate) in pixels
	param.calib.cv = c_v; // principal point (v-coordinate) in pixels
	param.base = B; // baseline in meters
	VisualOdometryStereo viso(param);
	for (int i = 0; i < N; ++i) {
		groundPlaneEqFile >> GP[i][0];
		groundPlaneEqFile >> GP[i][1];
		groundPlaneEqFile >> GP[i][2];
		groundPlaneEqFile >> GP[i][3];
	}
	groundPlaneEqFile.close();

	cout << "Begin reading model" << endl;

	int ncl, numberOfFeatures, maxcat, ntree, nrnodes;
	fread(&ncl, 1, sizeof(int), model);
	PointData::ncl() = ncl;
	fread(&numberOfFeatures, 1, sizeof(int), model);
	fread(&maxcat, 1, sizeof(int), model);
	fread(&nrnodes, 1, sizeof(int), model);
	fread(&ntree, 1, sizeof(int), model);

	int* cat = new int[numberOfFeatures];
	double* classwt = new double[ncl];
	double* cut = new double[ncl];
	int* ndbigtree = new int[nrnodes * ntree];
	int* nodestatus = new int[nrnodes * ntree];
	int* bestvar = new int[nrnodes * ntree];
	int* treemap = new int[nrnodes * ntree * 2];
	int* nodepred = new int[nrnodes * ntree];
	double* xbestsplit = new double[nrnodes * ntree];

	fread(xbestsplit, 1, nrnodes * ntree*sizeof(*xbestsplit), model);
	fread(classwt, 1, ncl * sizeof(*classwt), model);
	fread(cut, 1, ncl * sizeof(*cut), model);
	fread(treemap, 1, nrnodes * ntree * 2 * sizeof(*treemap), model);
	fread(nodestatus, 1, nrnodes * ntree * sizeof(*nodestatus), model);
	fread(cat, 1, numberOfFeatures * sizeof(*cat), model);
	fread(nodepred, 1, nrnodes * ntree * sizeof(*nodepred), model);
	fread(bestvar, 1, nrnodes * ntree * sizeof(*bestvar), model);
	fread(ndbigtree, 1, nrnodes * ntree * sizeof(*ndbigtree), model);
	fclose(model);
	vector<int> frames;
	if (dataset == "Real") {
		frames = { 10, 20, 30, 40, 50, 54, 55, 56, 57, 58 };
	}
	else if(dataset == "Virtual1"){
		frames = { 1, 50, 100, 150, 200, 260, 280, 300, 320, 340, 360, 370 };
	}
	
	StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
	sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);

	Votes*** votesTable = NULL;

	for (int frame : frames)
	{
		Mat X_Predict = Mat(0, numberOfFeatures, CV_64F);//predict data for each superpixel
		Mat frameLeft = imread(dirLeft + "left (" + to_string(frame) + ").png", IMREAD_COLOR);
		Mat frameDisp = imread(dirDisparity + "disp (" + to_string(frame) + ").png", IMREAD_GRAYSCALE);
		Mat groundTruthImg;
		if (dataset == "Real") {
			groundTruthImg = imread(dirLabels + "gt (" + to_string(frame) + ").png", IMREAD_COLOR);
		}
		else {
			groundTruthImg = imread(dirLabels + "gt (" + to_string(frame) + ").png", IMREAD_GRAYSCALE);
		}
		int heightImg = frameDisp.rows;
		int widthImg = frameDisp.cols;
		if (dataset == "Real")
		{
			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					uchar a = frameDisp.at<uchar>(h, w);
					if (frameDisp.at<uchar>(h, w) < 5)
						frameDisp.at<uchar>(h, w) = 0;
				}
			}
		}
		Mat labelImg = Mat(heightImg, widthImg, CV_32S);
		getSegmentationMask(frameLeft, labelImg);
		double minVal;
		double maxVal;
		minMaxLoc(labelImg, &minVal, &maxVal);
		int numberOfSuperPixels = (int)maxVal + 1;

		Mat pcloud = Mat(heightImg, widthImg, CV_32FC3);
		computePointCloud(frameDisp, pcloud, B, f, c_u, c_v);
		
		double* nalpha = new double[numberOfSuperPixels] {};
		double* heightAbGr = new double[numberOfSuperPixels] {};
		double* localPl = new double[numberOfSuperPixels] {};
		double* neighborPl = new double[numberOfSuperPixels] {};
		double* cameraPath = new double[numberOfSuperPixels] {};
		double* meanL = new double[numberOfSuperPixels] {};
		double* meanA = new double[numberOfSuperPixels] {};
		double* meanB = new double[numberOfSuperPixels] {};
		int* isValid = new int[numberOfSuperPixels] {};

		if (dataset == "Virtual1" && withFusion){
			if (votesTable == NULL) {
				votesTable = new Votes**[heightImg];
				for (int i = 0; i < heightImg; ++i){
					votesTable[i] = new Votes*[widthImg];
					for (int j = 0; j < widthImg; ++j){
						votesTable[i][j] = new Votes();
					}
				}
			}
			Mat frameRight = imread(dirRight + "right (" + to_string(frame) + ").png", IMREAD_COLOR);
			Mat leftGray, rightGray;
			cvtColor(frameLeft, leftGray, CV_RGB2GRAY);
			cvtColor(frameRight, rightGray, CV_RGB2GRAY);

			int32_t dims[] = { widthImg, heightImg, widthImg };

			//pose = pose * Matrix::inv(viso.getMotion());
			//pose = Matrix::inv(viso.getMotion());

			list<PointData> **previousPoints = new list<PointData>*[heightImg];
			for (int i = 0; i < heightImg; ++i) {
				previousPoints[i] = new list<PointData>[widthImg];
			}
			PointData* newPoint;
			int outOfBounds = 0;
			if (viso.process((uint8_t*)leftGray.ptr(), (uint8_t*)rightGray.ptr(), dims)) {
				pose = viso.getMotion();
				globalPC.applyTransformation(pose);
				list<PointData>::iterator it = globalPC.points.begin();
				while (it != globalPC.points.end()) {
					int u = it->getUFromPoint(f, c_u);
					int v = it->getVFromPoint(f, c_v);
					if (u >= 0 && u < widthImg && v >= 0 && v < heightImg) {
						previousPoints[v][u].push_back(*it);
					}
					else {
						outOfBounds++;
					}
					it = globalPC.points.erase(it);
				}
			}
			int onlyCurrent = 0, onlyPast = 0, fusePerfect = 0, notFused = 0;
			list<PointData>::iterator it;
			PointData* maxTLPoint;
			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					Vec3f coords = pcloud.at<Vec3f>(h, w);
					if (previousPoints[h][w].size()) {
						//There are points that reproject at these coordonates
						if (coords[2] != inf && coords[2] != -inf) {
							int maxTL = -1;
							
							for (it = previousPoints[h][w].begin(); it != previousPoints[h][w].end(); ++it) {
								double dist = it->distanceToPoint(coords[0], coords[1], coords[2]);
								if (dist < 0.5) {
									if (it->trackLength > maxTL) {
										maxTL = it->trackLength;
										maxTLPoint = &*it;

									}
								}
							}
							if (maxTL == -1) {
								//Current point doesn't fuse with any previous point. Create a new point
								//in the global PC
								votesTable[h][w]->resetVotes();
								notFused++;
								Vec3b colorInfo = frameLeft.at<Vec3b>(h, w);
								globalPC.points.push_back(PointData(
									coords[0], coords[1], coords[2], colorInfo[2], colorInfo[1], colorInfo[0], 1, true));
							}
							else {
								//Current point fused with one or more points. Use the current points coords and color
								//but aknowledge the fusion by using the highest track length found between the points that fused
								votesTable[h][w]->setVotes(maxTLPoint->previousClassifications, maxTL);
								fusePerfect++;
								Vec3b colorInfo = frameLeft.at<Vec3b>(h, w);
								globalPC.points.push_back(PointData(
									coords[0], coords[1], coords[2], colorInfo[2], colorInfo[1], colorInfo[0],
									maxTL + 1, true, maxTLPoint->previousClassifications));
							}
						}
						else {
							//The current point is invalid. Will choose the point
							//with the longest trackLength to use for this frame
							onlyPast++;
							int maxTL = -1;
							for (it = previousPoints[h][w].begin(); it != previousPoints[h][w].end(); ++it) {
								if (it->trackLength > maxTL) {
									maxTL = it->trackLength;
									maxTLPoint = &*it;
								}
							}
							votesTable[h][w]->setVotes(maxTLPoint->previousClassifications, maxTL);
							maxTLPoint->trackLength++;
							globalPC.points.push_back(PointData(*maxTLPoint));
							coords[0] = maxTLPoint->x;
							coords[1] = maxTLPoint->y;
							coords[2] = maxTLPoint->z;
							pcloud.at<Vec3f>(h, w) = coords;

						}
					}
					else
					{
						//There are no points that reproject here. We will add the point from 
						//the current frame to the global pointcloud
						if (coords[2] != inf && coords[2] != -inf) {
							votesTable[h][w]->resetVotes();
							onlyCurrent++;
							Vec3b colorInfo = frameLeft.at<Vec3b>(h, w);
							globalPC.points.push_back(PointData(
								coords[0], coords[1], coords[2], colorInfo[2], colorInfo[1], colorInfo[0], 1, true));
						}
					}
				}
			}
			
			for (int i = 0; i < heightImg; ++i)  {
				delete[] previousPoints[i];
			}
			delete[] previousPoints;
			cout << frame << endl << "Only current - " << onlyCurrent << endl <<
				"Only prev - " << onlyPast << endl << "Fused - " << fusePerfect << endl
				<< "Not fused - " << notFused << endl << "Out of bounds - " << outOfBounds << endl << endl << endl;
			//cout << endl << "Frame - " << frame << ' ' << globalPC.points.size() << endl;
			//createPlyFileFromPointcloud(frameLeft, pcloud, baseFilePath + "plySCSS" + to_string(frame) + ".ply");
			/*if (frame <= 3)
			createPlyFileFromPointcloud(frameLeft, pcloud, baseFilePath + "ply" + to_string(frame) + ".ply");
			else
			return 0;
			*/
		}
		Mat ad = getFeatures(frameLeft, labelImg, numberOfSuperPixels, pcloud, GP[frame - 1],
			nalpha, heightAbGr, localPl, neighborPl, cameraPath, meanL, meanA, meanB, isValid);
		//colorMaps(frameLeft, labelImg, numberOfSuperPixels, nalpha, heightAbGr, localPl, neighborPl, cameraPath, meanL, meanA, meanB, frameDisp);

		for (int i = 0; i < numberOfSuperPixels; ++i) {
			double data[] = { nalpha[i], heightAbGr[i], localPl[i], neighborPl[i],
				cameraPath[i], meanL[i], meanA[i], meanB[i] };
			Mat currentSuperpixel = Mat(1, numberOfFeatures, CV_64F, &data);
			X_Predict.push_back(currentSuperpixel);
		}
		Mat labFrame;
		cvtColor(frameLeft, labFrame, CV_RGB2Lab);
		float* totalL = new float[numberOfSuperPixels]{};
		float* totalA = new float[numberOfSuperPixels]{};
		float* totalB = new float[numberOfSuperPixels]{};
		int* totalCount = new int[numberOfSuperPixels]{};

		for (int h = 0; h < heightImg; ++h) {
			for (int w = 0; w < widthImg; ++w) {
				int sp = labelImg.at<int>(h, w);
				Vec3b pixelData = labFrame.at<Vec3b>(h, w);
				totalL[sp] += (int)pixelData[0];
				totalA[sp] += (int)pixelData[1];
				totalB[sp] += (int)pixelData[2];
				totalCount[sp]++;
			}
		}
		for (int i = 0; i < numberOfSuperPixels; ++i) {
			totalL[i] /= totalCount[i];
			totalA[i] /= totalCount[i];
			totalB[i] /= totalCount[i];
		}

		Mat neighborEnergy = Mat(numberOfSuperPixels, numberOfSuperPixels, CV_32F);
		
		for (int i = 0; i < numberOfSuperPixels; ++i) {
			for (int j = i+1; j < numberOfSuperPixels; ++j) {
				if (ad.at<int>(i, j) == 1) {
					float L = totalL[i] - totalL[j];
					float A = totalA[i] - totalA[j];
					float B = totalB[i] - totalB[j];
					float norm = sqrt(L*L + A*A + B*B);
					neighborEnergy.at<float>(i, j) = 1.9 *(1.0 / (0.1 * norm + 1.0));
				}
				else {
					neighborEnergy.at<float>(i, j) = 0;
				}
			}
		}


		double *XPred = (double *)X_Predict.data;
		int dimxpred[2] = { X_Predict.cols, X_Predict.rows };//reversed to match transposition
		//(dimx[0] - number of features)
		//(dimx[1] - number of samples)
		int ntest = dimxpred[1];
		int p_size = dimxpred[0];


		double* countts = new double[ncl * ntest]();
		int* nodeclass = nodepred;
		int* jts = new int[ntest]();
		int* jet = new int[ntest]();
		int* treesize = ndbigtree;
		int keepPred = 0;
		int intProximity = 0;
		int nodes = 0;
		int *node = new int[ntest]();
		double *proxMat = new double[1];
		cout << "Running random forest" << endl;
		classForest(&p_size, &ntest, &ncl, &maxcat, &nrnodes, &ntree, XPred, xbestsplit,
			classwt, cut, countts, treemap, nodestatus, cat, nodeclass, jts, jet,
			bestvar, node, treesize, &keepPred, &intProximity, proxMat, &nodes);
		cout << "RF ended" << endl;
	
		
		if (dataset == "Virtual1" && withFusion) {
			for (PointData& point : globalPC.points) {
				int u = point.getUFromPoint(f, c_u);
				int v = point.getVFromPoint(f, c_v);
				int label = labelImg.at<int>(v, u);
				point.addVotes(countts + (ncl * label));
			}
			Votes** totalVotes = new Votes*[numberOfSuperPixels];
			for (int i = 0; i < numberOfSuperPixels; ++i) {
				totalVotes[i] = new Votes();
			}
			int* numberOfVoters = new int[numberOfSuperPixels]{};
			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					int label = labelImg.at<int>(h, w);
					totalVotes[label]->addVotes(votesTable[h][w]);
					numberOfVoters[label]++;

				}
			}
			for (int sp = 0; sp < numberOfSuperPixels; ++sp) {
				if (totalVotes[sp]->numberOfVoteCasts != 0) {
					totalVotes[sp]->addCurrentFrameVotes(countts + (ncl * sp), numberOfVoters[sp]);
					double* meanVotes = totalVotes[sp]->getMean();
					for (int i = 0; i < ncl; ++i) {
						countts[ncl * sp + i] = meanVotes[i];
					}
					delete meanVotes;
				}
			}
			for (int i = 0; i < numberOfSuperPixels; ++i)  {
				delete totalVotes[i];
			}
			delete[] totalVotes;
		}

		//Postprocessing - GCO
		//Local term
		int *data = new int[numberOfSuperPixels*ncl];
		for (int i = 0; i < numberOfSuperPixels; i++){
			for (int label = 0; label < ncl; label++){
				if (countts[ncl * i + label] != 0){
					data[i*ncl + label] = (int)(-scaleFactor*(log(countts[ncl * i + label] / ntree)));
				}
				else{
					data[i*ncl + label] = inf;
				}
			}
		}

		//smoothness term
		int *smooth = new int[ncl*ncl];
		for (int l1 = 0; l1 < ncl; l1++){
			for (int l2 = 0; l2 < ncl; l2++){
				if (l1 == l2) {
					smooth[l1 + l2*ncl] = 0;
				}
				else {
					smooth[l1 + l2*ncl] = 1;
				}
			}
		}
		//StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
		try{
			GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numberOfSuperPixels, ncl);
			gc->setDataCost(data);
			gc->setSmoothCost(smooth);
			for (int i = 0; i < numberOfSuperPixels; ++i) {
				gc->setLabel(i, jts[i] - 1);
			}
			for (int i = 0; i < numberOfSuperPixels; ++i) {
				for (int j = i + 1; j < numberOfSuperPixels; ++j) {
					if (neighborEnergy.at<float>(i, j)){
						gc->setNeighbors(i, j, (int)(neighborEnergy.at<float>(i, j) * scaleFactor));
					}
				}
			}
			if (withGCO) {
				gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
			}
			for (int i = 0; i < numberOfSuperPixels; i++)
				jts[i] = gc->whatLabel(i);
			delete gc;
		}
		catch (GCException e){
			e.Report();
		}
		delete[] smooth;
		delete[] data;
	
		int count = 0;
		int countInvalid = 0;
		Mat original = Mat(heightImg, widthImg, CV_8UC3);
		Mat predicted = Mat(heightImg, widthImg, CV_8UC3);

		if (dataset == "Real") {

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					labelImg.at<int>(h, w) = jts[labelImg.at<int>(h, w)];
					/*if (groundTruthImg.at<Vec3b>(h, w) != Vec3b(0,0,0)) {
						labelImg.at<int>(h, w) = jts[labelImg.at<int>(h, w)];
					}
					else {
						labelImg.at<int>(h, w) = 0;
					}*/
				}
			}
			int *classCountGroundTruth = new int[ncl]{};
			int *classCountPredicted = new int[ncl]{};
			original = groundTruthImg;
			groundTruthImg = rgb2ind(groundTruthImg, MAP);

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					int groundTruth = (int)groundTruthImg.at<uchar>(h, w);
					if (groundTruth != 0){
						classCountGroundTruth[groundTruth]++;
						if (labelImg.at<int>(h, w) == groundTruth){
							count++;
							classCountPredicted[groundTruth]++;
						}
					}
					else {
						countInvalid++;
					}
				}
			}
			string resultsDirectoryPath = baseFilePath + saveDirectory;
			ofstream ofs;
			string accFilepath = resultsDirectoryPath + "Acc.txt";
			if (frame == 1) {
				ofs.open(accFilepath, ofstream::out);
			}
			else {
				ofs.open(accFilepath, ofstream::out | ofstream::app);
			}
			for (int i = 0; i < ncl; ++i){
				if (classCountGroundTruth[i] != 0){
					double classAcc = (double)classCountPredicted[i] / classCountGroundTruth[i];
					ofs << "Clasa " << i << " - " << classAcc * 100 << '%'<<endl;
				}
			}
			ofs.close();

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					Vec3b p;
					p[2] = MAP[labelImg.at<int>(h, w)][0];
					p[1] = MAP[labelImg.at<int>(h, w)][1];
					p[0] = MAP[labelImg.at<int>(h, w)][2];
					predicted.at<Vec3b>(h, w) = p;
				}
			}

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					Vec3b p;
					p[2] = MAP[groundTruthImg.at<uchar>(h, w)][0];
					p[1] = MAP[groundTruthImg.at<uchar>(h, w)][1];
					p[0] = MAP[groundTruthImg.at<uchar>(h, w)][2];
					original.at<Vec3b>(h, w) = p;
				}
			}
		}
		else if (dataset == "Virtual1"){

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					labelImg.at<int>(h, w) = jts[labelImg.at<int>(h, w)];
				}
			}

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					int groundTruth = (int)groundTruthImg.at<uchar>(h, w);
					if (labelImg.at<int>(h, w) == groundTruth){
						count++;
					}
				}
			}

			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					Vec3b p = original.at<Vec3b>(h, w);
					p[2] = MAPVirtualSet[(int)groundTruthImg.at<uchar>(h, w)][0];
					p[1] = MAPVirtualSet[(int)groundTruthImg.at<uchar>(h, w)][1];
					p[0] = MAPVirtualSet[(int)groundTruthImg.at<uchar>(h, w)][2];
					original.at<Vec3b>(h, w) = p;
				}
			}


			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					Vec3b p = predicted.at<Vec3b>(h, w);
					p[2] = MAPVirtualSet[labelImg.at<int>(h, w)][0];
					p[1] = MAPVirtualSet[labelImg.at<int>(h, w)][1];
					p[0] = MAPVirtualSet[labelImg.at<int>(h, w)][2];
					predicted.at<Vec3b>(h, w) = p;
				}
			}

		}		

		double acc = ((double)count / (double)((heightImg*widthImg) - countInvalid)) * 100.0;
		cout << "Acc_" << frame << " = " << acc << endl;
		
		
		
		string resultsDirectoryPath = baseFilePath + saveDirectory;
		imwrite(resultsDirectoryPath + to_string(frame) + "_orig.png", original);
		/*Mat combined;
		addWeighted(frameLeft, 0.15, predicted, 0.85, 0, combined);*/
		imwrite(resultsDirectoryPath + to_string(frame) + "_pred.png", predicted);
		ofstream ofs;
		string accFilepath = resultsDirectoryPath + "Acc.txt";
		if (frame == 1) {
			ofs.open(accFilepath, ofstream::out);
		}
		else {
			ofs.open(accFilepath, ofstream::out | ofstream::app);
		}

		ofs << "Acc_" << frame << " = " << acc << endl;



	}
	sdkStopTimer(&my_timer);
	//cout << frames.size() << " frames - "<< "Processed in:[" << sdkGetTimerValue(&my_timer) << "]ms" << endl << flush;
	return 0;
}