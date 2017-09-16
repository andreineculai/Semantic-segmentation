#include <fstream>
#include <string>
#include "../Utils/utils.h"
#include "../Utils/gSlicUtils.h"
#include "../Utils/pointcloud.h"
#include "../gSLICr/NVTimer.h"

#define N 53
#define NCL 11
#define NFeatures 8

using namespace std;
using namespace cv;


float B, f, c_u, c_v;
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
double GP[N][4];
double totalSP = 0;

string  dirDisparity, dirLeft, dirRight, dirGP, dirLabels, groundPlaneEqFileName, baseFilePath, modelFileName, dataset;
PointCloud globalPC;
void setConstants() {
	//B = 0.15;//Virtual
	//f = 346.41;
	//c_u = 400;
	//c_v = 200;

	//B = 0.149515;//Boxes
	//f = 702.012;
	//c_u = 340.433;
	//c_v = 251.25;

	f = 645.2; //Kitti
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

	modelFileName = "modelRealPrezentare.binary";
}

int main() {
	StopWatchInterface *my_timer;
	sdkCreateTimer(&my_timer);
	sdkResetTimer(&my_timer);
	sdkStartTimer(&my_timer);
	setConstants();
	Mat X_Train = Mat(0, NFeatures, CV_64F);//training data for each superpixel
	vector<int> totalLabels;//labels for each superpixel
	ifstream groundPlaneEqFile(baseFilePath + groundPlaneEqFileName);
	Matrix pose = Matrix::eye(4);
	float inf = std::numeric_limits<float>::infinity();
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
	for (int frame = 0; frame < N; frame++) {
		Mat frameLeft = imread(dirLeft + "left (" + to_string(frame + 1) + ").png", IMREAD_COLOR);
		Mat frameDisp = imread(dirDisparity + "disp (" + to_string(frame + 1) + ").png", IMREAD_GRAYSCALE);
		Mat groundTruthImg;
		if (dataset == "Real") {
			groundTruthImg = imread(dirLabels + "gt (" + to_string(frame + 1) + ").png", IMREAD_COLOR);
		}
		else {
			groundTruthImg = imread(dirLabels + "gt (" + to_string(frame + 1) + ").png", IMREAD_GRAYSCALE);
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

		if (dataset == "Virtual1"){
			Mat frameRight = imread(dirRight + "right (" + to_string(frame + 1) + ").png", IMREAD_COLOR);
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
			for (int h = 0; h < heightImg; ++h) {
				for (int w = 0; w < widthImg; ++w) {
					Vec3f coords = pcloud.at<Vec3f>(h, w);
					if (previousPoints[h][w].size()) {
						//There are points that reproject at these coordonates
						list<PointData>::iterator it = previousPoints[h][w].begin();
						if (coords[2] != inf && coords[2] != -inf) {
							int maxTL = -1;
							PointData maxTLPoint;
							for (; it != previousPoints[h][w].end(); ++it) {
								double dist = it->distanceToPoint(coords[0], coords[1], coords[2]);
								if (dist < 0.5) {
									if (it->trackLength > maxTL) {
										maxTL = it->trackLength;
										maxTLPoint = *it;
									}
								}
							}
							if (maxTL == -1) {
								//Current point doesn't fuse with any previous point. Create a new point
								//in the global PC
								notFused++;
								Vec3b colorInfo = frameLeft.at<Vec3b>(h, w);
								globalPC.points.push_back(
									PointData(coords[0], coords[1], coords[2], colorInfo[2], colorInfo[1], colorInfo[0]));
							}
							else {
								//Current point fused fused with one or more points. Use the current points coords and color
								//but aknowledge the fusion by using the highest track length found between the points that fused
								fusePerfect++;
								Vec3b colorInfo = frameLeft.at<Vec3b>(h, w);
								PointData newPoint = PointData(coords[0], coords[1], coords[2], colorInfo[2], colorInfo[1], colorInfo[0], maxTL + 1);
								globalPC.points.push_back(newPoint);

							}
						}
						else {
							//The current point is invalid. Will choose the point
							//with the longest trackLength to use for this frame
							onlyPast++;
							int maxTL = it->trackLength;
							PointData maxTLPoint = *it;
							++it;
							for (; it != previousPoints[h][w].end(); ++it) {
								if (it->trackLength > maxTL) {
									maxTL = it->trackLength;
									maxTLPoint = *it;
								}
							}
							maxTLPoint.trackLength++;
							globalPC.points.push_back(maxTLPoint);
							coords[0] = maxTLPoint.x;
							coords[1] = maxTLPoint.y;
							coords[2] = maxTLPoint.z;
							pcloud.at<Vec3f>(h, w) = coords;

						}
					}
					else
					{
						//There are no points that reproject here. We will add the point from 
						//the current frame to the global pointcloud
						if (coords[2] != inf && coords[2] != -inf) {
							onlyCurrent++;
							Vec3b colorInfo = frameLeft.at<Vec3b>(h, w);
							globalPC.points.push_back(
								PointData(coords[0], coords[1], coords[2], colorInfo[2], colorInfo[1], colorInfo[0]));
						}
					}
				}
			}

			for (int i = 0; i < heightImg; ++i)
				delete[] previousPoints[i];
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
		getFeatures(frameLeft, labelImg, numberOfSuperPixels, pcloud, GP[frame],
			nalpha, heightAbGr, localPl, neighborPl, cameraPath, meanL, meanA, meanB, isValid);
		if (frame == 0)
		{
			colorMaps(frameLeft, labelImg, numberOfSuperPixels, nalpha, heightAbGr, localPl, neighborPl, cameraPath, meanL, meanA, meanB, frameDisp);
		}
		int* labels = new int[numberOfSuperPixels]{};
		if (dataset != "Virtual" && dataset != "Virtual1") {
			groundTruthImg = rgb2ind(groundTruthImg, MAP);//comment for virtual
		}
		int **count = new int*[numberOfSuperPixels];
		for (int i = 0; i < numberOfSuperPixels; ++i) {
			count[i] = new int[NCL]{};
		}
		for (int h = 0; h < heightImg; ++h) {
			for (int w = 0; w < widthImg; ++w) {
				count[labelImg.at<int>(h, w)][groundTruthImg.at<uchar>(h, w)]++;
			}
		}
		//process
		for (int i = 0; i < numberOfSuperPixels; ++i) {
			int max = count[i][0];
			int maxIndex = 0;
			for (int j = 1; j < NCL; ++j) {
				if (count[i][j] > max) {
					max = count[i][j];
					maxIndex = j;
				}
			}
			labels[i] = maxIndex;

		}
		if (dataset == "Real") {
			for (int i = 0; i < numberOfSuperPixels; ++i) {
				{
					if (labels[i] == 0) {//real && boxes
						isValid[i] = 0;
					}
				}
			}
		}
		//free memory
		for (int i = 0; i < numberOfSuperPixels; ++i)
			delete[] count[i];
		delete[] count;

		for (int i = 0; i < numberOfSuperPixels; ++i) {
			if (isValid[i]){
				double data[NFeatures] = { nalpha[i], heightAbGr[i], localPl[i], neighborPl[i],
					cameraPath[i], meanL[i], meanA[i], meanB[i] };
				Mat currentSuperpixel = Mat(1, NFeatures, CV_64F, &data);
				X_Train.push_back(currentSuperpixel);
				totalLabels.push_back(labels[i] + 1);//classRF requires classes to start from 1
			}
		}
		delete[] nalpha;
		delete[] heightAbGr;
		delete[] localPl;
		delete[] neighborPl;
		delete[] cameraPath;
		delete[] isValid;
		delete[] labels;
		cout << "Frame " << frame + 1 << endl;
	}
	double zeros[NFeatures] = { 0 };
	Mat currentSuperpixel = Mat(1, NFeatures, CV_64F, &zeros);
	X_Train.push_back(currentSuperpixel);
	totalLabels.push_back(1);//classRF requires classes to start from 1

	createClassificationModel(X_Train, totalLabels, baseFilePath + modelFileName, NCL, NFeatures);

	sdkStopTimer(&my_timer);
	cout << "Processed in:[" << sdkGetTimerValue(&my_timer) << "]ms" << endl << flush;
	return 0;
}