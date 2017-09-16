#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <limits>
#include <iostream>
#include <cmath>
#include "SLIC.h"
#include "../LIBVISO2/viso_stereo.h"

using namespace std;
using namespace cv;

double dist2Plane(Mat superpixelPoints, double plane[4]);

double dist2GP(Mat superpixelPoints, double plane[4]);

Mat getFeatures(Mat image, Mat labels, int superpixelsNumber, Mat pointCloud, double groundPlane[4],
	double nalpha[], double heightAbGr[], double localPl[], double neighborPl[], double cameraPath[],
	double meanL[], double meanA[], double meanB[], int isValid[]);

Mat rgb2ind(Mat image, int Map[][3]);

Mat rgba2ind(Mat image, int Map[][3]);

Mat getAdjacencyMatrix(const int* klabels, int width, int height, int K);

double* fitplane(Mat superpixelPoints);

vector<string> getFilesInDir(string path);

void colorMaps(Mat image, Mat labels, int numberOfSuperpixels, double nalpha[], double heightAbGr[], double localPl[],
	double neighborPl[], double cameraPath[], double meanL[], double meanA[], double meanB[], Mat disp);

void createPlyFileFromPointcloud(Mat image, Mat pointcloud, String filepath);

int groupClassesRealDataset(int cl);

void resetVotes(int *votes, int ncl);

void setVotes(int *votes, int* global, int ncl);

void createClassificationModel(Mat X_Train, vector<int> totalLabels, string fileName, int ncl, int numberOfFeatures);

void computePointCloud(Mat frameDisp, Mat &pcloud, float B, float f, float c_u, float c_v);