#include "utils.h"
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>


Mat getFeatures(Mat image, Mat labels, int superpixelsNumber, Mat pointCloud, double groundPlane[4],
	double nalpha[], double heightAbGr[], double localPl[], double neighborPl[], double cameraPath[],
	double meanL[], double meanA[], double meanB[], int isValid[]) {
	Mat surfaceNorm = Mat::zeros(3, superpixelsNumber, CV_64F);
	float inf = std::numeric_limits<float>::infinity();
	double GPnormal[3] = { groundPlane[0], groundPlane[1], groundPlane[2] };
	Mat* superpixelPoints = new Mat[superpixelsNumber];
	for (int i = 0; i < superpixelsNumber; ++i) {
		superpixelPoints[i] = Mat(0, 3, CV_32F);
	}
	for (int h = 0; h < labels.rows; ++h) {
		for (int w = 0; w < labels.cols; ++w) {
			int a = labels.at<int>(h, w);
			/*if (pointCloud.at<float>(h*labels.cols + w, 2) != inf && pointCloud.at<float>(h*labels.cols + w, 2) != -inf) {
			superpixelPoints[labels.at<int>(h, w)].push_back(pointCloud.row(h*labels.cols + w));
			}*/
			if (pointCloud.at<Vec3f>(h, w)[2] != inf && pointCloud.at<Vec3f>(h, w)[2] != -inf) {
				float data[] = { pointCloud.at<Vec3f>(h, w)[0],
					pointCloud.at<Vec3f>(h, w)[1],
					pointCloud.at<Vec3f>(h, w)[2] };
				Mat currentPoint = Mat(1, 3, CV_32F, &data);
				superpixelPoints[labels.at<int>(h, w)].push_back(currentPoint);
			}
		}
	}
	for (int i = 0; i < superpixelsNumber; ++i) {
		int validSize = superpixelPoints[i].rows;
		if (validSize > 3) {
			double* fittedPlane = fitplane(superpixelPoints[i]);
			double a = fittedPlane[0];
			double b = fittedPlane[1];
			double c = fittedPlane[2];
			double d = fittedPlane[3];
			delete[] fittedPlane;

			/*double max = abs(a);
			max = abs(b) > max ? abs(b) : max;
			max = abs(c) > max ? abs(c) : max;
			max = abs(d) > max ? abs(d) : max;
			a /= max; b /= max; c /= max; d /= max;*/
			surfaceNorm.at<double>(0, i) = a;
			surfaceNorm.at<double>(1, i) = b;
			surfaceNorm.at<double>(2, i) = c;
			double modGP = sqrt(GPnormal[0] * GPnormal[0] +
				GPnormal[1] * GPnormal[1] +
				GPnormal[2] * GPnormal[2]);
			double modfittedPlane = sqrt(a * a + b * b + c * c);
			double dotProduct = (GPnormal[0] * surfaceNorm.at<double>(0, i) +
				GPnormal[1] * surfaceNorm.at<double>(1, i) +
				GPnormal[2] * surfaceNorm.at<double>(2, i)) / (modGP * modfittedPlane);
			nalpha[i] = acos(dotProduct);
			nalpha[i] = nalpha[i] > M_PI ? nalpha[i] - M_PI : nalpha[i];
			heightAbGr[i] = dist2GP(superpixelPoints[i], groundPlane);
			double plane[4] = { a, b, c, d };
			localPl[i] = dist2Plane(superpixelPoints[i], plane);
			//mean first column

			//distance to camera path
			double sumColumn = 0;
			for (int j = 0; j < validSize; ++j) {
				sumColumn += superpixelPoints[i].at<float>(j, 0);
			}
			cameraPath[i] = abs(sumColumn / validSize);
		}
	}
	delete[] superpixelPoints;
	// neighbor planarity
	Mat ad = getAdjacencyMatrix((int*)labels.data, labels.cols, labels.rows, superpixelsNumber);
	for (int i = 0; i < superpixelsNumber; ++i) {
		const int* currentSuperpixelNeighbors = ad.ptr<int>(i);
		double currentSN[3] = { surfaceNorm.at<double>(0, i),
			surfaceNorm.at<double>(1, i),
			surfaceNorm.at<double>(2, i) };
		if (currentSN[0] * currentSN[0] +
			currentSN[1] * currentSN[1] +
			currentSN[2] * currentSN[2] != 0) {
			isValid[i] = 1;
			vector<int> neighbors;
			for (int j = 0; j < superpixelsNumber; ++j) {
				if (currentSuperpixelNeighbors[j])
					neighbors.push_back(j);
			}
			int numberOfNeighbors = neighbors.size();
			int numberOfValidNeighbors = 0;
			for (int j = 0; j < numberOfNeighbors; ++j) {
				double neighborSN[3] = { surfaceNorm.at<double>(0, neighbors[j]),
					surfaceNorm.at<double>(1, neighbors[j]),
					surfaceNorm.at<double>(2, neighbors[j]) };
				if (neighborSN[0] * neighborSN[0] +
					neighborSN[1] * neighborSN[1] +
					neighborSN[2] * neighborSN[2] != 0) {
					numberOfValidNeighbors++;
					double prod[3];
					prod[0] = currentSN[1] * neighborSN[2] - currentSN[2] * neighborSN[1];
					prod[1] = currentSN[2] * neighborSN[0] - currentSN[0] * neighborSN[2];
					prod[2] = currentSN[0] * neighborSN[1] - currentSN[1] * neighborSN[0];
					neighborPl[i] += sqrt((prod[0] * prod[0] + prod[1] * prod[1] + prod[2] * prod[2]));
				}
			}
			neighborPl[i] = neighborPl[i] ? neighborPl[i] / numberOfValidNeighbors : 0;
		}
	}

	Mat labFrame;
	cvtColor(image, labFrame, CV_RGB2Lab);
	int* totalCount = new int[superpixelsNumber]{};

	for (int h = 0; h < image.rows; ++h) {
		for (int w = 0; w < image.cols; ++w) {
			int sp = labels.at<int>(h, w);
			if (isValid[sp]) {
				Vec3b pixelData = labFrame.at<Vec3b>(h, w);
				meanL[sp] += (int)pixelData[0];
				meanA[sp] += (int)pixelData[1];
				meanB[sp] += (int)pixelData[2];
				totalCount[sp]++;
			}
		}
	}
	for (int i = 0; i < superpixelsNumber; ++i) {
		if (isValid[i]) {
			meanL[i] /= totalCount[i];
			meanA[i] /= totalCount[i];
			meanB[i] /= totalCount[i];
		}
	}

	return ad;
}