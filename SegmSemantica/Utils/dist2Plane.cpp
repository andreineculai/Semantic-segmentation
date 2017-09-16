#include "utils.h"

double dist2Plane(Mat superpixelPoints, double plane[4]) {
	double distanceSum = 0;
	double A = plane[0];
	double B = plane[1];
	double C = plane[2];
	double D = plane[3];

	int numberOfPoints = superpixelPoints.rows;
	for (int pointNumber = 0; pointNumber < numberOfPoints; ++pointNumber) {
		float x = superpixelPoints.at<float>(pointNumber, 0);
		float y = superpixelPoints.at<float>(pointNumber, 1);
		float z = superpixelPoints.at<float>(pointNumber, 2);
		double currentDistance = abs(A * x + B * y + C * z + D) / sqrt(A * A + B * B + C * C);
		distanceSum += currentDistance * currentDistance;
	}
	return distanceSum;
}

double dist2GP(Mat superpixelPoints, double plane[4]) {
	double distanceSum = 0;
	double A = plane[0];
	double B = plane[1];
	double C = plane[2];
	double D = plane[3];

	int numberOfPoints = superpixelPoints.rows;
	for (int pointNumber = 0; pointNumber < numberOfPoints; ++pointNumber) {
		float x = superpixelPoints.at<float>(pointNumber, 0);
		float y = superpixelPoints.at<float>(pointNumber, 1);
		float z = superpixelPoints.at<float>(pointNumber, 2);
		double currentDistance = abs(A * x + B * y + C * z + D) / sqrt(A * A + B * B + C * C);
		distanceSum += currentDistance;
	}
	return distanceSum / numberOfPoints;
}
