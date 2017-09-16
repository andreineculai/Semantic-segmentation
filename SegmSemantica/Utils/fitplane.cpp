#include "utils.h"


double* fitplane(Mat superpixelPoints){
	int numberOfPoints = superpixelPoints.rows;
	// find least squares plane
	CvMat *res = cvCreateMat(3, 1, CV_32FC1); // a, b, c for plane ax + by + c = 0;
	// matA * res = matB;
	CvMat *matA = cvCreateMat(numberOfPoints, 3, CV_32FC1);
	CvMat *matB = cvCreateMat(numberOfPoints, 1, CV_32FC1);

	int idx = 0;
	for (int point = 0; point < numberOfPoints; ++point)
	{
		cvmSet(matA, idx, 0, (double)superpixelPoints.at<float>(point, 0));
		cvmSet(matA, idx, 1, (double)superpixelPoints.at<float>(point, 1));
		cvmSet(matA, idx, 2, 1);

		cvmSet(matB, idx, 0, (double)superpixelPoints.at<float>(point, 2));
		++idx;
	}

	// solve the ecuation matA * res = matB;
	cvSolve(matA, matB, res, CV_SVD);

	// ax + by + c = z
	double a, b, c;
	a = cvmGet(res, 0, 0);
	b = cvmGet(res, 1, 0);
	c = cvmGet(res, 2, 0);
	double max = abs(a);
	max = abs(b) > max ? abs(b) : max;
	max = abs(c) > max ? abs(c) : max;
	double* fittedPlane = new double[4] { a/max, b/max, -1.0/max, c/max };
	//deallocate Mat
	cvReleaseMat(&matA);
	cvReleaseMat(&matB);
	cvReleaseMat(&res);
	return fittedPlane;
}