#include "../Utils/utils.h"

void computePointCloud(Mat frameDisp, Mat &pcloud, float B, float f, float c_u, float c_v) {
	Mat depthMap;
	int heightImg = frameDisp.rows;
	int widthImg = frameDisp.cols;

	depthMap.create(heightImg, widthImg, CV_32F);
	for (int h = 0; h < heightImg; ++h) {
		for (int w = 0; w < widthImg; ++w) {
			depthMap.at<float>(h, w) = (f*B)*(1 / (float)frameDisp.at<uchar>(h, w));
		}
	}

	for (int h = 0; h < heightImg; ++h) {
		for (int w = 0; w < widthImg; ++w) {
			float depth = depthMap.at<float>(h, w);
			float x = (w - c_u) * (depth / f);
			float y = (h - c_v) * (depth / f);
			pcloud.at<Vec3f>(h, w) = Vec3f(x, y, depth);
		}
	}
}