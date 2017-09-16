#include <opencv2/contrib/contrib.hpp>
#include "utils.h"

Mat getColorMapForData(Mat labels, double data[], int nrElements);
Mat normalizeAndApplyColorMap(Mat mat);

void colorMaps(Mat image, Mat labels, int numberOfSuperpixels, double nalpha[], double heightAbGr[], double localPl[], double neighborPl[],
	double cameraPath[], double meanL[], double meanA[], double meanB[], Mat disp) {

	imwrite("..\\..\\VisualizedFeatures\\image.png", image);
	imwrite("..\\..\\VisualizedFeatures\\nalpha.png", getColorMapForData(labels, nalpha, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\heightAbGr.png", getColorMapForData(labels, heightAbGr, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\localPl.png", getColorMapForData(labels, localPl, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\neighborPl.png", getColorMapForData(labels, neighborPl, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\cameraPath.png", getColorMapForData(labels, cameraPath, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\meanL.png", getColorMapForData(labels, meanL, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\meanA.png", getColorMapForData(labels, meanA, numberOfSuperpixels));
	imwrite("..\\..\\VisualizedFeatures\\meanB.png", getColorMapForData(labels, meanB, numberOfSuperpixels));


	imwrite("..\\..\\VisualizedFeatures\\disp.png", normalizeAndApplyColorMap(disp));


	/*imshow("image", image);
	imshow("nalpha", getColorMapForData(labels, nalpha, numberOfSuperpixels));
	imshow("heightAbGr", getColorMapForData(labels, heightAbGr, numberOfSuperpixels));
	imshow("localPl", getColorMapForData(labels, localPl, numberOfSuperpixels));
	imshow("neighborPl", getColorMapForData(labels, neighborPl, numberOfSuperpixels));*/
	/*imshow("cameraPath", getColorMapForData(labels, cameraPath, numberOfSuperpixels));
	imshow("meanL", getColorMapForData(labels, meanL, numberOfSuperpixels));
	imshow("meanA", getColorMapForData(labels, meanA, numberOfSuperpixels));
	imshow("meanB", getColorMapForData(labels, meanB, numberOfSuperpixels));*/
	waitKey(0);
}

Mat getColorMapForData(Mat labels, double data[], int nrElements) {
	Mat in = Mat(labels.rows, labels.cols, CV_8U);

	double max = 0, min = 5000000;
	for (int i = 0; i < nrElements; ++i) {
		max = data[i] > max ? data[i] : max;
		min = data[i] < min ? data[i] : min;
	}

	int heightImg = labels.rows;
	int widthImg = labels.cols;
	for (int h = 0; h < heightImg; ++h) {
		int* labelsRow = labels.ptr<int>(h);
		uchar* inRow = in.ptr<uchar>(h);
		for (int w = 0; w < widthImg; ++w) {
			int normalizedValue = (data[labelsRow[w]] * 255.0) / max;
			inRow[w] = (char)normalizedValue;
		}
	}
	
	Mat out;
	applyColorMap(in, out, COLORMAP_JET);
	return out;
}


Mat normalizeAndApplyColorMap(Mat mat) {
	Mat coloredMap;
	Mat normalized = Mat(mat.rows, mat.cols, CV_8U);
	double minVal;
	double maxVal;
	minMaxLoc(mat, &minVal, &maxVal);
	int heightImg = mat.rows;
	int widthImg = mat.cols;
	for (int h = 0; h < heightImg; ++h) {
		uchar* dispRow = mat.ptr<uchar>(h);
		uchar* normalizedRow = normalized.ptr<uchar>(h);
		for (int w = 0; w < widthImg; ++w) {
			int normalizedValue = (dispRow[w] * 255.0) / maxVal;
			dispRow[w] = (char)normalizedValue;
			normalizedRow[w] = (uchar)normalizedValue;
		}
	}
	applyColorMap(normalized, coloredMap, COLORMAP_JET);
	return coloredMap;
}