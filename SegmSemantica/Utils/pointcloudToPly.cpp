#include "utils.h"
#include <fstream>

void createPlyFileFromPointcloud(Mat image, Mat pointcloud, String filepath) {
	ofstream ply(filepath);
	int heightImg = image.rows;
	int widthImg = image.cols;
	
	ply << "ply" << endl;
	ply << "format ascii 1.0" << endl;
	ply << "element vertex " << heightImg * widthImg << endl;
	ply << "property float x" << endl;
	ply << "property float y" << endl;
	ply << "property float z" << endl;
	ply << "property uchar red" << endl;
	ply << "property uchar green" << endl;
	ply << "property uchar blue" << endl;
	ply << "end_header" << endl;

	
	Vec3f currentPoint;
	Vec3b colorInfo;
	for (int h = 0; h < heightImg; ++h) {
		for (int w = 0; w < widthImg; ++w) {
			currentPoint = pointcloud.at<Vec3f>(h, w);
			colorInfo = image.at<Vec3b>(h, w);
			ply << currentPoint[0] << ' ' << currentPoint[1] << ' ' << currentPoint[2] << ' ' <<
				(int)colorInfo[2] << ' ' << (int)colorInfo[1] << ' ' << (int)colorInfo[0] << endl;
		}
	}
}
