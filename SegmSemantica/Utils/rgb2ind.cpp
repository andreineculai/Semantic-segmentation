#include "utils.h"

Mat rgb2ind(Mat image, int Map[][3]){
	int heightImg = image.rows;
	int widthImg = image.cols;
	Mat indexImg = Mat(heightImg, widthImg, CV_8U);
	int numberOfMapElements = 11;
	for (int h = 0; h < heightImg; ++h) {
		Vec3b *row = image.ptr<Vec3b>(h);
		for (int w = 0; w < widthImg; ++w) {
			int i;
			for (i = 0; i < numberOfMapElements; ++i) {
				if (row[w][2] == Map[i][0] &&
					row[w][1] == Map[i][1] &&
					row[w][0] == Map[i][2]) {
					
					indexImg.at<uchar>(h, w) = i;
					//indexImg.at<uchar>(h, w) = groupClassesRealDataset(i);
					break;
				}
			}
			

			if (i == numberOfMapElements) {
				//indexImg.at<uchar>(h, w) = groupClassesRealDataset(numberOfMapElements - 1);
				indexImg.at<uchar>(h, w) = numberOfMapElements - 1;
			}
		}
	}
	return indexImg;
}

int groupClassesRealDataset(int cl) {
	switch (cl)
	{
	case 0: return 0;
	case 1: return 1;
	case 2: return 2;
	case 3: return 3;
	case 4: return 4;
	case 5: return 5;
	case 6: return 4;
	case 7: return 4;
	case 8: return 5;
	case 9: return 6;
	case 10: return 1;
	default:
		break;
	}
}