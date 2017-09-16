#include "utils.h"


/////* Input:
////        - int* klabels: the labeled image (each pixel has a value equal to its superpixel index)
////        - int K: the number of superpixels
//// * Output:
////        - Mat M: the adjacency matrix (M[i][j] = M[j][i] = 1 if the superpixels i and j are adjacent, and = 0 otherwise)
////*/
Mat getAdjacencyMatrix(const int* klabels, int width, int height, int K)
{
	/// Create a KxK matrix and initialize to 0
	Mat M(K, K, CV_32S, Scalar(0));

	/// Scan the labeled image
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			// Get the label of the current pixel and the ones of its neighbors
			int k = klabels[y*width + x];
			int kleft = klabels[y*width + x - 1];
			int kright = klabels[y*width + x + 1];
			int kup = klabels[(y - 1)*width + x];
			int kdown = klabels[(y + 1)*width + x];
			if (k != kleft)
			{
				M.at<int>(k, kleft) = 1;
				M.at<int>(kleft, k) = 1;
			}
			if (k != kright)
			{
				M.at<int>(k, kright) = 1;
				M.at<int>(kright, k) = 1;
			}
			if (k != kup)
			{
				M.at<int>(k, kup) = 1;
				M.at<int>(kup, k) = 1;
			}
			if (k != kdown)
			{
				M.at<int>(k, kdown) = 1;
				M.at<int>(kdown, k) = 1;
			}
		}
	}
	return M;
}