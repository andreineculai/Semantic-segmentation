#include "../Utils/utils.h"
#include "../RandomForest\rf.h"

void createClassificationModel(Mat X_Train, vector<int> totalLabels, string fileName, int ncl, int numberOfFeatures) {
	//prepare training parameters
	cout << X_Train.rows << endl;
	cout << X_Train.cols << endl;
	double *X = (double *)X_Train.data;
	int *Y = totalLabels.data();
	int dimx[2] = { X_Train.cols, X_Train.rows };
	//(dimx[0] - number of features)
	//(dimx[1] - number of samples)
	int* cat = new int[dimx[0]];
	for (int i = 0; i < dimx[0]; ++i) {
		cat[i] = 1;
	}
	int maxcat = 1;
	int sampsize = dimx[1];
	int strata = 1;
	int Options[10] = { 0, 0, 0, 0, 0, 0, 1, 1, 0, 0 };
	int ntree = 80;
	int mtry = floor(sqrt((double)dimx[0]));
	int ipi = 0;
	double* classwt = new double[ncl];
	for (int i = 0; i < ncl; ++i) {
		classwt[i] = 1;
	}
	double* cut = new double[ncl];
	for (int i = 0; i < ncl; ++i) {
		cut[i] = 1.0 / ncl;
	}
	int nodesize = 1;
	int* outcl = new int[dimx[1]]();
	int* counttr = new int[ncl * dimx[1]]();
	double prox = 1;
	double* imprt = new double[dimx[0]]();
	double impsd = 1;
	double impmat = 1;
	int nrnodes = 2 * (int)floor((double)dimx[1] / nodesize) + 1;
	int* ndbigtree = new int[nrnodes * ntree]();
	int* nodestatus = new int[nrnodes * ntree]();
	int* bestvar = new int[nrnodes * ntree]();
	int* treemap = new int[nrnodes * ntree * 2]();
	int* nodepred = new int[nrnodes * ntree]();
	double* xbestsplit = new double[nrnodes * ntree]();
	double* errtr = new double[(ncl + 1) * ntree]();
	int testdat = 0;
	double xts = 1;
	int clts = 1;
	int nts = 1;//twonorm example had it 0
	double* countts = new double[ncl * nts]();
	int outclts = 1;//twonorm example had it 0
	int labelts = 0;
	double proxts = 1;
	double errts = 1;
	int* inbag = new int[dimx[1]]();
	printf("Random Forest\n");
	classRF(X, dimx, Y, &ncl, cat, &maxcat, &sampsize, &strata, Options, &ntree, &mtry,
		&ipi, classwt, cut, &nodesize, outcl, counttr, &prox, imprt, &impsd, &impmat,
		&nrnodes, ndbigtree, nodestatus, bestvar, treemap, nodepred, xbestsplit, errtr,
		&testdat, &xts, &clts, &nts, countts, &outclts, labelts, &proxts, &errts, inbag);
	printf("Writing results to file\n");
	FILE* model;
	fopen_s(&model, fileName.c_str(), "wb");
	while (model == NULL) {
		fopen_s(&model, fileName.c_str(), "wb");
	}
	fwrite(&ncl, 1, sizeof(ncl), model);
	fwrite(&numberOfFeatures, 1, sizeof(numberOfFeatures), model);
	fwrite(&maxcat, 1, sizeof(maxcat), model);
	fwrite(&nrnodes, 1, sizeof(nrnodes), model);
	fwrite(&ntree, 1, sizeof(ntree), model);

	fwrite(xbestsplit, 1, nrnodes * ntree*sizeof(*xbestsplit), model);
	fwrite(classwt, 1, ncl * sizeof(*classwt), model);
	fwrite(cut, 1, ncl * sizeof(*cut), model);
	fwrite(treemap, 1, nrnodes * ntree * 2 * sizeof(*treemap), model);
	fwrite(nodestatus, 1, nrnodes * ntree * sizeof(*nodestatus), model);
	fwrite(cat, 1, dimx[0] * sizeof(*cat), model);
	fwrite(nodepred, 1, nrnodes * ntree * sizeof(*nodepred), model);
	fwrite(bestvar, 1, nrnodes * ntree * sizeof(*bestvar), model);
	fwrite(ndbigtree, 1, nrnodes * ntree * sizeof(*ndbigtree), model);
	fclose(model);

}