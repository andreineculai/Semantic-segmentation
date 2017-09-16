#include "utils.h"
#include <list>
class PointData {
public:
	//static int PointData::ncl;
	float x, y, z;
	int r, g, b;
	int trackLength;
	int* previousClassifications = NULL;

	static int& ncl() { static int ncl; return ncl; }

	PointData(){}

	int getUFromPoint(float f, float c_u) {
		//double a = ((x * f) / z + c_u);
		//if (a < 0 || a >= 800) cout << a << endl;
		return round((x * f) / z + c_u);
	}

	int getVFromPoint(float f, float c_v) {
		//double a = ((y * f) / z + c_v);
		//if (a < 0 || a >= 400) cout << a << endl;
		return round((y * f) / z + c_v);
	}

	double distanceToPoint(float x2, float y2, float z2) {
		return sqrt((x2 - x)*(x2 - x) + (y2 - y)*(y2 - y) + (z2 - z)*(z2 - z));
	}


	PointData(const PointData &pd){
		this->x = pd.x;
		this->y = pd.y;
		this->z = pd.z;
		this->r = pd.r;
		this->g = pd.g;
		this->b = pd.b;
		trackLength = pd.trackLength;
		if (pd.previousClassifications) {
			initializePreviousClassifications(pd.previousClassifications);
		}
	}

	~PointData(){
		if (previousClassifications)
			delete[] previousClassifications;
	}

	PointData(float x, float y, float z, int r, int g, int b, int tl = 1, bool predict = false, int* votes = NULL) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->r = r;
		this->g = g;
		this->b = b;
		trackLength = tl;
		if (predict) {
			if (votes == NULL) {
				this->initializePreviousClassifications();
			}
			else {
				this->initializePreviousClassifications(votes);
			}

		}
	}

	void initializePreviousClassifications() {
		previousClassifications = new int[ncl()]{};
	}

	void initializePreviousClassifications(int* votes) {
		previousClassifications = new int[ncl()]{};
		if (votes) {
			for (int i = 0; i < ncl(); ++i){
				previousClassifications[i] = votes[i];
			}
		}
	}

	void addVotes(double* votes){
		///int sum = 0;
		for (int i = 0; i < ncl(); ++i){
		//	sum += votes[i];
			previousClassifications[i] += (int)votes[i];
		}
		//cout << "s = " << sum << endl;
	}



};

class PointCloud {
public:
	list<PointData> points;

	void applyTransformation(Matrix pose) {
		for (list<PointData>::iterator it = points.begin(); it != points.end(); ++it) {
			Vec3f point = Vec3f(it->x, it->y, it->z);
			Vec3f newPoint = Vec3f(0, 0, 0);
			for (int32_t i = 0; i < pose.m - 1; i++) {
				for (int32_t j = 0; j < pose.n - 1; j++) {
					newPoint[i] += pose.val[i][j] * point[j];
				}
				newPoint[i] += pose.val[i][pose.n - 1];
			}
			it->x = newPoint[0];
			it->y = newPoint[1];
			it->z = newPoint[2];
		}
	}

	~PointCloud() {
		points.clear();
	}
};


class Votes {
public:
	int* votes;
	int numberOfVoteCasts;

	Votes() {
		numberOfVoteCasts = 0;
		votes = new int[PointData::ncl()]{};
	}

	~Votes() {
		delete[] votes;
	}

	void resetVotes() {
		numberOfVoteCasts = 0;
		for (int i = 0; i < PointData::ncl(); ++i) {
			votes[i] = 0;
		}
	}

	void setVotes(int * votes, int tl) {
		for (int i = 0; i < PointData::ncl(); ++i) {
			this->votes[i] = votes[i];
		}
		this->numberOfVoteCasts = tl;
	}

	void addCurrentFrameVotes(double * votes, int voters) {
		for (int i = 0; i < PointData::ncl(); ++i) {
			this->votes[i] += voters * votes[i];
		}
		this->numberOfVoteCasts += voters;
	}

	void addVotes(Votes * votes) {
		//int sum = 0;
		for (int i = 0; i < PointData::ncl(); ++i) {
			this->votes[i] += votes->votes[i];
		//	sum += votes->votes[i];
		}
		//cout << sum << endl;
		this->numberOfVoteCasts += votes->numberOfVoteCasts;
	}

	double* getMean() {
		double* meanVotes = new double[PointData::ncl()];
		for (int i = 0; i < PointData::ncl(); ++i) {
			meanVotes[i] = (double)votes[i] / numberOfVoteCasts;
		}
		return meanVotes;
	}

};
