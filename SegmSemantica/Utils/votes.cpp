#include "utils.h"
#include "pointcloud.h"

class Votes {
	int* votes;
	int tl;

	Votes() {
		tl = 0;
		votes = new int[PointData::ncl()];
	}

	void resetVotes() {
		for (int i = 0; i < PointData::ncl(); ++i) {
			votes[i] = 0;
		}
	}

	void setVotes(int * votes) {
		for (int i = 0; i < PointData::ncl(); ++i) {
			this->votes[i] = votes[i];
		}
	}

};

