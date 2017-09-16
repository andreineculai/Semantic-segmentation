#include "../gSLICr/gSLICr.h"
#include "../gSLICr/gSLICr_defines.h"
#include "utils.h"


void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg);

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg);

void load_image(const gSLICr::IntImage* inimg, Mat& outimg);

gSLICr::objects::settings gSlicSettingsFactory(int height, int width);

const gSLICr::IntImage* getSegmentationMask(Mat frameLeft, Mat& segmented);
