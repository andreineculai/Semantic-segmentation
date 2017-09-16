#include "gSlicUtils.h"


const gSLICr::IntImage* getSegmentationMask(Mat frameLeft, Mat& segmented) {
	gSLICr::objects::settings gslic_settings = gSlicSettingsFactory(frameLeft.rows, frameLeft.cols);
	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(gslic_settings);

	// gSLICr takes gSLICr::UChar4Image as input and out put
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(gslic_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(gslic_settings.img_size, true, true);

	Size s(gslic_settings.img_size.x, gslic_settings.img_size.y);

	load_image(frameLeft, in_img);
	gSLICr_engine->Process_Frame(in_img);
	/*gSLICr_engine->Draw_Segmentation_Result(out_img);
	Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);
	load_image(out_img, boundry_draw_frame);
	imwrite(baseFilePath + "segmented.png", boundry_draw_frame);
	*///imshow("segmentation", boundry_draw_frame);
	//waitKey(0);
	const gSLICr::IntImage *mask = gSLICr_engine->Get_Seg_Res();
	load_image(mask, segmented);
	delete in_img;
	delete out_img;
	delete gSLICr_engine;
}

void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y; y++)
	for (int x = 0; x < outimg->noDims.x; x++)
	{
		int idx = x + y * outimg->noDims.x;
		outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
		outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
		outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
	}
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
	for (int x = 0; x < inimg->noDims.x; x++)
	{
		int idx = x + y * inimg->noDims.x;
		outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
		outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
		outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
	}
}

void load_image(const gSLICr::IntImage* inimg, Mat& outimg)
{
	const int* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
	for (int x = 0; x < inimg->noDims.x; x++)
	{
		int idx = x + y * inimg->noDims.x;
		outimg.at<int>(y, x) = inimg_ptr[idx];
	}
}

gSLICr::objects::settings gSlicSettingsFactory(int height, int width) {
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = width;
	my_settings.img_size.y = height;
	my_settings.no_segs = floor(height*width / 100) + 1;
	my_settings.spixel_size = 32;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::CIELAB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step
	return my_settings;
}