#include <iostream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "dataset_prepare.h"

using namespace std;
using cv::Mat;
using cv::Point;
using cv::Scalar;
using cv::Size;
using namespace boost::filesystem;
using namespace boost::program_options;


variables_map opts;

void parse_options(int argc, char* argv[]) {
	options_description desc{ "Options" };
	desc.add_options()
		("help,h", "Help screen")
		("src", value<string>()->default_value("dataset"), "Source folder")
		("dst", value<string>()->default_value("dataset"), "Destination folder")
		("minsize", value<int>()->default_value(30), "Minimum marker size")
		("maxsize", value<int>()->default_value(416 / 2), "Maximum marker size")
		("stepsize", value<int>()->default_value(40), "Minimum marker size")
		;

	store(parse_command_line(argc, argv, desc), opts);

	if (opts.count("help")) {
		std::cout << desc << '\n';
		exit(0);
	}
}


namespace dp {


	class op {
	public:
		virtual void operator()(const cv::Mat& img) const {}
	};

	class resize : public op {

		const int min_size, max_size, size_step;
		const op& next;

	public:
		resize(int min_size, int max_size, int size_step, op& next) :
			min_size(min_size),
			max_size(max_size),
			size_step(size_step),
			next(next) {
		}

		virtual void operator()(const cv::Mat& img) const {
			for (int size = min_size; size < max_size; size += size_step) {
				cv::Mat resized;
				cv::resize(img, resized, cv::Size(size, size), 0, 0, cv::INTER_LANCZOS4);
				next(resized);
			}
		}
	};

	class shear : public op {

		const float smin, smax, sstep;
		const op& next;

	public:
		shear(float smin/* = 0*/, float smax/* = 1.5*/, float sstep/* = 0.5*/, op& next) :
			smin(smin),
			smax(smax),
			sstep(sstep),
			next(next) {
		}

		virtual void operator()(const cv::Mat& img) const {
			for (float s = smin; s < smax; s += sstep) {
				Mat M(2, 3, CV_32F);

				M.at<float>(0, 0) = 1;
				M.at<float>(0, 1) = 0;
				M.at<float>(0, 2) = 0;

				M.at<float>(1, 0) = s;
				M.at<float>(1, 1) = 1;
				M.at<float>(1, 2) = 0;

				vector<cv::Point2f> roi_points = {
					{0, 0},
					{(float)img.cols, 0},
					{(float)img.cols, (float)img.rows},
					{0, (float)img.rows}
				}, roi_points_sheared;

				cv::transform(roi_points, roi_points_sheared, M);

				auto bounding_sheared = cv::boundingRect(roi_points_sheared);

				Mat sheared;
				cv::warpAffine(img, sheared, M, bounding_sheared.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, Scalar(127, 127, 127, 0));

				next(sheared);

			}
		}
	};


	class rotate : public op {

		const float min_angle, max_angle, angle_step;
		const op& next;

	public:
		rotate(float min_angle/* = -45*/, float max_angle/* = 45*/, float angle_step/* = 15*/, op& next) :
			min_angle(min_angle),
			max_angle(max_angle),
			angle_step(angle_step),
			next(next) {
		}

		virtual void operator()(const cv::Mat& img) const {
			for (float angle = min_angle; angle < max_angle; angle += angle_step) {

				auto center = Point((img.cols - 1) / 2, (img.rows - 1) / 2);

				Mat matRotation = cv::getRotationMatrix2D(center, angle, 1);
				cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
				matRotation.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
				matRotation.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

				Mat rotated;
				cv::warpAffine(img, rotated, matRotation, bbox.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, Scalar(127, 127, 127, 0));

				next(rotated);
			}

		}
	};


	class brightness : public op {

		const float min_brightness, max_brightness, brightness_step;
		const op& next;

	public:
		brightness(float min_brightness/* = -16 * 14*/, float max_brightness/* = 16 * 14*/, float brightness_step/* = 16 * 4*/, op& next) :
			min_brightness(min_brightness),
			max_brightness(max_brightness),
			brightness_step(brightness_step),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img) const {
			for (float brightness = min_brightness; brightness < max_brightness; brightness += brightness_step) {
				Mat bright = img + cv::Scalar(brightness, brightness, brightness, 0);
				next(bright);
			}
		}
	};

	class blur : public op {

		const int min_blur, max_blur, blur_step;
		const op& next;

	public:
		blur(int min_blur/* = -16 * 14*/, int max_blur/* = 16 * 14*/, int blur_step/* = 16 * 4*/, op& next) :
			min_blur(min_blur),
			max_blur(max_blur),
			blur_step(blur_step),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img) const {
			for (int blur = min_blur; blur < max_blur; blur += blur_step) {

				if (0 == blur) {
					next(img);
				}
				else {
					Mat blurred;
					cv::GaussianBlur(img, blurred, cv::Size((blur - 1) * 2 + 1, (blur - 1) * 2 + 1), 0, 0);
					next(blurred);
				}
			}
		}
	};


	class show : public op {

	public:
		show() {}

		virtual void operator()(const cv::Mat& img) const {

			cv::imshow("img", img);
			cv::waitKey(1);
		}
	};
}

void prepare_marker(const std::string& path) {
	cout << path << endl;

	cv::Mat img = cv::imread(path);

	cv::cvtColor(img, img, CV_32F);

	int min_size = opts["minsize"].as<int>();
	int max_size = opts["maxsize"].as<int>();
	int size_step = opts["stepsize"].as<int>();

	

	dp::resize(min_size, max_size, size_step,
		dp::shear(0, 1.5, 0.5, 
			dp::rotate(-45, 45, 15,
				dp::brightness(-16 * 14, 16 * 14, 16 * 4,
					dp::blur(0, 20, 4,
						dp::show())))))(img);
}


int main(int argc, char* argv[]) {


	parse_options(argc, argv);


	boost::filesystem::path src_dir(opts["src"].as<std::string>());

	for (boost::filesystem::recursive_directory_iterator i(src_dir / "markers"), end; i != end; i++) {

		if (is_regular_file(*i)) {
			prepare_marker(i->path().string());
		}
	}

	return 0;
}