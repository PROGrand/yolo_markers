#include <iostream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include "dataset_prepare.h"

using namespace std;
using cv::Mat;
using cv::Point;
using cv::Scalar;
using cv::Size;
using cv::Rect;
using namespace boost::filesystem;
using namespace boost::program_options;


variables_map opts;

void parse_options(int argc, char* argv[]) {
	options_description desc{ "Options" };
	desc.add_options()
		("help,h", "Help screen")
		("show", "Show intermediate view")
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


	struct context {
		int id;
		Rect selection;
	};


	class op {
	public:
		virtual void operator()(const cv::Mat& img, dp::context& context) const {}
	};

	class resize : public op {

		const int min_size, max_size, size_step;
		const op& next;

	public:
		resize(int min_size, int max_size, int size_step, const op& next) :
			min_size(min_size),
			max_size(max_size),
			size_step(size_step),
			next(next) {
		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {
			for (int size = min_size; size < max_size; size += size_step) {
				cv::Mat resized;
				cv::resize(img, resized, cv::Size(size, size), 0, 0, cv::INTER_LANCZOS4);
				next(resized, context);
			}
		}
	};

	class shear : public op {

		const float shear_min, shear_max, shear_step;
		const op& next;

	public:
		shear(float smin/* = 0*/, float smax/* = 1.5*/, float sstep/* = 0.5*/, const op& next) :
			shear_min(smin),
			shear_max(smax),
			shear_step(sstep),
			next(next) {
		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {
			for (float s = shear_min; s <= shear_max; s += shear_step) {
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

				next(sheared, context);

			}
		}
	};


	class rotate : public op {

		const float min_angle, max_angle, angle_step;
		const op& next;

	public:
		rotate(float min_angle/* = -45*/, float max_angle/* = 45*/, float angle_step/* = 15*/, const op& next) :
			min_angle(min_angle),
			max_angle(max_angle),
			angle_step(angle_step),
			next(next) {
		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {
			for (float angle = min_angle; angle <= max_angle; angle += angle_step) {

				auto center = Point((img.cols - 1) / 2, (img.rows - 1) / 2);

				Mat matRotation = cv::getRotationMatrix2D(center, angle, 1);
				cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
				matRotation.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
				matRotation.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

				Mat rotated;
				cv::warpAffine(img, rotated, matRotation, bbox.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, Scalar(127, 127, 127, 0));

				next(rotated, context);
			}

		}
	};


	class brightness : public op {

		const float min_brightness, max_brightness, brightness_step;
		const op& next;

	public:
		brightness(float min_brightness/* = -16 * 14*/, float max_brightness/* = 16 * 14*/, float brightness_step/* = 16 * 4*/, const op& next) :
			min_brightness(min_brightness),
			max_brightness(max_brightness),
			brightness_step(brightness_step),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {
			for (float brightness = min_brightness; brightness <= max_brightness; brightness += brightness_step) {
				Mat bright = img + cv::Scalar(brightness, brightness, brightness, 0);
				next(bright, context);
			}
		}
	};

	class blur : public op {

		const int min_blur, max_blur, blur_step;
		const op& next;

	public:
		blur(int min_blur/* = -16 * 14*/, int max_blur/* = 16 * 14*/, int blur_step/* = 16 * 4*/, const op& next) :
			min_blur(min_blur),
			max_blur(max_blur),
			blur_step(blur_step),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {
			for (int blur = min_blur; blur <= max_blur; blur += blur_step) {

				if (0 == blur) {
					next(img, context);
				}
				else {
					Mat blurred;
					cv::GaussianBlur(img, blurred, cv::Size((blur - 1) * 2 + 1, (blur - 1) * 2 + 1), 0, 0);
					next(blurred, context);
				}
			}
		}
	};

	class noise : public op {

		const float mean, sigma;
		const op& next;

	public:
		noise(float mean/* = 50*/, float sigma/* = 50*/, const op& next) :
			mean(mean),
			sigma(sigma),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {
			Mat noise = Mat::zeros(img.rows, img.cols, img.type());
			std::vector<float> m = { mean };
			std::vector<float> s = { sigma };
			cv::randn(noise, m, s);

			Mat noised = img + noise;
			next(noised, context);
		}
	};

	class pad : public op {

		const int width, height;
		const op& next;

	public:
		pad(int width/* = 416*/, int height/* = 416*/, const op& next) :
			width(width),
			height(height),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {

			Mat padded(height, width, img.type(), Scalar(127, 127, 127, 0));

			if (img.cols <= width && img.rows <= height) {
				cv::Rect rect((width - img.cols) / 2, (height - img.rows) / 2, img.cols, img.rows);
				img.copyTo(padded(rect));
				context.selection = rect;
				next(padded, context);
			}
		}
	};


	class save : public op {

		const boost::filesystem::path folder;
		const op& next;

	public:
		save(const boost::filesystem::path& folder, const op& next) :
			folder(folder),
			next(next)
		{

		}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {

			boost::filesystem::create_directories(folder);

			volatile static int num = 0;
			auto path = folder / (boost::format("%1i.jpg") % num).str();
			auto path_txt = folder / (boost::format("%1i.txt") % num).str();

			num++;

			std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 100 };
			cv::imwrite(path.string(), img, params);

			std::ofstream txt(path_txt.string());
			txt << context.id << " " << (float)context.selection.x / img.cols << " " << (float)context.selection.y / img.rows << " "
				<< (float)context.selection.width / img.cols << " " << (float)context.selection.height / img.rows;
			txt.close();
			

			next(img, context);
		}
	};


	class show : public op {

	public:
		show() {}

		virtual void operator()(const cv::Mat& img, dp::context& context) const {

			if (opts.count("show")) {
				Mat out = img.clone();
				cv::rectangle(out, context.selection, Scalar(0, 0, 255), 3, cv::LINE_AA);
				cv::imshow("img", out);
				cv::waitKey(1);
			}
		}
	};
}

void prepare_marker(const boost::filesystem::path& src, const boost::filesystem::path& dst) {
	cout << src.string() << endl;
	

	int min_size = opts["minsize"].as<int>();
	int max_size = opts["maxsize"].as<int>();
	int size_step = opts["stepsize"].as<int>();

	static int id = 0;

	dp::context context;
	context.id = id++;

	cv::Mat img = cv::imread(src.string());
	cv::cvtColor(img, img, CV_32F);

	dp::resize(min_size, max_size, size_step,
		dp::shear(0, 1.0, 0.5, 
			dp::rotate(-45, 45, 15,
				dp::pad(416, 416,
					dp::brightness(-16 * 2, 16 * 10, 16 * 4,
						dp::blur(0, 1, 1,
							dp::noise(10, 10,
								dp::save(dst,
									dp::show()))))))))(img, context);

}


int main(int argc, char* argv[]) {


	parse_options(argc, argv);


	boost::filesystem::path src_dir(opts["src"].as<std::string>());
	boost::filesystem::path dst_dir(opts["dst"].as<std::string>());

	for (boost::filesystem::recursive_directory_iterator i(src_dir / "markers"), end; i != end; i++) {

		if (is_regular_file(*i)) {
			prepare_marker(i->path(), dst_dir / "positive");
		}
	}

	return 0;
}