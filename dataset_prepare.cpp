#include <iostream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "dataset_prepare.h"

using namespace std;
using cv::Mat;
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
		("stepsize", value<int>()->default_value(10), "Minimum marker size")
		;
	
	store(parse_command_line(argc, argv, desc), opts);

	if (opts.count("help")) {
		std::cout << desc << '\n';
		exit(0);
	}
}

static Mat gamma_table(float gamma) {

	static std::map<float, Mat> lookUpTables;

	auto it = lookUpTables.find(gamma);
	if (it != lookUpTables.end()) {
		return it->second;
	}

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow((float)i / 255.0, gamma) * 255.0);

	lookUpTables[gamma] = lookUpTable;

	return lookUpTable;
}


void prepare_marker(const std::string& path) {
	cout << path << endl;

	cv::Mat img = cv::imread(path);

	cv::cvtColor(img, img, CV_32F);

	int min_size = opts["minsize"].as<int>();
	int max_size = opts["maxsize"].as<int>();
	int size_step = opts["stepsize"].as<int>();

	for (int size = min_size; size < max_size; size += size_step) {
		cv::Mat resized;
		cv::resize(img, resized, cv::Size(size, size), 0, 0, cv::INTER_LANCZOS4);

		for (float brightness = -16 * 14; brightness < 16 * 14; brightness += 16 * 4) {

			Mat contrasted = resized + cv::Scalar(brightness, brightness, brightness, 0);
			cv::imshow("contrasted", contrasted);
			cv::waitKey(50);
		}

	}
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