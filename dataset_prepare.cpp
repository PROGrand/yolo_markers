#include <iostream>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::program_options;


void parse_options(int argc, char* argv[]) {
	options_description desc{ "Options" };
	desc.add_options()
		("help,h", "Help screen")
		("src", value<string>()->default_value("."), "Source folder")
		("age", value<int>()->notifier(on_age), "Age");
}


int main(int argc, char* argv[]) {


	parse_options(argc, argv);

	cv::Mat white = cv::Mat::zeros(360, 640, CV_8UC3);

	white = cv::Scalar(0, 255, 0, 0);

	cv::imshow("white", white);

	cv::waitKey(0);

	boost::filesystem::path targetDir(".");

	for (boost::filesystem::recursive_directory_iterator i(targetDir), end; i != end; i++) {

		if (is_regular_file(*i)) {
			cout << i->path().string() << endl;

			prepare_file(i->path());
		}
	}

	return 0;
}