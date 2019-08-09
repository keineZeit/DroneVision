#pragma once

#include <iostream>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class VideoStream {
public:
	VideoStream();
	~VideoStream();
	void start(int dev);
	void start(string url);
	cv::Mat read();
	void release();
	double get(int propId);

private:
	cv::VideoCapture stream;
	std::thread streamThread;
	bool stopped;
	cv::Mat frame;
};
