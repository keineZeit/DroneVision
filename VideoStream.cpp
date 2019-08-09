#pragma once

#include <iostream>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "VideoStream.hpp"

using namespace std;

void multithread_update(cv::VideoCapture &_stream, cv::Mat &_frame, bool &_stopped) {
	while (true) {
		if (_stopped) return;
		_stream >> _frame;
	}
}

VideoStream::VideoStream() {
	this->stopped = false;
}

void VideoStream::start(int dev) {
	this->stream.open(dev);
	if (!this->stream.isOpened())
		throw "[ERROR] Error when reading stream";
	streamThread = std::thread(multithread_update, std::ref(this->stream), std::ref(this->frame), std::ref(this->stopped));
}

void VideoStream::start(string url) {
	this->stream.open(url);
	if (!this->stream.isOpened())
		throw "[ERROR] Error when reading stream";
	streamThread = std::thread(multithread_update, std::ref(this->stream), std::ref(this->frame), std::ref(this->stopped));
}

cv::Mat VideoStream::read() {
	return this->frame;
}

void VideoStream::release() {
	this->stopped = true;
	streamThread.join();
}

double VideoStream::get(int propId) {
	return this->stream.get(propId);
}

VideoStream::~VideoStream() {}
