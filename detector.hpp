#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

class Detector {
public:
	virtual Mat predict(Mat inputFrame) = 0;
};


class HaarFaceDetector : public Detector {
public:
	HaarFaceDetector(string _modelName, float _threshold);
	~HaarFaceDetector();
	Mat predict(Mat inputFrame);

protected:
	CascadeClassifier model;
	float threshold;
};


class YoloObjectDetector : public Detector {
public:
	YoloObjectDetector(string _modelName, float _threshold);
	~YoloObjectDetector();
	Mat predict(Mat inputFrame);

protected:
	Net model;
	float threshold;
	vector<string> classes;
	vector<String> names;

private:
	Mat blob;
	vector<String> setOutputsNames();
	void setClasses(string pathToYoloFiles);
	void drawPredictions(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
	void postprocess(Mat& frame, const vector<Mat>& outs);
};
