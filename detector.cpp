#include "detector.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

constexpr auto OPENCV_INSTALL_PATH = "C:\\opencv\\opencv-4.1.1\\build\\install\\";

// --- HAAR FACE DETECTOR --------------------------------------------------------------------------

HaarFaceDetector::HaarFaceDetector(string _modelName, float _threshold = 0.7) {
	this->threshold = _threshold;

	string modelPath = OPENCV_INSTALL_PATH + (string)"etc\\haarcascades\\" + _modelName;
	bool modelLoaded = this->model.load(modelPath);
	if (!modelLoaded) {
		cerr << "[ERROR] Error Loading XML file" << endl;
		return;
	}
}

Mat HaarFaceDetector::predict(Mat inputFrame) {
	vector<cv::Rect> faces;
	Point p1;
	Point p2;
	this->model.detectMultiScale(inputFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	for (int i = 0; i < faces.size(); i++) {
		p1.x = faces[i].x;
		p1.y = faces[i].y;
		p2.x = faces[i].x + faces[i].width;
		p2.y = faces[i].y + faces[i].height;
		rectangle(inputFrame, p1, p2, Scalar(51, 204, 51), 3, 8, 0);
	}
	return inputFrame;
}

HaarFaceDetector::~HaarFaceDetector() {}


// --- YOLO OBJECT DETECTOR ------------------------------------------------------------------------

YoloObjectDetector::YoloObjectDetector(string _modelName, float _threshold = 0.5) {

	this->threshold = _threshold;
	string pathToYoloFiles = "models\\yolo\\";

	// Load names of classes
	setClasses(pathToYoloFiles);

	// Load the network
	string modelConfiguration = pathToYoloFiles + _modelName + ".cfg";
	string modelWeights = pathToYoloFiles + _modelName + ".weights";
	this->model = readNetFromDarknet(modelConfiguration, modelWeights);
	this->model.setPreferableBackend(DNN_BACKEND_OPENCV);
	this->model.setPreferableTarget(DNN_TARGET_CPU);

	// Get the names of the output layers
	this->names = setOutputsNames();
}

Mat YoloObjectDetector::predict(Mat inputFrame) {

	if (inputFrame.empty()) {
		cerr << "No input image" << endl;
		return inputFrame;
	}

	blobFromImage(inputFrame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
	this->model.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	this->model.forward(outs, this->names);

	// Remove the bounding boxes with low confidence
	postprocess(inputFrame, outs);

	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = this->model.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for compute the image : %.2f ms", t);
	cout << '\r' << label << flush;

	return inputFrame;
}

void YoloObjectDetector::setClasses(string pathToYoloFiles) {
	string classesFile = pathToYoloFiles + "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);
}

vector<String> YoloObjectDetector::setOutputsNames() {
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = this->model.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = this->model.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i) {
			names[i] = layersNames[outLayers[i] - 1];
		}
	}
	return names;
}

void YoloObjectDetector::postprocess(Mat& frame, const vector<Mat>& outs) {
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > this->threshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->threshold, 0.4, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPredictions(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void YoloObjectDetector::drawPredictions(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 255, 255), 1);

	//Get the label for the class name and its confidence
	string conf_label = format("%.2f", conf);
	string label = "";
	if (!classes.empty())
	{
		label = classes[classId] + ":" + conf_label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
}

YoloObjectDetector::~YoloObjectDetector() {}