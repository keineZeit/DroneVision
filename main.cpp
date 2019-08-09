#include <iostream>
#include "VideoStream.hpp"
#include "detector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void videoLoop(VideoStream &stream, Detector* detector);

int main() {

	VideoStream stream;
	HaarFaceDetector faceDetector("haarcascade_frontalface_alt2.xml", 0.7);
	YoloObjectDetector objDetector("yolov2-tiny", 0.5);

	videoLoop(stream, &objDetector);

	//system("pause");
	return 0;
}

void videoLoop(VideoStream &stream, Detector* detector) {
	Mat frame;
	Mat inferredFrame;

	stream.start(0);

	string windowName = "Detector";
	namedWindow(windowName, WINDOW_NORMAL);
	//resizeWindow(windowName, 640, 480);

	while (true) {
		frame = stream.read();
		if (frame.empty()) continue;

		inferredFrame = (*detector).predict(frame);

		imshow(windowName, inferredFrame);
		if (waitKey(30) >= 0) {
			stream.release();
			destroyWindow(windowName);
			break;
		}
	}
}
