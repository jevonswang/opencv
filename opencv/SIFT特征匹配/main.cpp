#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace cv;
using namespace std;

int main(){

	string filename1,filename2;
	cout << "please input the name of the first image: ";
	cin >> filename1;
	cout << "please input the name of the second image: ";
	cin >> filename2;

	Mat input1 = imread(filename1);
	Mat input2 = imread(filename2);

	//reduce the size of image to avoid the memory leaks.
	Size size = Size(input1.cols*0.8, input1.rows*0.8);
	Mat img1 = Mat(size, CV_32S);
	resize(input1, img1, size);
	Mat img2 = Mat(size, CV_32S);
	resize(input2, img2, size);


	initModule_nonfree();//initialize the module
	
	//get the feature points
	SiftFeatureDetector *detector = new SiftFeatureDetector(0.01, 3,0.1,6,1.6);
	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	//compute the feature descriptors
	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");
	Mat descriptor1, descriptor2;
	descriptor_extractor->compute(img1, keypoints1, descriptor1);
	descriptor_extractor->compute(img2, keypoints2, descriptor2);

	//match the features of the two pictures
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> matches;
	descriptor_matcher->match(descriptor1, descriptor2, matches);

	Mat img_matches;
	drawMatches(img1,keypoints1,img2,keypoints2,matches,img_matches);
	imwrite("matches.jpg",img_matches);

	//draw the featured resulted.
	Mat sift_result1,sift_result2;
	drawKeypoints(img1,keypoints1,sift_result1);
	drawKeypoints(img2,keypoints2,sift_result2);

	//add text to the image.
	int count = matches.size();
	stringstream ss;
	string str;
	ss<<count;
	ss>>str;
	string text = str+" match points found.";
	
	CvMat cvMat1 = sift_result1;
	CvMat cvMat2 = sift_result2;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 2, 2, 1, 4, 8);
	cvPutText(&cvMat1, text.c_str(), cvPoint(cvMat1.width - 1000, cvMat1.height - 50), &font, CV_RGB(255, 0, 0));
	cvPutText(&cvMat2, text.c_str(), cvPoint(cvMat2.width - 1000, cvMat2.height - 50), &font, CV_RGB(255, 0, 0));

	//read into the files
	imwrite(filename1+"_result.jpg", sift_result1);
	imwrite(filename2+"_result.jpg", sift_result2);

	waitKey(0);
	return 0;
}












