// OpenCVlab4.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void readCsv(const string& filename, vector<Mat>& images,vector<Mat> &cimages, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		cout<<"未找到该csv文件！"<<endl;
		exit(0);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty()) {
		    cimages.push_back(imread(path));
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {

	string csvFile="csv.txt";		//对人分类的csv
	string csv[5]={"csv0.txt","csv1.txt","csv2.txt","csv3.txt","csv4.txt"};		//每个人的不同照片分类的csv
	string Name[5]={"Brad Pitt","George Clooney","YuanYe","Taylor Swift","TA"};		//人的姓名
	string haarFile="haarcascade_frontalface_alt.xml";		//用于haar特征的xml

	vector<Mat> images;			//用于模型预测的灰度图像
	vector<Mat> cimages;		//彩色图像
	vector<int> labels;			//标签

	vector<Mat> subImages[5];	//子分类模型的灰度图像
	vector<Mat> subCimages[5];  //子分类模型的彩色图像
	vector<int> subLabels[5];   //子分类模型的标签

	CascadeClassifier haarCascade;
	haarCascade.load(haarFile);		//从文件中加载haar特征

	readCsv(csvFile,images,cimages,labels);		//读取对人分类的csv文件

	int imgWidth=images[0].cols;	
	int imgHeight=images[0].rows;

	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();	//创建Fisher人脸识别器
	model->train(images, labels);	//对人分类模型训练

	Ptr<FaceRecognizer> subModel[5];

	//对子分类模型进行Fisher人脸识别器创建，读取csv和模型训练，用于找出最像的图片
	for (int i = 0; i < 5; i++)
	{
		subModel[i]=createFisherFaceRecognizer();	
		readCsv(csv[i],subImages[i],subCimages[i],subLabels[i]);
		subModel[i]->train(subImages[i],subLabels[i]);
	}


	VideoCapture cap(0);

	if(!cap.isOpened()) {
		cerr << "无法检测到摄像头！" << endl;
		return -1;
	}

	vector< Rect_<int> > faces;
	bool find=false;
	int prediction;		//对人身份的预测
	int subPrediction;		//对最像图片的预测
	Mat frame;

	for(;;) {
		cap >> frame;	//从摄像头获取图像
		
		Mat original = frame.clone();
		Mat matchFrame=original(cv::Rect(0,0,100,100));		//获取贴缩放图像的框架

		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:

		if(!find){
			faces.clear();
			haarCascade.detectMultiScale(gray, faces);		//用haar特征检测人脸
		}

		for(int i = 0; i <faces.size(); i++) {

			Rect face_i = faces[i];

			if(face_i.width<200) continue;		//如果人脸太小则排除

			Mat face = gray(face_i);

			if(!find){
				Mat face_resized;
				cv::resize(face, face_resized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_CUBIC);	//对图像中的人脸进行缩放与模型大小一致
				// Now perform the prediction, see how easy that is:
				prediction = model->predict(face_resized);			//利用训练好的对人的身份进行预测
				subPrediction = subModel[prediction]->predict(face_resized);	//利用训练好的子模型对最像的照片进行预测
			}

			//将最像的图片叠加到视频图像的左上角
			Mat matchFace,matchFaceResize;
			matchFace=subCimages[prediction][subPrediction];
			cv::resize(matchFace, matchFaceResize, Size(100,100));
			matchFaceResize.copyTo(matchFrame);

			//绘制人脸方框
			rectangle(original, face_i, CV_RGB(0, 255,0), 1);

			//输出人的身份信息
			string box_text=Name[prediction];
			
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

			find = true;
			
		}

		//显示处理过的视频图像
		imshow("face_recognizer", original);


		char key = (char) waitKey(20);

		// 如果按下Esc则推出
		if(key == 27)
			break;
		
		//如果按下空格重新开始识别
		if(key == ' ') {
				find=false;
		}
	}
	return 0;
}