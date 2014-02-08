// OpenCVlab4.cpp : �������̨Ӧ�ó������ڵ㡣
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
		cout<<"δ�ҵ���csv�ļ���"<<endl;
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

	string csvFile="csv.txt";		//���˷����csv
	string csv[5]={"csv0.txt","csv1.txt","csv2.txt","csv3.txt","csv4.txt"};		//ÿ���˵Ĳ�ͬ��Ƭ�����csv
	string Name[5]={"Brad Pitt","George Clooney","YuanYe","Taylor Swift","TA"};		//�˵�����
	string haarFile="haarcascade_frontalface_alt.xml";		//����haar������xml

	vector<Mat> images;			//����ģ��Ԥ��ĻҶ�ͼ��
	vector<Mat> cimages;		//��ɫͼ��
	vector<int> labels;			//��ǩ

	vector<Mat> subImages[5];	//�ӷ���ģ�͵ĻҶ�ͼ��
	vector<Mat> subCimages[5];  //�ӷ���ģ�͵Ĳ�ɫͼ��
	vector<int> subLabels[5];   //�ӷ���ģ�͵ı�ǩ

	CascadeClassifier haarCascade;
	haarCascade.load(haarFile);		//���ļ��м���haar����

	readCsv(csvFile,images,cimages,labels);		//��ȡ���˷����csv�ļ�

	int imgWidth=images[0].cols;	
	int imgHeight=images[0].rows;

	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();	//����Fisher����ʶ����
	model->train(images, labels);	//���˷���ģ��ѵ��

	Ptr<FaceRecognizer> subModel[5];

	//���ӷ���ģ�ͽ���Fisher����ʶ������������ȡcsv��ģ��ѵ���������ҳ������ͼƬ
	for (int i = 0; i < 5; i++)
	{
		subModel[i]=createFisherFaceRecognizer();	
		readCsv(csv[i],subImages[i],subCimages[i],subLabels[i]);
		subModel[i]->train(subImages[i],subLabels[i]);
	}


	VideoCapture cap(0);

	if(!cap.isOpened()) {
		cerr << "�޷���⵽����ͷ��" << endl;
		return -1;
	}

	vector< Rect_<int> > faces;
	bool find=false;
	int prediction;		//������ݵ�Ԥ��
	int subPrediction;		//������ͼƬ��Ԥ��
	Mat frame;

	for(;;) {
		cap >> frame;	//������ͷ��ȡͼ��
		
		Mat original = frame.clone();
		Mat matchFrame=original(cv::Rect(0,0,100,100));		//��ȡ������ͼ��Ŀ��

		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:

		if(!find){
			faces.clear();
			haarCascade.detectMultiScale(gray, faces);		//��haar�����������
		}

		for(int i = 0; i <faces.size(); i++) {

			Rect face_i = faces[i];

			if(face_i.width<200) continue;		//�������̫С���ų�

			Mat face = gray(face_i);

			if(!find){
				Mat face_resized;
				cv::resize(face, face_resized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_CUBIC);	//��ͼ���е���������������ģ�ʹ�Сһ��
				// Now perform the prediction, see how easy that is:
				prediction = model->predict(face_resized);			//����ѵ���õĶ��˵���ݽ���Ԥ��
				subPrediction = subModel[prediction]->predict(face_resized);	//����ѵ���õ���ģ�Ͷ��������Ƭ����Ԥ��
			}

			//�������ͼƬ���ӵ���Ƶͼ������Ͻ�
			Mat matchFace,matchFaceResize;
			matchFace=subCimages[prediction][subPrediction];
			cv::resize(matchFace, matchFaceResize, Size(100,100));
			matchFaceResize.copyTo(matchFrame);

			//������������
			rectangle(original, face_i, CV_RGB(0, 255,0), 1);

			//����˵������Ϣ
			string box_text=Name[prediction];
			
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

			find = true;
			
		}

		//��ʾ���������Ƶͼ��
		imshow("face_recognizer", original);


		char key = (char) waitKey(20);

		// �������Esc���Ƴ�
		if(key == 27)
			break;
		
		//������¿ո����¿�ʼʶ��
		if(key == ' ') {
				find=false;
		}
	}
	return 0;
}