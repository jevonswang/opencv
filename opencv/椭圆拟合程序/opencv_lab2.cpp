#include <opencv2\opencv.hpp>
#include<cv.h>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

int findThreshold(IplImage *imgBasicGlobalThreshold){	//  ����ȫ����ֵ��
                                           
	int pg[256],i;
	int  t,t1,t2,k1,k2;
    double u,u1,u2;    
	int start = 0;
	int end =256;

    for (i=0;i<256;i++) pg[i]=0;

    for (i=0;i<imgBasicGlobalThreshold->imageSize;i++)// ֱ��ͼͳ��
		pg[(unsigned char)(imgBasicGlobalThreshold->imageData[i])]++;
  
    t=0;     
    u=0;
    for (i=start;i<end;i++){
		t+=pg[i];        
        u+=i*pg[i];
    }
    
	k2=(int) (u/t);        //  ����˷�Χ�Ҷȵ�ƽ��ֵ    
    do{
		k1=k2;
        t1=0;    
        u1=0;
        for (i=start;i<=k1;i++) {             //  ����ͻҶ�����ۼӺ�
			t1+=pg[i];    
            u1+=i*pg[i];
        }
        t2=t-t1;
        u2=u-u1;

        if (t1) 
			u1=u1/t1;                     //  ����ͻҶ����ƽ��ֵ
        else 
			u1=0;
        
		if (t2) 
			u2=u2/t2;                     //  ����߻Ҷ����ƽ��ֵ
        else 
			u2=0;
        
		k2=(int) ((u1+u2)/2);                 //  �õ��µ���ֵ����ֵ
     
	}while(k1!=k2);                           //  ����δ�ȶ�������
     
	return k1;                              //  ������ֵ
}

int main(){
	cout<<"Please input the filename: ";
	string filename;
	cin>>filename;

    Mat img = imread(filename);
	if(img.empty()){
        cout<<"error";
        return -1;
    }

	IplImage in = img;
	IplImage *color = &in;
	//cvShowImage("color",color);

	/*��ֵ��*/
	IplImage *gray = cvCreateImage(cvGetSize(color),8,1);
	cvCvtColor(color,gray,CV_BGR2GRAY);
	IplImage *binary = cvCreateImage(cvGetSize(gray),8,1);
	int threshold = findThreshold(gray);
	cvThreshold(gray,binary,threshold,255,CV_THRESH_BINARY);

    //cvSmooth(binary,binary);
	cvShowImage("Binary",binary);

	/*��̬ѧ����*/
	cvErode(binary,binary,NULL,1);
	cvDilate(binary,binary,NULL,1);
	cvShowImage("Morphological",binary);
	
	CvMemStorage *stor;
    CvSeq *cont;
    CvBox2D32f *box;
    CvPoint *PointArray;
    CvPoint2D32f *PointArray2D32f;

    stor = cvCreateMemStorage(0);
    cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),stor);

    cvFindContours(binary,stor,&cont,sizeof(CvContour),
        CV_RETR_LIST,CV_CHAIN_APPROX_NONE,cvPoint(0,0));
	
	
	int ovalCount = 0;//��Բ�ĸ���
	int circleCount = 0;//Բ�ĸ���
    //������������������Բ���
    for (;cont;cont = cont ->h_next)
    {
        int i;
        int count= cont->total;//��������
        
		CvPoint center;
        CvSize size;

        /*�����������6������cvFitEllipse_32f��Ҫ��*/
        if (count<6)
        {
            continue;
        }
    
        //�����ڴ���㼯
        PointArray = (CvPoint *)malloc(count*sizeof(CvPoint));
        PointArray2D32f = (CvPoint2D32f*)malloc(count*sizeof(CvPoint2D32f));
    
        //�����ڴ����Բ����
        box = (CvBox2D32f *)malloc(sizeof(CvBox2D32f));

        //�õ��㼯
        cvCvtSeqToArray(cont,PointArray,CV_WHOLE_SEQ);
    
        //��CvPoint�㼯ת��ΪCvBox2D32f����
        for (i=0;i<count;i++)
        {
            PointArray2D32f[i].x=(float)PointArray[i].x;
            PointArray2D32f[i].y=(float)PointArray[i].y;
        }

        //��ϵ�ǰ����
        cvFitEllipse(PointArray2D32f,count,box);

        //���Ƶ�ǰ����
        //cvDrawContours(color,cont,CV_RGB(0,255,0),CV_RGB(0,255,0),0,1,8,cvPoint(0,0));

        //����Բ���ݴӸ���ת��Ϊ������ʾ
        center.x = cvRound(box->center.x);
        center.y = cvRound(box->center.y);
        size.width = cvRound(box->size.width*0.5);
        size.height = cvRound(box->size.height*0.5);
      

		int minWidth = 35;
		int minHeight = 35;
		int maxWidth = 200;
		int maxHeight = 200;

        //����Բ
		if(size.width>minWidth && size.height>minHeight && size.width<maxWidth && size.height < maxHeight){
			cvEllipse(color,center,size,box->angle,0,360,CV_RGB(0,0,255),1,CV_AA,0);
			cout<<"Ellipse "<<++ovalCount<<": "<<endl;
			cout<<"   Center:("<<center.x<<","<<center.y<<")"<<endl;
			cout<<"   Long axis: "<<size.height<<endl;
			cout<<"   Short axis: "<<size.width<<endl;
		}
        free(PointArray);
        free(PointArray2D32f);
        free(box);
	}

	cvShowImage("Result",color);

	waitKey();

    return 0;
}


