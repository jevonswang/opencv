#include <opencv2\opencv.hpp>
#include<cv.h>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

int findThreshold(IplImage *imgBasicGlobalThreshold){	//  基本全局阈值法
                                           
	int pg[256],i;
	int  t,t1,t2,k1,k2;
    double u,u1,u2;    
	int start = 0;
	int end =256;

    for (i=0;i<256;i++) pg[i]=0;

    for (i=0;i<imgBasicGlobalThreshold->imageSize;i++)// 直方图统计
		pg[(unsigned char)(imgBasicGlobalThreshold->imageData[i])]++;
  
    t=0;     
    u=0;
    for (i=start;i<end;i++){
		t+=pg[i];        
        u+=i*pg[i];
    }
    
	k2=(int) (u/t);        //  计算此范围灰度的平均值    
    do{
		k1=k2;
        t1=0;    
        u1=0;
        for (i=start;i<=k1;i++) {             //  计算低灰度组的累加和
			t1+=pg[i];    
            u1+=i*pg[i];
        }
        t2=t-t1;
        u2=u-u1;

        if (t1) 
			u1=u1/t1;                     //  计算低灰度组的平均值
        else 
			u1=0;
        
		if (t2) 
			u2=u2/t2;                     //  计算高灰度组的平均值
        else 
			u2=0;
        
		k2=(int) ((u1+u2)/2);                 //  得到新的阈值估计值
     
	}while(k1!=k2);                           //  数据未稳定，继续
     
	return k1;                              //  返回阈值
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

	/*二值化*/
	IplImage *gray = cvCreateImage(cvGetSize(color),8,1);
	cvCvtColor(color,gray,CV_BGR2GRAY);
	IplImage *binary = cvCreateImage(cvGetSize(gray),8,1);
	int threshold = findThreshold(gray);
	cvThreshold(gray,binary,threshold,255,CV_THRESH_BINARY);

    //cvSmooth(binary,binary);
	cvShowImage("Binary",binary);

	/*形态学操作*/
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
	
	
	int ovalCount = 0;//椭圆的个数
	int circleCount = 0;//圆的个数
    //绘制所有轮廓并用椭圆拟合
    for (;cont;cont = cont ->h_next)
    {
        int i;
        int count= cont->total;//轮廓个数
        
		CvPoint center;
        CvSize size;

        /*个数必须大于6，这是cvFitEllipse_32f的要求*/
        if (count<6)
        {
            continue;
        }
    
        //分配内存给点集
        PointArray = (CvPoint *)malloc(count*sizeof(CvPoint));
        PointArray2D32f = (CvPoint2D32f*)malloc(count*sizeof(CvPoint2D32f));
    
        //分配内存给椭圆数据
        box = (CvBox2D32f *)malloc(sizeof(CvBox2D32f));

        //得到点集
        cvCvtSeqToArray(cont,PointArray,CV_WHOLE_SEQ);
    
        //将CvPoint点集转化为CvBox2D32f集合
        for (i=0;i<count;i++)
        {
            PointArray2D32f[i].x=(float)PointArray[i].x;
            PointArray2D32f[i].y=(float)PointArray[i].y;
        }

        //拟合当前轮廓
        cvFitEllipse(PointArray2D32f,count,box);

        //绘制当前轮廓
        //cvDrawContours(color,cont,CV_RGB(0,255,0),CV_RGB(0,255,0),0,1,8,cvPoint(0,0));

        //将椭圆数据从浮点转化为整数表示
        center.x = cvRound(box->center.x);
        center.y = cvRound(box->center.y);
        size.width = cvRound(box->size.width*0.5);
        size.height = cvRound(box->size.height*0.5);
      

		int minWidth = 35;
		int minHeight = 35;
		int maxWidth = 200;
		int maxHeight = 200;

        //画椭圆
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


