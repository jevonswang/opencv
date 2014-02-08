#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <string>

using namespace cv;
using namespace std;

void main(int argc,char *argv[])
{
	string file_in,file_out;
	int threshold;

	if(argc!=1){
		file_in = argv[1];
		string str = argv[2];
		threshold = atoi(str.c_str());;
		file_out = argv[3];
	}else{
		cout<<"Please input the input file:";
		cin>>file_in;
		cout<<"Please input the threshold:";
		cin>>threshold;
		cout<<"Please input the output file:";
		cin>>file_out;
	}

    /** ��������Ƶ�ļ� */
    cv::VideoCapture vc;
    vc.open(file_in);
    
    if ( vc.isOpened() )
    {
        /** �������Ƶ�ļ� */
        VideoWriter vw;
        vw.open(file_out, // �����Ƶ�ļ���
                (int)vc.get( CV_CAP_PROP_FOURCC ), 
                (double)vc.get( CV_CAP_PROP_FPS ), 
                cv::Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ), (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // ��Ƶ��С
                false ); // �Ƿ������ɫ��Ƶ

        /** ����ɹ��������Ƶ�ļ� */
        if ( vw.isOpened() )
        {
            while ( true )
            {
                /** ��ȡ��ǰ��Ƶ֡ */
                cv::Mat in;
				vc >> in;

                /** ����Ƶ��ȡ��ϣ�����ѭ�� */
                if ( in.empty() )
                {
                    break;
                }

				IplImage s = in;//ԭͼ
				IplImage *color = &s;

				cvShowImage(file_in.c_str(), color );//��ʾԭͼ

				char c = cvWaitKey(30);//�ȴ�

				//תΪ�Ҷ�ͼ
				IplImage *gray = cvCreateImage(cvGetSize(color),  8,1);
				cvCvtColor(color,gray,CV_BGR2GRAY);

				//תΪ��ֵͼ
				IplImage *binary = cvCreateImage(cvGetSize(gray),  8,1);//��ֵͼ
				cvThreshold(gray,binary,threshold,255,CV_THRESH_BINARY);


				//��������
				CvFont font;
				double hscale = 0.5;
				double vscale = 0.5;
				int linewidth = 1;
				cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX |  CV_FONT_ITALIC,hscale,vscale,0,linewidth); 
				CvScalar textColor =cvScalar(255,255,255);
				CvPoint textPos =cvPoint(0,15);
				cvPutText(binary,"Wang Zhefeng 3110000026", textPos, &font,textColor);

				//��ʾ���ͼ��
				cvShowImage(file_out.c_str(), binary );

                /** ����Ƶд���ļ� */
				Mat out(binary);
                vw << out;
            }
        }
    }
    /** �ֶ��ͷ���Ƶ������Դ */
    vc.release();
}