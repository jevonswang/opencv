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

    /** 打开输入视频文件 */
    cv::VideoCapture vc;
    vc.open(file_in);
    
    if ( vc.isOpened() )
    {
        /** 打开输出视频文件 */
        VideoWriter vw;
        vw.open(file_out, // 输出视频文件名
                (int)vc.get( CV_CAP_PROP_FOURCC ), 
                (double)vc.get( CV_CAP_PROP_FPS ), 
                cv::Size( (int)vc.get( CV_CAP_PROP_FRAME_WIDTH ), (int)vc.get( CV_CAP_PROP_FRAME_HEIGHT ) ), // 视频大小
                false ); // 是否输出彩色视频

        /** 如果成功打开输出视频文件 */
        if ( vw.isOpened() )
        {
            while ( true )
            {
                /** 读取当前视频帧 */
                cv::Mat in;
				vc >> in;

                /** 若视频读取完毕，跳出循环 */
                if ( in.empty() )
                {
                    break;
                }

				IplImage s = in;//原图
				IplImage *color = &s;

				cvShowImage(file_in.c_str(), color );//显示原图

				char c = cvWaitKey(30);//等待

				//转为灰度图
				IplImage *gray = cvCreateImage(cvGetSize(color),  8,1);
				cvCvtColor(color,gray,CV_BGR2GRAY);

				//转为二值图
				IplImage *binary = cvCreateImage(cvGetSize(gray),  8,1);//二值图
				cvThreshold(gray,binary,threshold,255,CV_THRESH_BINARY);


				//插入文字
				CvFont font;
				double hscale = 0.5;
				double vscale = 0.5;
				int linewidth = 1;
				cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX |  CV_FONT_ITALIC,hscale,vscale,0,linewidth); 
				CvScalar textColor =cvScalar(255,255,255);
				CvPoint textPos =cvPoint(0,15);
				cvPutText(binary,"Wang Zhefeng 3110000026", textPos, &font,textColor);

				//显示输出图像
				cvShowImage(file_out.c_str(), binary );

                /** 将视频写入文件 */
				Mat out(binary);
                vw << out;
            }
        }
    }
    /** 手动释放视频捕获资源 */
    vc.release();
}