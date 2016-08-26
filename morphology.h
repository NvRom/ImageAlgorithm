/*
形态学操作
@author NvRom
*/
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//分别表示为矩形，十字，椭圆
enum morShape{
	MOR_RECT , MOR_CROSS , MOR_ELLOPSE
};

//形态学基本操作
enum  morOp{
	EROSION , DILATION
};

/*
得到指定大小和规模的结构体
@param shape:结构体的形状，有矩形、十字形、椭圆形。若没指定形状，默认为矩形
@param size:结构体的大小，若没指定大小，默认为3*3
@param anchor:结构体的原点位置，默认anchor为(-1,-1),表示结构体的中心
*/
cv::Mat getSE(int shape , cv::Size size = cv::Size(3,3) , cv::Point anchor = cv::Point(-1,-1));

/*
若结构体原点是默认，则返回中心点
此函数被getSE函数调用
@param size:结构体的大小
@param anchor:结构体的原点位置
*/
cv::Point getNormalAnchor(cv::Size size , cv::Point anchor);

/*
判断结构体与原图像上是否匹配，匹配返回true，否则返回false
@param src：输入图像，二值化的图像
@param op：操作类型，有腐蚀（erosion）和膨胀（dilation）两种基本操作
@param (rowIndex,colIndex)：kernel原点所在图像的位置
@param kernel：结构体（SE），可以通过getSE函数得到，提供以下几种kernel，分别为：MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor：结构体的原点，默认anchor为（-1，-1）表示原点位于结构体的中心位置
*/
bool SEMatchSrc(cv::Mat src , int op , int rowIndex , int colIndex , cv::Mat kernel , cv::Point anchor);

/*
图像的腐蚀操作
@param src：输入图像，要求是灰度图
@param dst：输出图像，大小和类型与src一样
@param kernel：结构体（SE），可以通过getSE函数得到，提供以下几种kernel，分别为：MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor：结构体的原点，默认anchor为（-1，-1）表示原点位于结构体的中心位置
*/
void erosion(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
图像的膨胀操作
@param src：输入图像，要求是灰度图
@param dst：输出图像，大小和类型与src一样
@param kernel：结构体（SE），可以通过getSE函数得到，提供以下几种kernel，分别为：MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor：结构体的原点，默认anchor为（-1，-1）表示原点位于结构体的中心位置
*/
void dilation(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
图像开操作,先对图像进行腐蚀，然后再膨胀
@param src：输入图像，要求是灰度图
@param dst：输出图像，大小和类型与src一样
@param kernel：结构体（SE），可以通过getSE函数得到，提供以下几种kernel，分别为：MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor：结构体的原点，默认anchor为（-1，-1）表示原点位于结构体的中心位置
*/
void opening(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
图像关闭操作，先对图像进行膨胀，然后再腐蚀
参数同上
*/
void closing(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
逻辑与操作，求src与mask公共交集的部分，并返回src
@param src 输入图像
@param mask 模板，src以mask为目标进行恢复
*/
void logicAnd(cv::Mat &src , cv::Mat mask);

/*
判断两幅图像是否完全吻合
@param img 原图像，待比较的图像
@param mask 目标图像
*/
bool matchMask(cv::Mat img , cv::Mat mask);

/*
利用dilation操作对图像恢
@param mask：输入图像，要求是灰度图,同时限制marker无限膨胀，恢复成与mask局部一样的图像
@param marker：待恢复的图像，从mask做opening得到
@param kernel：结构体（SE），可以通过getSE函数得到，提供以下几种kernel，分别为：MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor：结构体的原点，默认anchor为（-1，-1）表示原点位于结构体的中心位置
*/
cv::Mat geodesticDilation(cv::Mat mask , cv::Mat marker , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));