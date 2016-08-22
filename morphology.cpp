/*
形态学操作
@author NvRom
*/
#include "morphology.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Point getNormalAnchor(cv::Size size , cv::Point anchor){
	if (anchor.x == -1){
		anchor.x = size.width / 2;
	}if (anchor.y == -1){
		anchor.y = size.height /2;
	}
	CV_Assert(anchor.inside(cv::Rect(0 , 0 , size.width , size.height)));
	return anchor;
}

bool SEMatchSrc(cv::Mat src , int op , int rowIndex , int colIndex , cv::Mat kernel , cv::Point anchor){
	CV_Assert(anchor.inside(cv::Rect(0 , 0 , kernel.cols , kernel.rows)));
	if (op == EROSION){
		for (int i = 0 ; i < kernel.rows ; i ++){
			for (int j = 0 ; j < kernel.cols ; j ++){
				if((kernel.at<uchar>(i , j) == 1 && src.at<uchar>(rowIndex + i - anchor.x , colIndex + j - anchor.y) == 255)
					|| kernel.at<uchar>(i , j) == 0){
						continue;
				}else
					return false;
			}
		}
		return true;
	}
	if (op == DILATION){
		for (int i = 0 ; i < kernel.rows ; i ++){
			for (int j = 0 ; j < kernel.cols ; j ++){
				if((kernel.at<uchar>(i , j) == 0) || (rowIndex + i - anchor.x  < 0) || (colIndex + j - anchor.y < 0 ))
					continue;
				else if ((kernel.at<uchar>(i , j) == 1 && src.at<uchar>(rowIndex + i - anchor.x , colIndex + j - anchor.y) == 255)){
					return true;
				}
			}
		}
		return false;
	}
}

cv::Mat getSE(int shape , cv::Size size , cv::Point anchor){
	int ellopse_x , ellopse_y;
	double ellopse_r = 0;//椭圆形的圆心
	int i ,j;//循环计数，
	CV_Assert(shape == MOR_RECT || shape == MOR_CROSS || shape == MOR_ELLOPSE);
	if (size == cv::Size(1,1)){
		shape = MOR_RECT;
	}
	if (shape == MOR_ELLOPSE){
		ellopse_x = size.width / 2;
		ellopse_y = size.height / 2;
		ellopse_r = ellopse_y ? static_cast<double>(1 / (ellopse_y * ellopse_y)) :0;
	}
	anchor = getNormalAnchor(size , anchor);
	cv::Mat elem(size , CV_8U);
	for (i  = 0 ; i < size.height ; i ++){
		//j0,j1表示从外部进入该shape以及从shape出去时，x轴坐标的值，两个可确定shape的边界
		//j0表示该下标为1
		int j0 = 0 , j1 = 0;
		uchar *ptr = elem.ptr(i);

		if(shape == MOR_RECT || (shape == MOR_CROSS && i == anchor.y))//矩形区域计算
			j1 = size.width;
		else if (shape = MOR_CROSS){//十字区域计算
			j0 = anchor.x;
			j1 = j0 + 1;
		}else{//椭圆形区域计算
			int dy = std::abs(ellopse_y - i);
			int dx = static_cast<int>(ellopse_x * std::sqrt((ellopse_y * ellopse_y - dy * dy)*ellopse_r));
			j0 = std::max(0 , ellopse_x - dx);
			j1 = std::min(size.width , ellopse_x + size.width);
		}
		//赋值
		for (j = 0 ; j < j0 ; j ++){
			ptr[j] = 0;
		}
		for (; j < j1 ; j ++){
			ptr[j] = 1;
		}
		for (; j < size.width ; j ++){
			ptr[j] = 0;
		}
	}
	return elem;

}

void erosion(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor){
	CV_Assert(src.size() == dst.size());
	if (kernel.empty()){
		kernel = getSE(MOR_RECT , cv::Size(3,3) , anchor);
	}
	cv::Size size = kernel.size();
	if (anchor == cv::Point(-1,-1)){
		anchor = getNormalAnchor(size ,anchor);
	}
	//腐蚀操作，图像结果会收缩
	int topMargin = anchor.y;
	int downMargin = size.height - anchor.y - 1;
	int leftMargin = anchor.x;
	int rightMargin = size.width - anchor.x - 1;

	for (int i = topMargin ; i < src.rows - downMargin ; i ++){
		for (int j = leftMargin ; j < src.cols - rightMargin ; j ++){
			if (SEMatchSrc(src , EROSION , i , j , kernel , anchor)){
				dst.at<uchar>(i,j) = 255;//匹配上，置255（白色）
			}else{
				dst.at<uchar>(i,j) = 0;//未匹配上，置0（黑色）
			}
		}
	}
	//边缘配色
	for (int i = 0 ; i < src.rows ; i ++){
		for (int j = 0 ; j < src.cols ;j ++){
			if (i < topMargin || i >= src.rows - downMargin){
				dst.at<uchar>(i,j) = src.at<uchar>(i,j);
			}else if (j < leftMargin || j >= src.cols - rightMargin){
				dst.at<uchar>(i,j) = src.at<uchar>(i,j);
			}
		}
	}

}

void dilation(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor){
	CV_Assert(src.size() == dst.size());
	if (kernel.empty()){
		kernel = getSE(MOR_RECT , cv::Size(3,3) , anchor);
	}
	cv::Size size = kernel.size();
	if (anchor == cv::Point(-1,-1)){
		anchor = getNormalAnchor(size ,anchor);
	}
	//膨胀操作，图像结果会扩张.margin的大小刚好跟erosion相反
	int downMargin = anchor.y;
	int topMargin = size.height - anchor.y - 1;
	int rightMargin = anchor.x;
	int leftMargin = size.width - anchor.x - 1;
	for (int i = 0 ; i < src.rows ; i ++){
		for (int j = 0 ; j < src.cols ; j ++){
			if (SEMatchSrc(src , DILATION , i , j , kernel , anchor)){
				dst.at<uchar>(i,j) = 255;//匹配上，置255（白色）
			}else{
				dst.at<uchar>(i,j) = 0;//未匹配上，置0（黑色）
			}
		}
	}
}

void opening(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor){
	erosion(src,dst,kernel,anchor);
	cv::Mat temp = dst.clone();
	dilation(temp,dst,kernel,anchor);
}

void closing(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor){
	dilation(src,dst,kernel,anchor);
	cv::Mat temp = dst.clone();
	erosion(temp,dst,kernel,anchor);
}