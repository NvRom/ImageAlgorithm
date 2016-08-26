/*
��̬ѧ����
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
				if((kernel.at<uchar>(i , j) == 0) || (rowIndex + i - anchor.x  < 0) || (colIndex + j - anchor.y < 0 )
					 || (rowIndex + i - anchor.x  >= src.rows) || (colIndex + j - anchor.y  >= src.cols ))
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
	double ellopse_r = 0;//��Բ�ε�Բ��
	int i ,j;//ѭ��������
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
		//j0,j1��ʾ���ⲿ�����shape�Լ���shape��ȥʱ��x�������ֵ��������ȷ��shape�ı߽�
		//j0��ʾ���±�Ϊ1
		int j0 = 0 , j1 = 0;
		uchar *ptr = elem.ptr(i);

		if(shape == MOR_RECT || (shape == MOR_CROSS && i == anchor.y))//�����������
			j1 = size.width;
		else if (shape = MOR_CROSS){//ʮ���������
			j0 = anchor.x;
			j1 = j0 + 1;
		}else{//��Բ���������
			int dy = std::abs(ellopse_y - i);
			int dx = static_cast<int>(ellopse_x * std::sqrt((ellopse_y * ellopse_y - dy * dy)*ellopse_r));
			j0 = std::max(0 , ellopse_x - dx);
			j1 = std::min(size.width , ellopse_x + size.width);
		}
		//��ֵ
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
	cv::Mat _dst(src.size() , CV_8U);
	if (kernel.empty()){
		kernel = getSE(MOR_RECT , cv::Size(3,3) , anchor);
	}
	cv::Size size = kernel.size();
	if (anchor == cv::Point(-1,-1)){
		anchor = getNormalAnchor(size ,anchor);
	}
	//��ʴ������ͼ����������
	int topMargin = anchor.y;
	int downMargin = size.height - anchor.y - 1;
	int leftMargin = anchor.x;
	int rightMargin = size.width - anchor.x - 1;

	for (int i = topMargin ; i < src.rows - downMargin ; i ++){
		for (int j = leftMargin ; j < src.cols - rightMargin ; j ++){
			if (SEMatchSrc(src , EROSION , i , j , kernel , anchor)){
				_dst.at<uchar>(i,j) = 255;//ƥ���ϣ���255����ɫ��
			}else{
				_dst.at<uchar>(i,j) = 0;//δƥ���ϣ���0����ɫ��
			}
		}
	}
	//��Ե��ɫ
	for (int i = 0 ; i < src.rows ; i ++){
		for (int j = 0 ; j < src.cols ;j ++){
			if (i < topMargin || i >= src.rows - downMargin){
				_dst.at<uchar>(i,j) = src.at<uchar>(i,j);
			}else if (j < leftMargin || j >= src.cols - rightMargin){
				_dst.at<uchar>(i,j) = src.at<uchar>(i,j);
			}
		}
	}
	dst = _dst.clone();
}

void dilation(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor){
	CV_Assert(src.size() == dst.size());
	cv::Mat _dst(src.size() , CV_8U);
	if (kernel.empty()){
		kernel = getSE(MOR_RECT , cv::Size(3,3) , anchor);
	}
	cv::Size size = kernel.size();
	if (anchor == cv::Point(-1,-1)){
		anchor = getNormalAnchor(size ,anchor);
	} 
	//���Ͳ��������se���з�ת
	cv::Mat _kernel = kernel.clone();
	for (int i = 0 ; i < kernel.cols ; i ++){
		for (int j = 0 ; j < kernel.rows ; j ++){
			CV_Assert(cv::Point(2*anchor.x - i ,2*anchor.y - j).inside(cv::Rect(0,0,kernel.cols,kernel.rows)));
			_kernel.at<uchar>(i , j) = kernel.at<uchar>(2*anchor.x - i ,2*anchor.y - j);
		}
	}
	//���Ͳ�����ͼ����������.margin�Ĵ�С�պø�erosion�෴
	int downMargin = anchor.y;
	int topMargin = size.height - anchor.y - 1;
	int rightMargin = anchor.x;
	int leftMargin = size.width - anchor.x - 1;
	for (int i = 0 ; i < src.rows ; i ++){
		for (int j = 0 ; j < src.cols ; j ++){
			if (SEMatchSrc(src , DILATION , i , j , _kernel , anchor)){
				_dst.at<uchar>(i,j) = 255;//ƥ���ϣ���255����ɫ��
			}else{
				_dst.at<uchar>(i,j) = 0;//δƥ���ϣ���0����ɫ��
			}
		}
	}
	dst = _dst.clone();
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

void logicAnd(cv::Mat &src , cv::Mat mask){
	CV_Assert(src.rows == mask.rows && src.cols == mask.cols);
	for (int i = 0 ; i < src.rows ;  i ++){
		uchar *ptrSrc = src.ptr(i);
		uchar *ptrMask = mask.ptr(i);
		for (int j = 0 ; j < src.cols ; j ++){
			if (ptrSrc[j] == 255){
				ptrSrc[j] = (ptrSrc[j] & ptrMask[j]);
			}
		}
	}
}

bool matchMask(cv::Mat img , cv::Mat mask){
	CV_Assert(img.rows == mask.rows && img.cols == mask.cols);
	for (int i = 0 ; i < img.rows ;  i ++){
		uchar *ptrSrc = img.ptr(i);
		uchar *ptrMask = mask.ptr(i);
		for (int j = 0 ; j < img.cols ; j ++){
			if (ptrSrc[j] != ptrMask[j]){
				return false;
			}
		}
	}
	return true;
}

cv::Mat geodesticDilation(cv::Mat mask , cv::Mat marker , cv::Mat kernel , cv::Point anchor){
	CV_Assert(marker.rows == mask.rows && marker.cols == mask.cols);
	cv::Mat openImg1(marker.size(),CV_8U);
	//openImg1��ʾ��i-1����openImg0��ʾ(i)
	do {
		openImg1 = marker.clone();
		dilation(openImg1 , marker , kernel , anchor);
		logicAnd(marker , mask);
	} while (!matchMask(marker , openImg1));
	return marker;
}