#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <math.h>
//对图像进行二值化操作,图像为灰度图像
//@author NvRom

//二值化操作的种类
enum OperationType
{
	MeanThreshold , PTileThreshold , MinimumThreshold , 
	IntermodesThreshold , IterativeBestThreshold , MomentPreservingThreshold
};

//显示直方图图像
cv::Mat getHistImg(cv::MatND hist);

//采用灰度平局值值法计算阈值
int getMeanThreshold(cv::MatND &hist);
//采用百分比阈值法计算阈值
int GetPTileThreshold(cv::MatND &hist , int tile = 50);
//基于谷底最小值的阈值
int GetMinimumThreshold(cv::MatND &hist);
//基于双峰平均值的阈值
int GetIntermodesThreshold(cv::MatND &hist);
//迭代最佳阈值
int GetIterativeBestThreshold(cv::MatND &hist);
//力矩保持法
int GetMomentPreservingThreshold(cv::MatND &hist , int &lowGrayValue , int  &highGrayValue);

//二值化操作函数，输入分别为：操作类型，灰度图图像 , 二值化的灰度值，默认为0，255
cv::Mat getBinaryImage(const int _OperationType , cv::Mat binary_src , int lowGrayValue = 0 , int  highGrayValue = 255){
	//第一步先对输入图像进行直方图计算,分别为通道数、层级、范围
	const int channels[1]={0};
	const int histSize[1]={256};
	float hranges[2]={0,255};
	const float* ranges[1]={hranges};
	cv::MatND hist;
	//函数原型见源码
	cv::calcHist(&binary_src,1,channels,cv::Mat(),hist,1,histSize,ranges);
	//二值化de阈值
	int threshold;
	int _row = binary_src.rows;
	int _col = binary_src.cols;
	cv::Mat histImg = getHistImg(hist);
	cv::imshow("直方图" , histImg);
	//根据不同操作分类
	switch (_OperationType)
	{
	case 0:
		threshold = getMeanThreshold(hist);
		break;
	case 1:
		threshold = GetPTileThreshold(hist);
		break;
	case 2:
		threshold = GetMinimumThreshold(hist);
		break;
	case 3:
		threshold = GetIntermodesThreshold(hist);
		break;
	case 4:
		threshold = GetIterativeBestThreshold(hist);
		break;
	case 5:
		threshold = GetMomentPreservingThreshold(hist , lowGrayValue , highGrayValue);
		break;
	default:
		break;
	}
//二值化
	for (int i = 0 ; i < _row ; i ++){
		uchar *data = binary_src.ptr<uchar>(i);
		for (int j = 0 ; j <_col ; j ++){
			if (data[j] <= threshold){
				data[j] = lowGrayValue;
			}else{
				data[j] = highGrayValue;
			}
		}
	}
	return binary_src;
}

//直方图图像
cv::Mat getHistImg(cv::MatND hist){
	double maxValue = 0 , minValue = 0;
	//找到直方图中最大值和最小值,minMaxLoc输入为单通道且返回最大/小值，而minMaxIdx输入是Inputarray，且返回下标
	cv::minMaxLoc(hist , &minValue , &maxValue , 0 , 0);
	//设置图像缩放比，为size的90%
	int histSize = hist.rows;
	double hpt = histSize * 0.9;
	cv::Mat _histImg(histSize , histSize , CV_8U  , cv::Scalar(255));
	//绘制zhifangtu
	for (int i = 0 ; i < histSize ; i ++){
		//如果写成int，则_value值会很大很大
		float _value = hist.at<float>(i);
		int imgLoc = static_cast<int>(_value * hpt / maxValue);
		cv::line(_histImg , cv::Point(i , histSize) , cv::Point(i , histSize - imgLoc) , cv::Scalar::all(0));
	}
	return _histImg;
}

//采用灰度平局值值法计算阈值
int getMeanThreshold(cv::MatND &hist){
	float sum = 0 ; 
	float amount = 0 ;
	for (int i = 0 ;i < 256 ; i ++){
		amount+=hist.at<float>(i);
		sum = sum + i * hist.at<float>(i);
	}
	return static_cast<int>(sum / amount);
}

//采用百分比阈值计算阈值
// Doyle于1962年提出P-Tile (即P分位数法)
int GetPTileThreshold(cv::MatND &hist , int tile){
	float sum = 0;
	float amount = 0;
	for (int i = 0 ; i < 256 ; i ++){
		amount+=hist.at<float>(i);
	}
	for(int i = 0 ; i < 256 ; i ++){
		sum += hist.at<float>(i);
		if(sum >= amount * tile / 100)
			return i;
	}
	return -1;
}

//基于谷底最小值的阈值
//此方法实用于具有明显双峰直方图的图像，其寻找双峰的谷底作为阈值，
//但是该方法不一定能获得阈值，对于那些具有平坦的直方图或单峰图像，该方法不合适。
//检测直方图是否为双峰
bool IsDimodal(double _hist[]){
	int count = 0;//双峰计数
	//注意i的下标1和255
	for (int i = 1 ; i < 255 ; i ++){
		if (_hist[i] > _hist[i - 1] && _hist[i] > _hist[i + 1]){
				count ++;
				if (count > 2)
					return false;
		}
	}
	if (count == 2)
		return true;
	else
		return false;
}
int GetMinimumThreshold(cv::MatND &hist){
	//会破坏数据，先得到一份拷贝
	double _hist[256];
	for (int i = 0 ; i < 256 ; i ++ ){
		_hist[i] = hist.at<float>(i);
	}
	int count = 0;//计数器
	//只有在有双峰的情况下才可以计算,没有双峰时，通过不断平滑直方图
	while (IsDimodal(_hist) == false){
		_hist[0] = (_hist[0] + _hist[0] + _hist[1])/3;
		for (int i = 1 ; i < 255 ;i ++){
			_hist[i] = (_hist[i - 1] + _hist[i] + _hist[i + 1]) / 3;
		}
		_hist[255] = (_hist[255] + _hist[255] + _hist[254]) / 3;
		//hist = cp_hist.clone();
		count ++;
		//若迭代超过1000次，则认为找不到双峰
		if (count > 1000)
			return -1;
	}
	//阈值为双峰之间的最小值.先找到一个峰，然后在判断谷底
	bool peakIsFound = false;
	for (int i = 1 ; i < 255 ; i ++){
		if (_hist[i] > _hist[i - 1] && _hist[i] > _hist[i + 1]){
			peakIsFound = true;
		}
		if (_hist[i] < _hist[i - 1] && _hist[i] < _hist[i + 1]){
				return i;
		}
	}
	return -1;
}

//基于双峰平均值的阈值
//该算法和基于谷底最小值的阈值方法类似，
//只是最后一步不是取得双峰之间的谷底值，而是取双峰的平均值作为阈值
int GetIntermodesThreshold(cv::MatND &hist){
	//会破坏数据，先得到一份拷贝
	double _hist[256];
	for (int i = 0 ; i < 256 ; i ++ ){
		_hist[i] = hist.at<float>(i);
	}
	int count = 0;//迭代次数计数器
	//只有在有双峰的情况下才可以计算,没有双峰时，通过不断平滑直方图
	while (IsDimodal(_hist) == false){
		_hist[0] = (_hist[0] + _hist[0] + _hist[1])/3;
		for (int i = 1 ; i < 255 ;i ++){
			_hist[i] = (_hist[i - 1] + _hist[i] + _hist[i + 1]) / 3;
		}
		_hist[255] = (_hist[255] + _hist[255] + _hist[254]) / 3;
		//hist = cp_hist.clone();
		count ++;
		//若迭代超过1000次，则认为找不到双峰
		if (count > 1000)
			return -1;
	}
	//与上一个算法的不同，在这里找两个峰点的中间值
	int peakPoint[2];
	for (int i = 1 , index = 0; i < 255 ; i ++){
		if (_hist[i] > _hist[i-1] && _hist[i] > _hist[i+1]){
			peakPoint[index ++] = i;
		}
	}
	return (peakPoint[0] + peakPoint[1]) / 2;
}

//迭代最佳阈值
//先假定一个阈值，然后计算在该阈值下的前景和背景的中心值，
//当前景和背景中心值得平均值和假定的阈值相同时，则迭代中止，并以此值为阈值进行二值化。
int GetIterativeBestThreshold(cv::MatND &hist){
	//中间值设为0,新的中间值设为128
	int _threshold_mid = 0;
	int _threshold_new = 128;

	float sum = 0 ; 
	float amount = 0 ;
	int count = 0;//循环迭代次数计数器
	while (_threshold_mid != _threshold_new){
		_threshold_mid = _threshold_new;
		//计算以_threshold_new为分界线，两边的灰度平均值,
		float sum = 0 ; 
		float amount = 0 ;
		for (int i = 0 ;i <= _threshold_new ; i ++){
			amount=hist.at<float>(i) + amount;
			sum = i * hist.at<float>(i) + sum;
		}
		//图像前景平均灰度值
		int a1 = static_cast<int>(sum / amount);
		sum = 0 ; 
		amount = 0 ;
		for (int i = _threshold_mid + 1 ;i < 256 ; i ++){
			amount+=hist.at<float>(i);
		}
		for (int i = _threshold_mid + 1 ; i < 256 ; i ++){
			sum += i * hist.at<float>(i) / amount;
		}
		//图像后景平均灰度值
		int a2 = static_cast<int>(sum);
		_threshold_new = static_cast<int>((a1+a2)/2);
		if (++ count > 1000){
			return -1;
		}
	}
	return _threshold_mid;
}

//力矩保持法
//该算法通过选择恰当的阈值从而使得二值后的图像和原始的灰度图像具有三个相同的初始力矩值。
//论文详见：http://www.sciencedirect.com/science/article/pii/0734189X85901331
int GetMomentPreservingThreshold(cv::MatND &hist , int &lowGrayValue , int  &highGrayValue){
	//存储论文中的i阶之和，amount表示像素总数
	double sum_1 = 0 , sum_2 = 0 , sum_3 = 0 , amount = 0;
	double sum_0 = 1;
	int threshold_index = -1;//返回的阈值
	//avec表示前i级灰度值占全部的百分比
	double avec[256];
	int sum = 0;
	for (int i = 0 ; i < 256 ; i ++){
		amount += hist.at<float>(i);
	}
	for (int i = 0 ; i < 256 ; i ++){
		sum += static_cast<int>(hist.at<float>(i));
		avec[i] = sum / amount;
		double  _count = static_cast<double>(hist.at<float>(i) / amount);
		sum_1 += _count * i;
		sum_2 += _count * i * i;
		sum_3 += _count * i * i * i;
	}
	//根据论文所给方法，解方程。下标按文中所示
	double Cd = sum_0 * sum_2 - sum_1 * sum_1;
	double C0 = (sum_1 * sum_3 - sum_2 * sum_2) / Cd;
	double C1 = (sum_2 * sum_1 - sum_0 * sum_3) / Cd;
	//Z0和Z1求值
	lowGrayValue = static_cast<int>(0.5 * (-C1 - sqrt(C1 * C1 - 4 * C0)));
	highGrayValue = static_cast<int>(0.5 * (-C1 + sqrt(C1 * C1 - 4 * C0)));
	//然后求P0和P1
	double Pd = highGrayValue - lowGrayValue;
	double P0 = (highGrayValue - sum_1) / Pd;
	//知道了P0后，可求得最靠近P0位置的huiduzhi
	float minValue = 0.25;
	for (int i = 0 ; i < 256 ; i ++){
		if (abs(avec[i] - P0) < minValue){
			minValue = abs(avec[i] - P0);
			threshold_index = i;
		}
	}
	return threshold_index;
}