#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <math.h>
//��ͼ����ж�ֵ������,ͼ��Ϊ�Ҷ�ͼ��
//@author NvRom

//��ֵ������������
enum OperationType
{
	MeanThreshold , PTileThreshold , MinimumThreshold , 
	IntermodesThreshold , IterativeBestThreshold , MomentPreservingThreshold
};

//��ʾֱ��ͼͼ��
cv::Mat getHistImg(cv::MatND hist);

//���ûҶ�ƽ��ֵֵ��������ֵ
int getMeanThreshold(cv::MatND &hist);
//���ðٷֱ���ֵ��������ֵ
int GetPTileThreshold(cv::MatND &hist , int tile = 50);
//���ڹȵ���Сֵ����ֵ
int GetMinimumThreshold(cv::MatND &hist);
//����˫��ƽ��ֵ����ֵ
int GetIntermodesThreshold(cv::MatND &hist);
//���������ֵ
int GetIterativeBestThreshold(cv::MatND &hist);
//���ر��ַ�
int GetMomentPreservingThreshold(cv::MatND &hist , int &lowGrayValue , int  &highGrayValue);

//��ֵ����������������ֱ�Ϊ���������ͣ��Ҷ�ͼͼ�� , ��ֵ���ĻҶ�ֵ��Ĭ��Ϊ0��255
cv::Mat getBinaryImage(const int _OperationType , cv::Mat binary_src , int lowGrayValue = 0 , int  highGrayValue = 255){
	//��һ���ȶ�����ͼ�����ֱ��ͼ����,�ֱ�Ϊͨ�������㼶����Χ
	const int channels[1]={0};
	const int histSize[1]={256};
	float hranges[2]={0,255};
	const float* ranges[1]={hranges};
	cv::MatND hist;
	//����ԭ�ͼ�Դ��
	cv::calcHist(&binary_src,1,channels,cv::Mat(),hist,1,histSize,ranges);
	//��ֵ��de��ֵ
	int threshold;
	int _row = binary_src.rows;
	int _col = binary_src.cols;
	cv::Mat histImg = getHistImg(hist);
	cv::imshow("ֱ��ͼ" , histImg);
	//���ݲ�ͬ��������
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
//��ֵ��
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

//ֱ��ͼͼ��
cv::Mat getHistImg(cv::MatND hist){
	double maxValue = 0 , minValue = 0;
	//�ҵ�ֱ��ͼ�����ֵ����Сֵ,minMaxLoc����Ϊ��ͨ���ҷ������/Сֵ����minMaxIdx������Inputarray���ҷ����±�
	cv::minMaxLoc(hist , &minValue , &maxValue , 0 , 0);
	//����ͼ�����űȣ�Ϊsize��90%
	int histSize = hist.rows;
	double hpt = histSize * 0.9;
	cv::Mat _histImg(histSize , histSize , CV_8U  , cv::Scalar(255));
	//����zhifangtu
	for (int i = 0 ; i < histSize ; i ++){
		//���д��int����_valueֵ��ܴ�ܴ�
		float _value = hist.at<float>(i);
		int imgLoc = static_cast<int>(_value * hpt / maxValue);
		cv::line(_histImg , cv::Point(i , histSize) , cv::Point(i , histSize - imgLoc) , cv::Scalar::all(0));
	}
	return _histImg;
}

//���ûҶ�ƽ��ֵֵ��������ֵ
int getMeanThreshold(cv::MatND &hist){
	float sum = 0 ; 
	float amount = 0 ;
	for (int i = 0 ;i < 256 ; i ++){
		amount+=hist.at<float>(i);
		sum = sum + i * hist.at<float>(i);
	}
	return static_cast<int>(sum / amount);
}

//���ðٷֱ���ֵ������ֵ
// Doyle��1962�����P-Tile (��P��λ����)
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

//���ڹȵ���Сֵ����ֵ
//�˷���ʵ���ھ�������˫��ֱ��ͼ��ͼ����Ѱ��˫��Ĺȵ���Ϊ��ֵ��
//���Ǹ÷�����һ���ܻ����ֵ��������Щ����ƽ̹��ֱ��ͼ�򵥷�ͼ�񣬸÷��������ʡ�
//���ֱ��ͼ�Ƿ�Ϊ˫��
bool IsDimodal(double _hist[]){
	int count = 0;//˫�����
	//ע��i���±�1��255
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
	//���ƻ����ݣ��ȵõ�һ�ݿ���
	double _hist[256];
	for (int i = 0 ; i < 256 ; i ++ ){
		_hist[i] = hist.at<float>(i);
	}
	int count = 0;//������
	//ֻ������˫�������²ſ��Լ���,û��˫��ʱ��ͨ������ƽ��ֱ��ͼ
	while (IsDimodal(_hist) == false){
		_hist[0] = (_hist[0] + _hist[0] + _hist[1])/3;
		for (int i = 1 ; i < 255 ;i ++){
			_hist[i] = (_hist[i - 1] + _hist[i] + _hist[i + 1]) / 3;
		}
		_hist[255] = (_hist[255] + _hist[255] + _hist[254]) / 3;
		//hist = cp_hist.clone();
		count ++;
		//����������1000�Σ�����Ϊ�Ҳ���˫��
		if (count > 1000)
			return -1;
	}
	//��ֵΪ˫��֮�����Сֵ.���ҵ�һ���壬Ȼ�����жϹȵ�
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

//����˫��ƽ��ֵ����ֵ
//���㷨�ͻ��ڹȵ���Сֵ����ֵ�������ƣ�
//ֻ�����һ������ȡ��˫��֮��Ĺȵ�ֵ������ȡ˫���ƽ��ֵ��Ϊ��ֵ
int GetIntermodesThreshold(cv::MatND &hist){
	//���ƻ����ݣ��ȵõ�һ�ݿ���
	double _hist[256];
	for (int i = 0 ; i < 256 ; i ++ ){
		_hist[i] = hist.at<float>(i);
	}
	int count = 0;//��������������
	//ֻ������˫�������²ſ��Լ���,û��˫��ʱ��ͨ������ƽ��ֱ��ͼ
	while (IsDimodal(_hist) == false){
		_hist[0] = (_hist[0] + _hist[0] + _hist[1])/3;
		for (int i = 1 ; i < 255 ;i ++){
			_hist[i] = (_hist[i - 1] + _hist[i] + _hist[i + 1]) / 3;
		}
		_hist[255] = (_hist[255] + _hist[255] + _hist[254]) / 3;
		//hist = cp_hist.clone();
		count ++;
		//����������1000�Σ�����Ϊ�Ҳ���˫��
		if (count > 1000)
			return -1;
	}
	//����һ���㷨�Ĳ�ͬ�������������������м�ֵ
	int peakPoint[2];
	for (int i = 1 , index = 0; i < 255 ; i ++){
		if (_hist[i] > _hist[i-1] && _hist[i] > _hist[i+1]){
			peakPoint[index ++] = i;
		}
	}
	return (peakPoint[0] + peakPoint[1]) / 2;
}

//���������ֵ
//�ȼٶ�һ����ֵ��Ȼ������ڸ���ֵ�µ�ǰ���ͱ���������ֵ��
//��ǰ���ͱ�������ֵ��ƽ��ֵ�ͼٶ�����ֵ��ͬʱ���������ֹ�����Դ�ֵΪ��ֵ���ж�ֵ����
int GetIterativeBestThreshold(cv::MatND &hist){
	//�м�ֵ��Ϊ0,�µ��м�ֵ��Ϊ128
	int _threshold_mid = 0;
	int _threshold_new = 128;

	float sum = 0 ; 
	float amount = 0 ;
	int count = 0;//ѭ����������������
	while (_threshold_mid != _threshold_new){
		_threshold_mid = _threshold_new;
		//������_threshold_newΪ�ֽ��ߣ����ߵĻҶ�ƽ��ֵ,
		float sum = 0 ; 
		float amount = 0 ;
		for (int i = 0 ;i <= _threshold_new ; i ++){
			amount=hist.at<float>(i) + amount;
			sum = i * hist.at<float>(i) + sum;
		}
		//ͼ��ǰ��ƽ���Ҷ�ֵ
		int a1 = static_cast<int>(sum / amount);
		sum = 0 ; 
		amount = 0 ;
		for (int i = _threshold_mid + 1 ;i < 256 ; i ++){
			amount+=hist.at<float>(i);
		}
		for (int i = _threshold_mid + 1 ; i < 256 ; i ++){
			sum += i * hist.at<float>(i) / amount;
		}
		//ͼ���ƽ���Ҷ�ֵ
		int a2 = static_cast<int>(sum);
		_threshold_new = static_cast<int>((a1+a2)/2);
		if (++ count > 1000){
			return -1;
		}
	}
	return _threshold_mid;
}

//���ر��ַ�
//���㷨ͨ��ѡ��ǡ������ֵ�Ӷ�ʹ�ö�ֵ���ͼ���ԭʼ�ĻҶ�ͼ�����������ͬ�ĳ�ʼ����ֵ��
//���������http://www.sciencedirect.com/science/article/pii/0734189X85901331
int GetMomentPreservingThreshold(cv::MatND &hist , int &lowGrayValue , int  &highGrayValue){
	//�洢�����е�i��֮�ͣ�amount��ʾ��������
	double sum_1 = 0 , sum_2 = 0 , sum_3 = 0 , amount = 0;
	double sum_0 = 1;
	int threshold_index = -1;//���ص���ֵ
	//avec��ʾǰi���Ҷ�ֵռȫ���İٷֱ�
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
	//�������������������ⷽ�̡��±갴������ʾ
	double Cd = sum_0 * sum_2 - sum_1 * sum_1;
	double C0 = (sum_1 * sum_3 - sum_2 * sum_2) / Cd;
	double C1 = (sum_2 * sum_1 - sum_0 * sum_3) / Cd;
	//Z0��Z1��ֵ
	lowGrayValue = static_cast<int>(0.5 * (-C1 - sqrt(C1 * C1 - 4 * C0)));
	highGrayValue = static_cast<int>(0.5 * (-C1 + sqrt(C1 * C1 - 4 * C0)));
	//Ȼ����P0��P1
	double Pd = highGrayValue - lowGrayValue;
	double P0 = (highGrayValue - sum_1) / Pd;
	//֪����P0�󣬿�������P0λ�õ�huiduzhi
	float minValue = 0.25;
	for (int i = 0 ; i < 256 ; i ++){
		if (abs(avec[i] - P0) < minValue){
			minValue = abs(avec[i] - P0);
			threshold_index = i;
		}
	}
	return threshold_index;
}