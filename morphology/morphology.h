/*
��̬ѧ����
@author NvRom
*/
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//�ֱ��ʾΪ���Σ�ʮ�֣���Բ
enum morShape{
	MOR_RECT , MOR_CROSS , MOR_ELLOPSE
};

//��̬ѧ��������
enum  morOp{
	EROSION , DILATION
};

/*
�õ�ָ����С�͹�ģ�Ľṹ��
@param shape:�ṹ�����״���о��Ρ�ʮ���Ρ���Բ�Ρ���ûָ����״��Ĭ��Ϊ����
@param size:�ṹ��Ĵ�С����ûָ����С��Ĭ��Ϊ3*3
@param anchor:�ṹ���ԭ��λ�ã�Ĭ��anchorΪ(-1,-1),��ʾ�ṹ�������
*/
cv::Mat getSE(int shape , cv::Size size = cv::Size(3,3) , cv::Point anchor = cv::Point(-1,-1));

/*
���ṹ��ԭ����Ĭ�ϣ��򷵻����ĵ�
�˺�����getSE��������
@param size:�ṹ��Ĵ�С
@param anchor:�ṹ���ԭ��λ��
*/
cv::Point getNormalAnchor(cv::Size size , cv::Point anchor);

/*
�жϽṹ����ԭͼ�����Ƿ�ƥ�䣬ƥ�䷵��true�����򷵻�false
@param src������ͼ�񣬶�ֵ����ͼ��
@param op���������ͣ��и�ʴ��erosion�������ͣ�dilation�����ֻ�������
@param (rowIndex,colIndex)��kernelԭ������ͼ���λ��
@param kernel���ṹ�壨SE��������ͨ��getSE�����õ����ṩ���¼���kernel���ֱ�Ϊ��MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor���ṹ���ԭ�㣬Ĭ��anchorΪ��-1��-1����ʾԭ��λ�ڽṹ�������λ��
*/
bool SEMatchSrc(cv::Mat src , int op , int rowIndex , int colIndex , cv::Mat kernel , cv::Point anchor);

/*
ͼ��ĸ�ʴ����
@param src������ͼ��Ҫ���ǻҶ�ͼ
@param dst�����ͼ�񣬴�С��������srcһ��
@param kernel���ṹ�壨SE��������ͨ��getSE�����õ����ṩ���¼���kernel���ֱ�Ϊ��MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor���ṹ���ԭ�㣬Ĭ��anchorΪ��-1��-1����ʾԭ��λ�ڽṹ�������λ��
*/
void erosion(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
ͼ������Ͳ���
@param src������ͼ��Ҫ���ǻҶ�ͼ
@param dst�����ͼ�񣬴�С��������srcһ��
@param kernel���ṹ�壨SE��������ͨ��getSE�����õ����ṩ���¼���kernel���ֱ�Ϊ��MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor���ṹ���ԭ�㣬Ĭ��anchorΪ��-1��-1����ʾԭ��λ�ڽṹ�������λ��
*/
void dilation(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
ͼ�񿪲���,�ȶ�ͼ����и�ʴ��Ȼ��������
@param src������ͼ��Ҫ���ǻҶ�ͼ
@param dst�����ͼ�񣬴�С��������srcһ��
@param kernel���ṹ�壨SE��������ͨ��getSE�����õ����ṩ���¼���kernel���ֱ�Ϊ��MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor���ṹ���ԭ�㣬Ĭ��anchorΪ��-1��-1����ʾԭ��λ�ڽṹ�������λ��
*/
void opening(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
ͼ��رղ������ȶ�ͼ��������ͣ�Ȼ���ٸ�ʴ
����ͬ��
*/
void closing(cv::Mat src , cv::Mat &dst , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));

/*
�߼����������src��mask���������Ĳ��֣�������src
@param src ����ͼ��
@param mask ģ�壬src��maskΪĿ����лָ�
*/
void logicAnd(cv::Mat &src , cv::Mat mask);

/*
�ж�����ͼ���Ƿ���ȫ�Ǻ�
@param img ԭͼ�񣬴��Ƚϵ�ͼ��
@param mask Ŀ��ͼ��
*/
bool matchMask(cv::Mat img , cv::Mat mask);

/*
����dilation������ͼ���
@param mask������ͼ��Ҫ���ǻҶ�ͼ,ͬʱ����marker�������ͣ��ָ�����mask�ֲ�һ����ͼ��
@param marker�����ָ���ͼ�񣬴�mask��opening�õ�
@param kernel���ṹ�壨SE��������ͨ��getSE�����õ����ṩ���¼���kernel���ֱ�Ϊ��MOR_RECT , MOR_CROSS , MOR_ELLOPSE
@param anchor���ṹ���ԭ�㣬Ĭ��anchorΪ��-1��-1����ʾԭ��λ�ڽṹ�������λ��
*/
cv::Mat geodesticDilation(cv::Mat mask , cv::Mat marker , cv::Mat kernel , cv::Point anchor = cv::Point(-1,-1));