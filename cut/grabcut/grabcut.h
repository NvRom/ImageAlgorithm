
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "gcgraph.hpp"
#include <limits>

using namespace cv;

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut �� Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;//��˹ģ�͵�Ȩֵ
    double* mean;//��˹ģ�Ͱ�����3ͨ���ľ�ֵ
    double* cov;//��˹ģ��Э����

    double inverseCovs[componentsCount][3][3]; //Э����������
    double covDeterms[componentsCount];  //Э���������ʽ

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

//������ǰ������һ����Ӧ��GMM����ϸ�˹ģ�ͣ�
GMM::GMM( Mat& _model )
{
	//һ�����صģ�Ψһ��Ӧ����˹ģ�͵Ĳ�����������˵һ����˹ģ�͵Ĳ�������
	//һ������RGB����ͨ��ֵ����3����ֵ��3*3��Э�������һ��Ȩֵ
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
		//һ��GMM����componentsCount����˹ģ�ͣ�һ����˹ģ����modelSize��ģ�Ͳ���
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

	//ע����Щģ�Ͳ����Ĵ洢��ʽ��������componentsCount��coefs����3*componentsCount��mean��
	//��3*3*componentsCount��cov��
    coefs = model.ptr<double>(0);  //GMM��ÿ�����صĸ�˹ģ�͵�Ȩֵ������ʼ�洢ָ��
    mean = coefs + componentsCount; //��ֵ������ʼ�洢ָ��
    cov = mean + 3*componentsCount;  //Э���������ʼ�洢ָ��

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
			 //����GMM�е�ci����˹ģ�͵�Э�������Inverse������ʽDeterminant
			 //Ϊ�˺������ÿ���������ڸø�˹ģ�͵ĸ��ʣ�Ҳ�������������
             calcInverseCovAndDeterm( ci ); 
}

//����һ�����أ���color=��B,G,R����άdouble����������ʾ���������GMM��ϸ�˹ģ�͵ĸ��ʡ�
//Ҳ���ǰ����������������componentsCount����˹ģ�͵ĸ������Ӧ��Ȩֵ�������ӣ�
//��������ĵĹ�ʽ��10���������res���ء�
//����൱�ڼ���Gibbs�����ĵ�һ�������ȡ���󣩡�
double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

//����һ�����أ���color=��B,G,R����άdouble����������ʾ�����ڵ�ci����˹ģ�͵ĸ��ʡ�
//������̣����߽׵ĸ�˹�ܶ�ģ�ͼ���ʽ����������ĵĹ�ʽ��10���������res����
double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

//��������������п�������GMM�е��ĸ���˹ģ�ͣ����������Ǹ���
int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;  //�ҵ����������Ǹ�������˵�����������Ǹ�
            max = p;
        }
    }
    return k;
}

//GMM����ѧϰǰ�ĳ�ʼ������Ҫ�Ƕ�Ҫ��͵ı�������
void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

//������������Ϊǰ�����߱���GMM�ĵ�ci����˹ģ�͵����ؼ���������ؼ������ù�
//�Ƽ��������˹ģ�͵Ĳ����ģ������������ء��������color������غ����ؼ�
//���������ص�RGB����ͨ���ĺ�sums�����������ֵ������������prods����������Э�����
//���Ҽ�¼������ؼ������ظ������ܵ����ظ������������������˹ģ�͵�Ȩֵ����
void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

//��ͼ��������ѧϰGMM�Ĳ�����ÿһ����˹������Ȩֵ����ֵ��Э�������
//�����൱�������С�Iterative minimisation����step 2
void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci]; //��ci����˹ģ�͵��������ظ���
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            //�����ci����˹ģ�͵�Ȩֵϵ��
			coefs[ci] = (double)n/totalSampleCount; 

            //�����ci����˹ģ�͵ľ�ֵ
			double* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

            //�����ci����˹ģ�͵�Э����
			double* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

            //�����ci����˹ģ�͵�Э���������ʽ
			double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                //�൱���������ʽС�ڵ���0�����Խ���Ԫ�أ����Ӱ��������������
				//Ϊ�˻������ȣ�Э������󣨲���������󣬵�����ļ�����Ҫ��������󣩡�
				// Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }
			
			//�����ci����˹ģ�͵�Э�������Inverse������ʽDeterminant
            calcInverseCovAndDeterm(ci);
        }
    }
}

//����Э�������Inverse������ʽDeterminant
void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
		//ȡ��ci����˹ģ�͵�Э�������ʼָ��
        double *c = cov + 9*ci;
        double dtrm =
              covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) 
								+ c[2]*(c[3]*c[7]-c[4]*c[6]);

        //��C++�У�ÿһ�����õ��������Ͷ�ӵ�в�ͬ������, ʹ��<limits>����Ի�
		//����Щ�����������͵���ֵ���ԡ���Ϊ�����㷨�Ľضϣ�����ʹ�ã���a=2��
		//b=3ʱ 10*a/b == 20/b������������ô���أ�
		//���С������epsilon�����������ˣ�С����ͨ��Ϊ���ø����������͵�
		//����1����Сֵ��1֮������ʾ����dtrm���������С��������ô������Ϊ�㡣
		//������ʽ��֤dtrm>0��������ʽ�ļ�����ȷ��Э����Գ�������������ʽ����0����
		CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
		//���׷��������
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}

//����beta��Ҳ����Gibbs�������еĵڶ��ƽ����е�ָ�����beta����������
//�߻��ߵͶԱȶ�ʱ�������������صĲ���Ӱ��ģ������ڵͶԱȶ�ʱ����������
//���صĲ����ܾͻ�Ƚ�С����ʱ����Ҫ����һ���ϴ��beta���Ŵ�������
//�ڸ߶Աȶ�ʱ������Ҫ��С����ͱȽϴ�Ĳ��
//����������Ҫ��������ͼ��ĶԱȶ���ȷ������beta������ļ����Ĺ�ʽ��5����
/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
			//�����ĸ��������������صĲ��Ҳ����ŷʽ�������˵���׷���
			//�����������ض�����󣬾��൱�ڼ������������ز��ˣ�
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left  >0���ж���Ϊ�˱�����ͼ��߽��ʱ�򻹼��㣬����Խ��
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) ); //���Ĺ�ʽ��5��

    return beta;
}

//����ͼÿ���Ƕ˵㶥�㣨Ҳ����ÿ��������Ϊͼ��һ�����㣬������Դ��s�ͻ��t�������򶥵�
//�ıߵ�Ȩֵ������������ͼ�����Ǽ�����ǰ�������ô����һ�����㣬���Ǽ����ĸ�������У�
//�������Ķ�������ʱ�򣬻��ʣ�����ĸ������Ȩֵ�����������������ͼ�����ÿ������
//�������Ķ���ıߵ�Ȩֵ�Ͷ���������ˡ�
//����൱�ڼ���Gibbs�����ĵڶ��������ƽ���������������й�ʽ��4��
/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, 
							Mat& uprightW, double beta, double gamma )
{
    //gammaDivSqrt2�൱�ڹ�ʽ��4���е�gamma * dis(i,j)^(-1)����ô����֪����
	//��i��j�Ǵ�ֱ����ˮƽ��ϵʱ��dis(i,j)=1�����ǶԽǹ�ϵʱ��dis(i,j)=sqrt(2.0f)��
	//�������ʱ���������������
	const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
	//ÿ������ıߵ�Ȩֵͨ��һ����ͼ��С��ȵ�Mat������
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left  //����ͼ�ı߽�
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

//���mask����ȷ�ԡ�maskΪͨ���û��������߳����趨�ģ����Ǻ�ͼ���Сһ���ĵ�ͨ���Ҷ�ͼ��
//ÿ������ֻ��ȡGC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD ����ö��ֵ���ֱ��ʾ������
//���û����߳���ָ�������ڱ�����ǰ��������Ϊ�������߿���Ϊǰ�����ء�����Ĳο���
//ICCV2001��Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images��
//Yuri Y. Boykov Marie-Pierre Jolly 
/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

//ͨ���û���ѡĿ��rect������mask��rect���ȫ����Ϊ����������ΪGC_BGD��
//rect�ڵ�����Ϊ GC_PR_FGD������Ϊǰ����
/*
  Initialize mask using rectangular.
*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, imgSize.width-rect.x);
    rect.height = min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

//ͨ��k-means�㷨����ʼ������GMM��ǰ��GMMģ��
/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;  //��������
    const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii

    Mat bgdLabels, fgdLabels; //��¼������ǰ����������������ÿ�����ض�ӦGMM���ĸ���˹ģ�ͣ������е�kn
    vector<Vec3f> bgdSamples, fgdSamples; //������ǰ��������������
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            //mask�б��ΪGC_BGD��GC_PR_BGD�����ض���Ϊ��������������
			if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
	
	//kmeans�в���_bgdSamplesΪ��ÿ��һ������
	//kmeans�����ΪbgdLabels�����汣�����������������ÿһ��������Ӧ�����ǩ��������ΪcomponentsCount���
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    //��������Ĳ����ÿ�����������ĸ�˹ģ�;�ȷ�����ˣ���ô�Ϳ��Թ���GMM��ÿ����˹ģ�͵Ĳ����ˡ�
	bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

//�����У�������С���㷨step 1��Ϊÿ�����ط���GMM�������ĸ�˹ģ�ͣ�kn������Mat compIdxs��
/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, 
									const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
			//ͨ��mask���жϸ��������ڱ������ػ���ǰ�����أ����ж�������ǰ�����߱���GMM�е��ĸ���˹����
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

//�����У�������С���㷨step 2����ÿ����˹ģ�͵�������������ѧϰÿ����˹ģ�͵Ĳ���
/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

//ͨ������õ����������ͼ��ͼ�Ķ���Ϊ���ص㣬ͼ�ı��������ֹ��ɣ�
//һ����ǣ�ÿ��������Sink���t������������Դ��Source������ǰ�������ӵıߣ�
//����ߵ�Ȩֵͨ��Gibbs������ĵ�һ������������ʾ��
//��һ����ǣ�ÿ�������������򶥵����ӵıߣ�����ߵ�Ȩֵͨ��Gibbs������ĵڶ�������������ʾ��
/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       GCGraph<double>& graph )
{
    int vtxCount = img.cols*img.rows;  //��������ÿһ��������һ������
    int edgeCount = 2*(4*vtxCount - 3*(img.cols + img.rows) + 2);  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
    //ͨ���������ͱ�������ͼ����Щ���������ͺ���������ο�gcgraph.hpp
	graph.create(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();  //�������������ͼ�е�����
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights			
            //����ÿ��������Sink���t������������Դ��Source������ǰ�������ӵ�Ȩֵ��
			//Ҳ������Gibbs������ÿһ�����ص���Ϊ�������ػ���ǰ�����أ��ĵ�һ��������
			double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                //��ÿһ�����ؼ�������Ϊ�������ػ���ǰ�����صĵ�һ���������Ϊ�ֱ���t��s�������Ȩֵ
				fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                //����ȷ��Ϊ���������ص㣬����Source�㣨ǰ����������Ϊ0����Sink�������Ϊlambda
				fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
			//���øö���vtxIdx�ֱ���Source���Sink�������Ȩֵ
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights  n-links
            //�����������򶥵�֮�����ӵ�Ȩֵ��
			//Ҳ������Gibbs�����ĵڶ��������ƽ���
			if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

//�����У�������С���㷨step 3���ָ���ƣ���С�����������㷨
/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    //ͨ��������㷨ȷ��ͼ����С�Ҳ�����ͼ��ķָ�
	graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            //ͨ��ͼ�ָ�Ľ��������mask��������ͼ��ָ�����ע����ǣ���Զ��
			//��������û�ָ��Ϊ��������ǰ��������
			if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

//���ĳɹ����ṩ�����ʹ�õ�ΰ���API��grabCut 
/*
****����˵����
	img�������ָ��Դͼ�񣬱�����8λ3ͨ����CV_8UC3��ͼ���ڴ���Ĺ����в��ᱻ�޸ģ�
	mask��������ͼ�����ʹ��������г�ʼ������ômask�����ʼ��������Ϣ����ִ�зָ�
		��ʱ��Ҳ���Խ��û��������趨��ǰ���뱳�����浽mask�У�Ȼ���ٴ���grabCut��
		�����ڴ������֮��mask�лᱣ������maskֻ��ȡ��������ֵ��
		GCD_BGD��=0����������
		GCD_FGD��=1����ǰ����
		GCD_PR_BGD��=2�������ܵı�����
		GCD_PR_FGD��=3�������ܵ�ǰ����
		���û���ֹ����GCD_BGD����GCD_FGD����ô���ֻ����GCD_PR_BGD��GCD_PR_FGD��
	rect���������޶���Ҫ���зָ��ͼ��Χ��ֻ�иþ��δ����ڵ�ͼ�񲿷ֲű�����
	bgdModel��������ģ�ͣ����Ϊnull�������ڲ����Զ�����һ��bgdModel��bgdModel������
		��ͨ�������ͣ�CV_32FC1��ͼ��������ֻ��Ϊ1������ֻ��Ϊ13x5��
	fgdModel����ǰ��ģ�ͣ����Ϊnull�������ڲ����Զ�����һ��fgdModel��fgdModel������
		��ͨ�������ͣ�CV_32FC1��ͼ��������ֻ��Ϊ1������ֻ��Ϊ13x5��
	iterCount���������������������0��
	mode��������ָʾgrabCut��������ʲô��������ѡ��ֵ�У�
		GC_INIT_WITH_RECT��=0�����þ��δ���ʼ��GrabCut��
		GC_INIT_WITH_MASK��=1����������ͼ���ʼ��GrabCut��
		GC_EVAL��=2����ִ�зָ
*/
void cv::grabCut( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode )
{
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
    Mat compIdxs( img.size(), CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.size(), rect );
        else // flag == GC_INIT_WITH_MASK
            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    if( iterCount <= 0)
        return;

    if( mode == GC_EVAL )
        checkMask( img, mask );

    const double gamma = 50;
    const double lambda = 9*gamma;
    const double beta = calcBeta( img );

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );

    for( int i = 0; i < iterCount; i++ )
    {
        GCGraph<double> graph;
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
        estimateSegmentation( graph, mask );
    }
}

