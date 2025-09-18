#ifndef FILTER_H
#define FILTER_H


#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>


using namespace std;
using namespace cv;


void Dilation (Mat& frame, int rows, int cols);
void Erosion(Mat& frame, int rows, int cols);
void ApplyThreshold(const Mat& input, Mat& output, uint8_t threshold, uint8_t maxValue);
Mat& eliminate_sides(Mat& frame, int rows, int cols, int side, Mat& output);
Mat& take_differences(Mat& previous, Mat& current, Mat& output);
Mat& take_differences_gray(Mat& previous, Mat& current, Mat& output);
Mat& substract_grays(Mat& previous, Mat& current, Mat& output);
Mat& Grayscale(Mat &frame);
Mat& CannyEdgeDetection(Mat& frame, int rows, int cols, int lowThreshold, int highThreshold, Mat& output);
// void Calculate_Gradient_magnitude(Mat& frame, int rows, int cols, int Sober_filter[3][3], Mat& gradientMagnitude, Mat& gradientDirection);
// void ComputeGradients(const Mat &frame, Mat &Gx, Mat &Gy);
Mat& Gaussion_Filter_gray(Mat& frame, int rows, int cols);
int count_boxes(Mat& frame, int rows1, int row2);
// void FillAreaBasedOnNeighbors(Mat& frame, int rows, int cols, Mat& output);
void MOG2(const Mat& frame, 
                Mat& foregroundMask,
                float alpha = 0.01f,  // learning rate
                int K = 5,           // number of Gaussians per pixel
                float T = 0.7f,      // background threshold (ratio of summed weights)
                float initialVar = 900.0f);


void ApplyThreshold3Channel(const Mat& input, Mat& output, Vec3b threshold, Vec3b maxValue);
Mat& GaussianBlurCustom(Mat& frame, int rows, int cols, int kernelSize, double sigma);










#endif