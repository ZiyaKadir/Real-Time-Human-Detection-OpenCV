#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>


using namespace std;
using namespace cv;


void ComputeGradients(const Mat &frame, Mat &Gx, Mat &Gy) {
    Gx = Mat::zeros(frame.size(), CV_32F);
    Gy = Mat::zeros(frame.size(), CV_32F);

    for(int y = 1; y < frame.rows - 1; y++){
        for(int x = 1; x < frame.cols - 1; x++){
            float valxm1 = (float)frame.at<uint8_t>(y, x - 1);
            float valxp1 = (float)frame.at<uint8_t>(y, x + 1);
            float valym1 = (float)frame.at<uint8_t>(y - 1, x);
            float valyp1 = (float)frame.at<uint8_t>(y + 1, x);

            Gx.at<float>(y,x) = (valxp1 - valxm1) * 0.5f;
            Gy.at<float>(y,x) = (valyp1 - valym1) * 0.5f;
        }
    }
}


void Erosion (Mat& frame, int rows, int cols){

    bool isEroded = false;
    Mat* output = new Mat(rows, cols, CV_8UC1);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            isEroded = true;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    if (frame.at<uint8_t>(i + k, j + l) == 0) {
                        isEroded = false;
                        break;
                    }
                }
                if (!isEroded) break;
            }
            output->at<uint8_t>(i, j) = isEroded ? 255 : 0;
        }
    }

    frame = output->clone();
}

void Dilation (Mat& frame, int rows, int cols){

    bool isDilated = false;
    Mat* output = new Mat(rows, cols, CV_8UC1);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            isDilated = false;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    if (frame.at<uint8_t>(i + k, j + l) == 255) {
                        isDilated = true;
                        break;
                    }
                }
                if (isDilated) break;
            }
            output->at<uint8_t>(i, j) = isDilated ? 255 : 0;
        }
    }

    frame = output->clone();
}

void ApplyThreshold(const Mat& input, Mat& output, uint8_t threshold, uint8_t maxValue) {
    output = Mat(input.rows, input.cols, CV_8UC1);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            uint8_t pixelValue = input.at<uint8_t>(i, j);

            if (pixelValue > threshold) {
                output.at<uint8_t>(i, j) = maxValue;
            } else {
                output.at<uint8_t>(i, j) = 0;
            }
        }
    }
}

void ApplyThreshold3Channel(const Mat& input, Mat& output, Vec3b threshold, Vec3b maxValue) {
    output = Mat(input.rows, input.cols, CV_8UC3);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            Vec3b pixelValue = input.at<Vec3b>(i, j);

            for (int c = 0; c < 3; c++) {
                if (pixelValue[c] > threshold[c]) {
                    output.at<Vec3b>(i, j)[c] = maxValue[c];
                } else {
                    output.at<Vec3b>(i, j)[c] = 0;
                }
            }
        }
    }
}


Mat& eliminate_sides(Mat& frame, int rows, int cols, int side, Mat& output) {
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < side; j++) {
            frame.at<uint8_t>(i, j) = 0;
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = cols - side; j < cols; j++) {
            frame.at<uint8_t>(i, j) = 0;
        }
    }

    for (int i = 0; i < side; i++) {
        for (int j = 0; j < cols; j++) {
            frame.at<uint8_t>(i, j) = 0;
        }
    }

    for (int i = rows - side; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            frame.at<uint8_t>(i, j) = 0;
        }
    }

    output = frame.clone();

    return output;
}

Mat& GaussianBlurCustom(Mat& frame, int rows, int cols, int kernelSize, double sigma) {
    // Create Gaussian kernel
    int halfSize = kernelSize / 2;
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize));
    double sum = 0.0;
    double s = 2.0 * sigma * sigma;

    for (int x = -halfSize; x <= halfSize; x++) {
        for (int y = -halfSize; y <= halfSize; y++) {
            double r = sqrt(x * x + y * y);
            kernel[x + halfSize][y + halfSize] = (exp(-(r * r) / s)) / (CV_PI * s);
            sum += kernel[x + halfSize][y + halfSize];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    // Apply Gaussian blur
    Mat* result = new Mat(rows, cols, frame.type());
    for (int i = halfSize; i < rows - halfSize; i++) {
        for (int j = halfSize; j < cols - halfSize; j++) {
            double sum = 0.0;
            for (int k = -halfSize; k <= halfSize; k++) {
                for (int l = -halfSize; l <= halfSize; l++) {
                    sum += frame.at<uint8_t>(i + k, j + l) * kernel[k + halfSize][l + halfSize];
                }
            }
            result->at<uint8_t>(i, j) = static_cast<uint8_t>(sum);
        }
    }

    return *result;
}
Mat& take_differences(Mat& previous, Mat& current, Mat& output) {
    // Calculate the absolute difference between the two frames
    
    for (int i = 0; i < previous.rows -1; i++) {
        for (int j = 0; j < previous.cols -1; j++) {
            Vec3b pixel1 = previous.at<Vec3b>(i, j);
            Vec3b pixel2 = current.at<Vec3b>(i, j);
            Vec3b diffPixel;
            diffPixel[0] = abs(pixel1[0] - pixel2[0]);
            diffPixel[1] = abs(pixel1[1] - pixel2[1]);
            diffPixel[2] = abs(pixel1[2] - pixel2[2]);
            output.at<Vec3b>(i, j) = diffPixel;
        }
    }

    return output;
}

Mat& take_differences_gray(Mat& previous, Mat& current, Mat& output) {
    // Calculate the absolute difference between the two frames
    for (int i = 0; i < previous.rows -1; i++) {
        for (int j = 0; j < previous.cols -1; j++) {
            // cout << "i: " << i << " j: " << j << endl;
            uint8_t pixel1 = previous.at<uint8_t>(i, j);
            uint8_t pixel2 = current.at<uint8_t>(i, j);
            uint8_t diffPixel;
            diffPixel = abs(pixel1 - pixel2);
            // cout << "diffPixel: " << diffPixel << endl;
            output.at<uint8_t>(i, j) = diffPixel;
            // cout << "output: " << output.at<uint8_t>(i, j) << endl;
        }
    }

    return output;
}

Mat& substract_grays(Mat& previous, Mat& current, Mat& output) {
    // Calculate the absolute difference between the two frames
    for (int i = 0; i < previous.rows -1; i++) {
        for (int j = 0; j < previous.cols -1; j++) {
            // cout << "i: " << i << " j: " << j << endl;
            uint8_t pixel1 = previous.at<uint8_t>(i, j);
            uint8_t pixel2 = current.at<uint8_t>(i, j);
            uint8_t diffPixel;
            diffPixel = pixel1 - pixel2;

            


            output.at<uint8_t>(i, j) = diffPixel;
        }
    }

    return output;
}



Mat& Grayscale(Mat &frame) {
    // Create a new matrix for the grayscale image (single channel)
    // Mat grayImage(frame.rows, frame.cols, CV_8UC1); // Single channel (grayscale)
    Mat* grayImage = new Mat(frame.rows, frame.cols, CV_8UC1);

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            Vec3b pixel = frame.at<Vec3b>(i, j);  // Access each pixel
            uint8_t grayValue = static_cast<uint8_t>(0.2989 * pixel[2] + 0.5870 * pixel[1] + 0.1140 * pixel[0]);
            (*grayImage).at<uint8_t>(i, j) = grayValue; // Store in the single channel grayscale image
        }
    }

    return *grayImage;
}

Mat& Gaussion_Filter_gray(Mat& frame, int rows, int cols){

    int Gaussion_filter[3][3] = 
    {{1, 2, 1},
     {2, 4, 2},
     {1, 2, 1}};

    int sum;

    Mat* result = new Mat(rows, cols, CV_8UC1);

    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {

            sum = 0;

            for(int k = -1; k <= 1; k++){
                for(int l = -1; l <= 1; l++){
                    sum += frame.at<uint8_t>(i+k, j+l) * Gaussion_filter[k+1][l+1];
                }
            }

            (*result).at<uint8_t>(i, j) = sum / 16;
        }
    }

    return *result;
}


void Calculate_Gradient_magnitude(Mat& frame, int rows, int cols, int Sober_filter[3][3], Mat& gradientMagnitude, Mat& gradientDirection){

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float Gx = 0, Gy = 0;
            
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    Gx += frame.at<uint8_t>(i + k, j + l) * Sober_filter[k + 1][l + 1];
                    Gy += frame.at<uint8_t>(i + k, j + l) * Sober_filter[l + 1][k + 1];
                }
            }

            gradientMagnitude.at<float>(i, j) = sqrt(Gx * Gx + Gy * Gy);
            gradientDirection.at<float>(i, j) = atan2(Gy, Gx);
        }
    }

}



Mat& CannyEdgeDetection(Mat& frame, int rows, int cols, int lowThreshold, int highThreshold, Mat& output) {

    int Sober_filter[3][3] = 
    {{-1, 0, 1},
     {-2, 0, 2},
     {-1, 0, 1}};

    Mat grayscale = Mat(rows, cols, CV_8UC1);
    Mat grayscale2 = Mat(rows, cols, CV_8UC1);

    Mat gradientMagnitude = Mat(rows, cols, CV_32FC1, Scalar(0));
    Mat gradientDirection = Mat(rows, cols, CV_32FC1, Scalar(0));
    Mat nonMaxSuppressed = Mat(rows, cols, CV_8UC1, Scalar(0));
    Mat edges = Mat(rows, cols, CV_8UC1, Scalar(0));


    // Median_Filter(frame, rows, cols);
    // Gaussion_Filter(frame, rows, cols);
    grayscale = Grayscale(frame); // take the grayscale image


    Calculate_Gradient_magnitude(grayscale, rows, cols, Sober_filter, gradientMagnitude, gradientDirection);


    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float angle = gradientDirection.at<float>(i, j) * 180.0 / CV_PI;
            angle = angle < 0 ? angle + 180 : angle;

            float magnitude = gradientMagnitude.at<float>(i, j);
            float neighbor1 = 0, neighbor2 = 0;

            if ((angle > 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180)) {
                neighbor1 = gradientMagnitude.at<float>(i, j - 1);
                neighbor2 = gradientMagnitude.at<float>(i, j + 1);
            } else if (angle > 22.5 && angle <= 67.5) {
                neighbor1 = gradientMagnitude.at<float>(i - 1, j + 1);
                neighbor2 = gradientMagnitude.at<float>(i + 1, j - 1);
            } else if (angle > 67.5 && angle <= 112.5) {
                neighbor1 = gradientMagnitude.at<float>(i - 1, j);
                neighbor2 = gradientMagnitude.at<float>(i + 1, j);
            } else if (angle > 112.5 && angle <= 157.5) {
                neighbor1 = gradientMagnitude.at<float>(i - 1, j - 1);
                neighbor2 = gradientMagnitude.at<float>(i + 1, j + 1);
            }

            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                nonMaxSuppressed.at<uint8_t>(i, j) = static_cast<uint8_t>(magnitude);
            } else {
                nonMaxSuppressed.at<uint8_t>(i, j) = 0;
            }
        }
    }

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            uint8_t value = nonMaxSuppressed.at<uint8_t>(i, j);

            if (value >= highThreshold) {
                edges.at<uint8_t>(i, j) = 255;
            } else if (value >= lowThreshold) {
                bool isConnectedToStrongEdge = false;
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (nonMaxSuppressed.at<uint8_t>(i + di, j + dj) >= highThreshold) {
                            isConnectedToStrongEdge = true;
                            break;
                        }
                    }
                    if (isConnectedToStrongEdge) break;
                }
                edges.at<uint8_t>(i, j) = isConnectedToStrongEdge ? 255 : 0;
            } else {
                edges.at<uint8_t>(i, j) = 0;
            }
        }
    }

    output = edges.clone();

    return output;
}



int count_boxes(Mat& frame, int rows1, int row2) {
    int count = 0;

    for (int i = rows1; i < row2; i++) {
        for (int j = 0; j < frame.cols; j++) {
            if (frame.at<uint8_t>(i, j) == 255) { // Calculate the average of all white indexes in every column
                count++;
            }
        }
    }

    return count;
    
}


void MOG2(const Mat& frame, 
                Mat& foregroundMask,
                float alpha = 0.01f,  // learning rate
                int K = 5,           // number of Gaussians per pixel
                float T = 0.7f,      // background threshold (ratio of summed weights)
                float initialVar = 900.0f)  // initial variance for newly replaced Gaussians
{
    static bool firstCall = true;
    static int rows = 0;
    static int cols = 0;

    static vector<Mat> means;
    static vector<Mat> variances;
    static vector<Mat> weights;

    if (firstCall || rows != frame.rows || cols != frame.cols) {
        firstCall = false;
        rows = frame.rows;
        cols = frame.cols;

        means.resize(K);
        variances.resize(K);
        weights.resize(K);

        for (int i = 0; i < K; i++) {
            means[i] = Mat(rows, cols, CV_32FC1, Scalar(0));
            variances[i] = Mat(rows, cols, CV_32FC1, Scalar(initialVar));
            weights[i] = Mat(rows, cols, CV_32FC1, (i == 0) ? Scalar(1.0f) : Scalar(0.0f));
        }
    }

    foregroundMask.create(rows, cols, CV_8UC1);

    for (int y = 0; y < rows; y++) {
        const uchar* rowPtr = frame.ptr<uchar>(y);
        for (int x = 0; x < cols; x++) {
            float pixelVal = static_cast<float>(rowPtr[x]);

            bool matched = false;
            int matchedIndex = -1;

            for (int i = 0; i < K; i++) {
                float m = means[i].at<float>(y, x);
                float v = variances[i].at<float>(y, x);
                float sigma = sqrt(v);

                if (fabs(pixelVal - m) <= 2.5f * sigma) {
                    matched = true;
                    matchedIndex = i;
                    break;
                }
            }

            if (matched) {
                float& w = weights[matchedIndex].at<float>(y, x);
                float& m = means[matchedIndex].at<float>(y, x);
                float& v = variances[matchedIndex].at<float>(y, x);

                w = (1.0f - alpha) * w + alpha;
                float diff = (pixelVal - m);
                m = m + alpha * diff;
                float diff2 = (pixelVal - m);
                v = v + alpha * (diff2 * diff2 - v);
            } else {
                int lowestIndex = 0;
                float lowestWeight = weights[0].at<float>(y, x);

                for (int i = 1; i < K; i++) {
                    float w_i = weights[i].at<float>(y, x);
                    if (w_i < lowestWeight) {
                        lowestWeight = w_i;
                        lowestIndex = i;
                    }
                }

                weights[lowestIndex].at<float>(y, x) = alpha;
                means[lowestIndex].at<float>(y, x) = pixelVal;
                variances[lowestIndex].at<float>(y, x) = initialVar;
            }

            float sumWeight = 0.0f;
            for (int i = 0; i < K; i++) {
                float& w = weights[i].at<float>(y, x);
                w = (1.0f - alpha) * w;
                sumWeight += w;
            }
            if (matched && matchedIndex >= 0) {
                weights[matchedIndex].at<float>(y, x) += alpha;
                sumWeight += alpha;
            }
            for (int i = 0; i < K; i++) {
                weights[i].at<float>(y, x) /= sumWeight;
            }
        }
    }

    for (int y = 0; y < rows; y++) {
        const uchar* rowPtr = frame.ptr<uchar>(y);
        uchar* outPtr = foregroundMask.ptr<uchar>(y);

        for (int x = 0; x < cols; x++) {
            vector<pair<float, int>> wVec;
            for (int i = 0; i < K; i++) {
                float w = weights[i].at<float>(y, x);
                wVec.push_back({w, i});
            }
            sort(wVec.begin(), wVec.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
                return a.first > b.first;
            });

            float accumWeight = 0.0f;
            int backgroundCount = 0;
            for (int i = 0; i < K; i++) {
                accumWeight += wVec[i].first;
                backgroundCount++;
                if (accumWeight > T) {
                    break;
                }
            }

            float pixelVal = static_cast<float>(rowPtr[x]);
            bool isBackground = false;

            for (int i = 0; i < backgroundCount; i++) {
                int gIndex = wVec[i].second;
                float m = means[gIndex].at<float>(y, x);
                float v = variances[gIndex].at<float>(y, x);
                float sigma = sqrt(v);

                if (fabs(pixelVal - m) <= 2.5f * sigma) {
                    isBackground = true;
                    break;
                }
            }

            outPtr[x] = (isBackground ? 0 : 255);
        }
    }
}



