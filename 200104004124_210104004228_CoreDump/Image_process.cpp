#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#include "Filter.h"


using namespace std;
using namespace cv;

vector<int> getHistogramAndCompress(const cv::Mat& edgeFrame, int compressedSize = 100)
{
    // Ensure the input image is single-channel 8-bit
    CV_Assert(edgeFrame.type() == CV_8UC1);

    // 1) Compute the number of "white" pixels in each column of the original
    vector<int> originalHistogram(edgeFrame.cols, 0);
    for (int col = 0; col < edgeFrame.cols; col++)
    {
        int whiteCount = 0;
        for (int row = 0; row < edgeFrame.rows; row++)
        {
            // Count if pixel is 255 (edge/white)
            if (edgeFrame.at<uchar>(row, col) == 255)
            {
                whiteCount++;
            }
        }
        originalHistogram[col] = whiteCount;
    }

    // 2) Compress (downsample) the histogram to compressedSize columns
    vector<int> compressedHistogram(compressedSize, 0);

    // Simple binning approach
    for (int col = 0; col < edgeFrame.cols; col++)
    {
        // Map the column to one of the new compressed bins
        int binIndex = static_cast<int>(
            static_cast<double>(col) * compressedSize / edgeFrame.cols
        );
        // Accumulate the count
        compressedHistogram[binIndex] += originalHistogram[col];
    }

    // Return the final compressed histogram
    return compressedHistogram;
}

int main() {
    // Replace the camera capture with video file capture
    string videoFilePath = "Video3.mp4"; // Update this with your MP4 file path
    VideoCapture cap(videoFilePath);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    int counter = 0;


    Mat frame;
    cap >> frame;


    Mat first = Mat(frame.rows, frame.cols, CV_8UC1);

    CannyEdgeDetection(frame, frame.rows, frame.cols, 20, 80, first);
    
    cap >> frame;

    Mat second = Mat(frame.rows, frame.cols, CV_8UC1);

    CannyEdgeDetection(frame, frame.rows, frame.cols, 20, 80, second);
    

    // imshow("First Canny Edge Detection", first);
    // waitKey(0);
    // making the edges thicker
    for (int i = 0; i <5; i++) {
        Dilation(second, second.rows, second.cols);
    }

    // imshow("Second Canny Edge Detection", second);
    // waitKey(0);

    Mat edges = Mat(first.rows, first.cols, CV_8UC1);

    substract_grays(first, second, edges);


    // imshow("DifferencesMapping from the current", edges);
    // waitKey(0);

    Mat current_frame = Mat(frame.rows, frame.cols, CV_8UC3);
    Mat previous_frame = Mat(frame.rows, frame.cols, CV_8UC3);
    cout << "Frame size: " << frame.size() << endl;

    
    cap >> previous_frame;

    Mat differences = Mat(frame.rows, frame.cols, CV_8UC1);
    Mat CannyEdge = Mat(frame.rows, frame.cols , CV_8UC1);
    Mat grayImage;
    Mat differences2 = Mat(frame.rows, frame.cols, CV_8UC1);
    Mat differences3 = Mat(frame.rows, frame.cols, CV_8UC3);
    Mat differences4 = Mat(frame.rows, frame.cols, CV_8UC4);
    Mat differences5 = Mat(frame.rows, frame.cols, CV_8UC1);


    Mat current_frame2 = Mat(frame.rows, frame.cols, CV_8UC3);



    // These variable for the MOG2 algorithm
    float alpha = 0.01f; int K = 5; float T = 0.7f; float initialVar = 900.0f;
    
    int average = 0;
    int average_2 = 0;

    
    bool is_human = false;
    bool is_human_2 = false;

    Mat gray = Mat(frame.rows, frame.cols, CV_8UC1);

    

    while (true) {
        // Capture a new frame from the video
        // for (int i = ; i < 2; i++) {
        cap >> current_frame;
        cap >> current_frame2;
        cap >> current_frame;

        gray = Grayscale(current_frame);
        

        GaussianBlurCustom(current_frame, current_frame.rows, current_frame.cols, 5, 1.4);

        take_differences(current_frame,previous_frame, differences3);
        differences4= Grayscale(differences3);
        
        ApplyThreshold(differences4, differences5, 30, 255);
        Erosion(differences5, differences5.rows, differences5.cols);

        // ApplyThreshold3Channel(differences3, differences4, Vec3b(150, 150, 150), Vec3b(255, 255, 255));

        // CustomMOG2(god, god2, 0.01f, 5, 0.7f, 900.0f);
        // MOG2(gray, differences5, alpha, K, T, initialVar);

        CannyEdgeDetection(current_frame, current_frame.rows, current_frame.cols, 80, 140, CannyEdge);
        
        // Erosion(CannyEdge, CannyEdge.rows, CannyEdge.cols);

        // substract the mapping frame to current frame to lefting human edges alone
        substract_grays(CannyEdge, second, differences);

        // Camera cannot be placed certain position so we need to eliminate the sides
        eliminate_sides(differences, differences.rows, differences.cols, 10, differences2);

        for (int i = 0; i < 3; i++) {
            Dilation(differences2, differences2.rows, differences2.cols);
        }
        

        average = count_boxes(differences2, ((differences2.rows)/2 ) -5 - 400 , ((differences2.rows)/2 )+5 - 400);

        average_2 = count_boxes(differences2, ((differences2.rows)/2 )-5 + 400, ((differences2.rows)/2 )+5 + 400);

        
        cout << "average: " << average << endl;
        cout << "average_2: " << average_2 << endl;

        if (is_human == false && average > 300 ) {
            cout << "Human detected" << endl;
            is_human = true;
        } else if (is_human == true && average < 100) {
            cout << "Human left" << endl;
            is_human = false;
            counter++;
        }


        if (is_human_2 == false && average_2 > 300 ) {
            cout << "Human detected" << endl;
            is_human_2 = true;
        } else if (is_human_2 == true && average_2 < 100 ) {
            cout << "Human left" << endl;
            is_human_2 = false;
            counter--;
        }


        cout << "Counter: " << counter << endl;


        cout << "Counter: " << counter << endl;



        imshow("Final Frame", differences2);
        


        current_frame.copyTo(previous_frame);
        // counter += 1;
        // Exit the loop if the user presses the 'q' key
        if (waitKey(15) == 'q') {
            break;
        }
    }

    // Release the video file and close all OpenCV windows
    cap.release();
    destroyAllWindows();

    return 0;
}






        