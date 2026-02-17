/*
    Hyuk Jin Chung
    2/16/2026
    Displays live video by looping over frames and applies various filters based on user's key press
*/

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"

// Converts the hsv image into a greyscale image with saturation darkening for easier segmentation
// Subtracts the Saturation from the Value to make colorful objects darker (only background should be white)
// Args: 8-bit color hsv image
// Return: 8-bit saturation darkened greyscale image
void darken(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *sPtr = src.ptr<cv::Vec3b>(i); // row pointer for src image
        uchar *dPtr = dst.ptr<uchar>(i);         // row pointer for dst image
        for (int j = 0; j < dst.cols; j++)
        {
            // pixel = V - 0.5S
            float val = sPtr[j][2] - 0.5f * sPtr[j][1];
            val = (val < 0) ? 0 : val; // clamp values
            dPtr[j] = (uchar)val;
        }
    }
}

// Creates a binary image using the greyscale input and the threshold value
// Returns a segmented binary image with values 0 or 255
void binImage(cv::Mat &src, uchar threshold, cv::Mat &dst)
{
    // create a dst image the same type and size as src
    dst.create(src.size(), src.type());

    for (int i = 0; i < dst.rows; i++)
    {
        uchar *sPtr = src.ptr<uchar>(i); // row pointer for src image
        uchar *dPtr = dst.ptr<uchar>(i); // row pointer for dst image
        for (int j = 0; j < dst.cols; j++)
        {
            // background is black and object is white (inverted)
            dPtr[j] = sPtr[j] < threshold ? 255 : 0;
        }
    }
}

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev;

    // open the video device (0 uses the default camera on the device)
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get size properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d x %d\n", refS.width, refS.height);

    cv::namedWindow("Live Video", cv::WINDOW_AUTOSIZE);      // original video feed
    cv::namedWindow("Intensity Video", cv::WINDOW_AUTOSIZE); // intensity (saturation darkened) feed
    cv::namedWindow("Binary Video", cv::WINDOW_AUTOSIZE);    // binary image feed
    cv::Mat src;                                             // initial RGB frame
    cv::Mat hsv;                                             // hsv frame
    cv::Mat blur;                                            // Gaussian blur frame
    cv::Mat intensity;                                       // saturation darkened image
    cv::Mat dst;                                             // final frame to be displayed

    for (;;) // infinite loop until break
    {
        *capdev >> src; // get a new frame from the camera, treat as a stream
        if (src.empty())
        {
            printf("Frame is empty\n");
            break;
        }

        // applies a 5x5 Gaussian blur and converts the image to HSV
        cv::GaussianBlur(src, blur, cv::Size(5, 5), 0);
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

        // darken the saturated parts of the image to allow for easier thresholding
        darken(hsv, intensity);

        // find the binary image threshold using k means clustering (k = 2)
        uchar threshold = kmeans_threshold(intensity);
        // uchar threshold = 100; // simple constant threshold for testing

        // create a binary image using a threshold
        binImage(intensity, threshold, dst);

        cv::imshow("Live Video", src);
        cv::imshow("Intensity Video", intensity);
        cv::imshow("Binary Video", dst);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
    }

    delete capdev;
    return (0);
}