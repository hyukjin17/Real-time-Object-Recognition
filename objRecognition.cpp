/*
    Hyuk Jin Chung
    2/16/2026
    Displays live video by looping over frames and creates a segmented binary image (background/foreground)
    -i flag (with an image filename) can be set to analyze an image instead of a video feed
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
            // square the normalized saturation to avoid darkening shadows on white background
            // allows the program to segment the object better (not count the shadow as object)
            float multiplier = (sPtr[j][1] / 255.0f) * (sPtr[j][1] / 255.0f);
            float val = sPtr[j][2] * (1.0f - multiplier);
            // // pixel = V - 0.5S
            // float val = sPtr[j][2] - 0.5f * sPtr[j][1];
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

// Uses the histogram of intensity values to find the 2 means (foreground and background)
// Much faster than using individual pixel values for the k means clustering
// Threshold is defined as the average of the 2 means
// Returns threshold value (uchar)
uchar kmeans_threshold(cv::Mat &src)
{
    // set initial means
    float m1 = 0.0f;
    float m2 = 255.0f;
    int hist[256] = {0};
    int maxIterations = 20;

    // create histogram of 256 intensity values
    for (int i = 0; i < src.rows; i++)
    {
        uchar *ptr = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++)
        {
            hist[ptr[j]]++;
        }
    }

    // only repeat until the means are found OR max iterations is reached
    for (int i = 0; i < maxIterations; i++)
    {
        float sum1 = 0, count1 = 0;
        float sum2 = 0, count2 = 0;

        // for every histogram bin (intensity level)
        for (int j = 0; j < 256; j++)
        {
            if (hist[i] == 0)
                continue; // skip empty bins

            // 1D distance metric
            float d1 = std::abs(i - m1);
            float d2 = std::abs(i - m2);

            if (d1 < d2) // closer to m1
            {
                sum1 += i * hist[i]; // Sum of all pixel values
                count1 += hist[i];   // Total count
            }
            else // closer to m2
            {
                sum2 += i * hist[i];
                count2 += hist[i];
            }
        }

        // calculate the new means
        // sum of all pixel values / # of pixels = new means
        float new_m1 = (count1 > 0) ? sum1 / count1 : m1;
        float new_m2 = (count2 > 0) ? sum2 / count2 : m2;

        // break if means converge
        if (std::abs(new_m1 - m1) < 0.5f && std::abs(new_m2 - m2) < 0.5f)
        {
            m1 = new_m1;
            m2 = new_m2;
            break;
        }

        // update means
        m1 = new_m1;
        m2 = new_m2;
    }

    // return the midpoint of the two means
    return (uchar)((m1 + m2) / 2.0f);
}

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev = nullptr;
    cv::Mat src, dst;             // initial RGB frame and final binary image
    cv::Mat hsv, blur, intensity; // hsv, Gaussian blur, saturation darkened image
    bool image_mode = false;      // default is video mode
    char *img_filepath = nullptr;

    // used for saving the output image
    int image_counter = 1;
    char filename[256];

    // parse image filepath if exists
    for (int i = 1; i < argc; i++)
    {
        // Check for "-i" flag followed by a filename
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
        {
            image_mode = true;
            img_filepath = argv[i + 1];
            printf("Image Mode Enabled. Loading: %s\n", img_filepath);
            i++; // Skip the filename in the loop
        }
    }

    // initialize the source (camera or image)
    if (image_mode)
    {
        src = cv::imread(img_filepath);
        if (src.empty())
        {
            printf("Error: Could not load image %s\n", img_filepath);
            return -1;
        }
    }
    else
    {
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
    }

    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);            // original video feed
    cv::namedWindow("Intensity Output", cv::WINDOW_AUTOSIZE); // intensity (saturation darkened) feed
    cv::namedWindow("Binary Output", cv::WINDOW_AUTOSIZE);    // binary image feed

    for (;;) // infinite loop until break
    {
        // only for video mode
        if (!image_mode)
        {
            *capdev >> src; // get a new frame from the camera, treat as a stream
            if (src.empty())
            {
                printf("Frame is empty\n");
                break;
            }
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

        // 5x5 kernel for the morphological operations
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);  // remove noise
        cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, element); // remove holes

        cv::imshow("Input", src);
        cv::imshow("Intensity Output", intensity);
        cv::imshow("Binary Output", dst);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
        else if (key == 's')
        {
            // save binary image result if 's' is pressed
            snprintf(filename, sizeof(filename), "binary_result%03d.jpg", image_counter);
            cv::imwrite(filename, dst);
            printf("Saved %s\n", filename);
        }
    }

    if (capdev)
        delete capdev;
    cv::destroyAllWindows();
    return (0);
}