/**
 * Hyuk Jin Chung
 * 2/16/2026
 *
 * Displays live video and creates a segmented binary image (background/foreground)
 * Collects training data using user input and saves labeled feature vectors into a csv file
 * Can classify objects based on training data in real time
 * -i flag (with an image filename) can be set to analyze an image instead of a video feed
 */

#include "features.hpp"
#include "csv.hpp"

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev = nullptr;
    cv::Mat src, bin, dst;        // initial RGB frame, initial binary image, and final binary image
    cv::Mat hsv, blur, intensity; // hsv, Gaussian blur, saturation darkened image
    cv::Mat clean, vis;
    bool image_mode = false; // default is video mode
    char *img_filepath = nullptr;

    // used for saving the output image
    int image_counter = 1;
    char bin_filename[256];
    char reg_filename[256];
    char box_filename[256];

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
    if (image_mode) // open the image
    {
        src = cv::imread(img_filepath);
        if (src.empty())
        {
            printf("Error: Could not load image %s\n", img_filepath);
            return -1;
        }
    }
    else // open the video device (0 -> default camera)
    {
        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened())
        {
            printf("Unable to open camera device\n");
            return (-1);
        }
    }

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

        src.copyTo(vis); // create a new visualization screen

        // applies a 5x5 Gaussian blur and converts the image to HSV
        cv::GaussianBlur(src, blur, cv::Size(5, 5), 0);
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
        // darken the saturated parts of the image to allow for easier thresholding
        darken(hsv, intensity);

        // find the binary image threshold using k means clustering (k = 2)
        uchar threshold = kmeans_threshold(intensity);
        // uchar threshold = 100; // simple constant threshold for testing

        // create a binary image using a threshold
        binImage(intensity, threshold, bin);

        // 5x5 kernel for the morphological operations
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(bin, dst, cv::MORPH_OPEN, element);  // remove noise
        cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, element); // remove holes

        cv::Mat region_map;
        std::vector<Region> regions = findRegions(dst, region_map, vis, 500); // min object area = 500

        // draw results
        cv::imshow("Regions", region_map); // colorful segmented image
        // cv::imshow("Input", src);                  // original src
        cv::imshow("Intensity Output", intensity); // greyscale
        cv::imshow("Binary Output", dst);          // cleaned up binary image
        cv::imshow("Bounding Box", vis);           // color image with bounding box and features

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
        else if (key == 's')
        {
            // save images if 's' is pressed
            snprintf(bin_filename, sizeof(bin_filename), "binary_result%03d.jpg", image_counter);
            cv::imwrite(bin_filename, dst); // binary image
            snprintf(reg_filename, sizeof(reg_filename), "regions_result%03d.jpg", image_counter);
            cv::imwrite(reg_filename, region_map); // colorful region map
            snprintf(box_filename, sizeof(box_filename), "bbox_result%03d.jpg", image_counter);
            cv::imwrite(box_filename, vis); // bounding box and axis image
            printf("Saved %s, %s, %s\n", bin_filename, reg_filename, box_filename);
        }
    }

    if (capdev)
        delete capdev;
    cv::destroyAllWindows();
    return (0);
}