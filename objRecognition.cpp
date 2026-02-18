/**
 * Hyuk Jin Chung
 * 2/16/2026
 * Displays live video by looping over frames and creates a segmented binary image (background/foreground)
 * -i flag (with an image filename) can be set to analyze an image instead of a video feed
 */

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"

// A struct to hold computed features
struct RegionFeatures
{
    cv::Point2d centroid;
    double orientation;    // Angle in degrees (0-180)
    double percent_filled; // Ratio of object area to bounding box area
    double aspect_ratio;   // Ratio of width/height (always stays between 0 and 1)
};

// A struct to hold region data for recognition later
struct Region
{
    int id; // valid regions
    int area;
    cv::Vec3b color;
};

// Global color palette (so colors don't flicker/change randomly every frame)
std::vector<cv::Vec3b> color_palette;

// Initializes a random color palette of 256 colors (only does this once at the start of the program)
void init_colors()
{
    if (!color_palette.empty())
        return;
    cv::RNG rng(12345); // arbitrary fixed seed for consistency
    for (int i = 0; i < 256; i++)
    {
        color_palette.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }
}

// Comparator for sorting regions (descending order by area)
bool compareRegionsByArea(const Region &a, const Region &b)
{
    return a.area > b.area;
}

/**
 * Performs Connected Components Analysis (CCA)
 * Filters out small regions and regions touching the border, and visualizes the remaining ones
 * Returns a list of valid Regions sorted by size (largest first)
 */
std::vector<Region> findRegions(cv::Mat &binary_img, cv::Mat &dst_colored, cv::Mat &features, int min_area)
{
    // initialize the color palette with 256 random colors
    init_colors();

    cv::Mat labels, stats, centroids;
    // extract region map from segmented binary image (8-connected, signed int)
    int num_labels = cv::connectedComponentsWithStats(binary_img, labels, stats, centroids, 8, CV_32S);

    std::vector<Region> valid_regions;

    // Create a black output image
    dst_colored = cv::Mat::zeros(binary_img.size(), CV_8UC3);
    int region_count = 1; // valid region counter

    // loop through all found regions (skip background ID 0)
    for (int i = 1; i < num_labels; i++)
    {
        // extract stats for every region
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // ignore small regions or regions touching the image boundary
        if (area < min_area || top <= 2 || left <= 2 || (left + width) >= binary_img.cols - 2 || (top + height) >= binary_img.rows - 2)
            continue;

        // store region data
        Region r;
        r.id = region_count++; // post-increment the valid region counter
        r.area = area;
        r.color = color_palette[i % 255]; // assign color based on region ID (modulo in case there are > 256 regions)

        valid_regions.push_back(r);

        // create a region mask just for this region to be passed into the feature extractor
        cv::Mat region_mask = (labels == i);
        // creates a binary image where only pixels with matching region_id are white (on a black background)

        // compute centroid
        cv::Point2d centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        // compute features for every valid region
        RegionFeatures feats = compute_region_features(region_mask, centroid, features);

        dst_colored.setTo(r.color, region_mask); // use the region mask to color the region
    }

    // sort regions by area (descending order, largest first)
    std::sort(valid_regions.begin(), valid_regions.end(), compareRegionsByArea);

    return valid_regions;
}

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
            // float multiplier = (sPtr[j][1] / 255.0f) * (sPtr[j][1] / 255.0f);
            // float val = sPtr[j][2] * (1.0f - multiplier);

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
            if (hist[j] == 0)
                continue; // skip empty bins

            // 1D distance metric
            float d1 = std::abs(j - m1);
            float d2 = std::abs(j - m2);

            if (d1 < d2) // closer to m1
            {
                sum1 += j * hist[j]; // Sum of all pixel values
                count1 += hist[j];   // Total count
            }
            else // closer to m2
            {
                sum2 += j * hist[j];
                count2 += hist[j];
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
    cv::Mat src, bin, dst;        // initial RGB frame, initial binary image, and final binary image
    cv::Mat hsv, blur, intensity; // hsv, Gaussian blur, saturation darkened image
    cv::Mat clean, vis;
    bool image_mode = false; // default is video mode
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
    if (image_mode) // open the image
    {
        src = cv::imread(img_filepath);
        if (src.empty())
        {
            printf("Error: Could not load image %s\n", img_filepath);
            return -1;
        }
    }
    else // open the video device (0 - default camera)
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