/**
 * Hyuk Jin Chung
 * 2/16/2026
 * 
 * Functions to preprocess and segment the image into valid regions
 * Image is converted into a binary image using a dynamic threshold
 * Binary image is used to segment the image into multiple valid regions with different colors
 */

#pragma once
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

// Converts the hsv image into a greyscale image with saturation darkening for easier segmentation
// Subtracts the Saturation from the Value to make colorful objects darker (only background should be white)
// Args: 8-bit color hsv image
// Return: 8-bit saturation darkened greyscale image
void darken(cv::Mat &src, cv::Mat &dst)

// Creates a binary image using the greyscale input and the threshold value
// Returns a segmented binary image with values 0 or 255
void binImage(cv::Mat &src, uchar threshold, cv::Mat &dst)

// Uses the histogram of intensity values to find the 2 means (foreground and background)
// Much faster than using individual pixel values for the k means clustering
// Threshold is defined as the average of the 2 means
// Returns threshold value (uchar)
uchar kmeans_threshold(cv::Mat &src)

/**
 * Performs Connected Components Analysis (CCA)
 * Filters out small regions and regions touching the border, and visualizes the remaining ones
 * Returns a list of valid Regions sorted by size (largest first)
 */
std::vector<Region> findRegions(const cv::Mat &binary_img, cv::Mat &dst_colored, cv::Mat &features, int min_area)