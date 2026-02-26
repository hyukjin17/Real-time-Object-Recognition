/**
 * Hyuk Jin Chung
 * 2/16/2026
 *
 * Functions to preprocess and segment the image into valid regions
 * Image is converted into a binary image using a dynamic threshold
 * Binary image is used to segment the image into multiple valid regions with different colors
 * Scale/translation/rotation invariant features are extracted from each region
 */

#include "features.hpp"

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

// Computes features for a specific region and draws the visualization
// Input: binary image mask with a single segmented region (white object on black background), region centroid
// Output: overlays all the features onto display_dst image
RegionFeatures compute_region_features(const cv::Mat &region_mask, const cv::Point2d centroid, cv::Mat &display_dst)
{
    RegionFeatures features;

    // calculate moments
    // true = binary image (treat non-zero pixels as 1)
    cv::Moments m = cv::moments(region_mask, true);

    // draw centroid
    features.centroid = centroid;
    cv::circle(display_dst, features.centroid, 7, cv::Scalar(0, 255, 0), -1); // green center dot

    // calculate orientation (axis of least central moment)
    // 0.5 * atan2(2 * mu11, mu20 - mu02)
    double theta = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02); // angle in radians
    features.theta_rad = theta;                                   // save angle in radiands to features
    features.orientation = theta * (180.0 / CV_PI);               // convert to degrees

    // visualize axis of least moment of inertia
    // draw a line through the centroid along the orientation angle
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);
    cv::Point2d major_axis(cos_t, sin_t);  // major axis vector
    cv::Point2d minor_axis(-sin_t, cos_t); // minor axis vector

    double line_len = 100.0; // length of the axis line
    cv::line(display_dst, features.centroid - major_axis * line_len,
             features.centroid + major_axis * line_len, cv::Scalar(0, 0, 255), 2); // red major axis line
    cv::line(display_dst, features.centroid - minor_axis * (line_len * 0.3),
             features.centroid + minor_axis * (line_len * 0.3), cv::Scalar(0, 0, 255), 2); // red minor axis line

    // compute the oriented bounding box
    std::vector<cv::Point> points;
    cv::findNonZero(region_mask, points); // finds every non-zero pixel in the region mask
    // initialize 4 extremes for the bounding box
    double max_pos_x = -1e9;
    double max_neg_x = 1e9;
    double max_pos_y = -1e9;
    double max_neg_y = 1e9;
    cv::Point p1, p2, p3, p4; // corner points

    // finds the furthest points from the 2 axes
    for (const auto &p : points)
    {
        // vector from Centroid to Point
        double vec_x = p.x - features.centroid.x;
        double vec_y = p.y - features.centroid.y;
        // find perpendicular distance (dot product with normal vectors)
        double dist_x = (vec_x * major_axis.x) + (vec_y * major_axis.y);
        double dist_y = (vec_x * minor_axis.x) + (vec_y * minor_axis.y);

        // check for new extremes
        if (dist_x > max_pos_x)
            max_pos_x = dist_x;
        if (dist_x < max_neg_x)
            max_neg_x = dist_x;
        if (dist_y > max_pos_y)
            max_pos_y = dist_y;
        if (dist_y < max_neg_y)
            max_neg_y = dist_y;
    }
    // 4 corners of the box
    p1 = features.centroid + (major_axis * max_neg_x) + (minor_axis * max_neg_y);
    p2 = features.centroid + (major_axis * max_pos_x) + (minor_axis * max_neg_y);
    p3 = features.centroid + (major_axis * max_pos_x) + (minor_axis * max_pos_y);
    p4 = features.centroid + (major_axis * max_neg_x) + (minor_axis * max_pos_y);
    // draw the oriented bounding box using the 4 corners
    cv::line(display_dst, p1, p2, cv::Scalar(255, 0, 0), 2);
    cv::line(display_dst, p2, p3, cv::Scalar(255, 0, 0), 2);
    cv::line(display_dst, p3, p4, cv::Scalar(255, 0, 0), 2);
    cv::line(display_dst, p4, p1, cv::Scalar(255, 0, 0), 2);

    // save the bounds to features
    features.minB1 = max_neg_x;
    features.maxB1 = max_pos_x;
    features.minB2 = max_neg_y;
    features.maxB2 = max_pos_y;

    // calculate bbox aspect ratio (height/width ratio)
    double width = max_pos_x - max_neg_x;
    double height = max_pos_y - max_neg_y;
    if (width < height) // make sure the ratio is between (0, 1]
        features.aspect_ratio = width / height;
    else
        features.aspect_ratio = height / width;

    // calculate % filled (object area / bbox area)
    double box_area = width * height;
    features.percent_filled = (box_area > 0) ? (m.m00 / box_area) : 0.0;

    // calculate Hu moments (scale/rotation invariant features)
    // OpenCV computes 7 Hu moments based on normalized central moments
    cv::HuMoments(m, features.hu_moments);

    // log-transform Hu moments to make them normalized (convert exponential scale to linear)
    for (int i = 0; i < 7; i++)
    {
        // sign is flipped to convert to positive values, magnitude is log_10
        // copysign preserves the sign and the entire value is multiplied by -1
        features.hu_moments[i] = -1 * copysign(1.0, features.hu_moments[i]) * log10(abs(features.hu_moments[i]));
    }

    // overlay % filled and aspect ratio for every object for visualization and testing
    char text[100];
    snprintf(text, sizeof(text), "Fill: %.2f / AR: %.2f", features.percent_filled, features.aspect_ratio);
    cv::putText(display_dst, text, features.centroid + cv::Point2d(20, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

    return features;
}

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
std::vector<Region> findRegions(const cv::Mat &binary_img, cv::Mat &dst_colored, cv::Mat &features, int min_area)
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
        r.original_id = i;
        r.area = area;
        r.color = color_palette[i % 255]; // assign color based on region ID (modulo in case there are > 256 regions)
        // compute centroid
        r.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        // create a region mask just for this region to be passed into the feature extractor
        cv::Mat region_mask = (labels == i);
        // creates a binary image where only pixels with matching region_id are white (on a black background)

        // compute features for every valid region
        r.features = compute_region_features(region_mask, r.centroid, features);

        dst_colored.setTo(r.color, region_mask); // use the region mask to color the region

        valid_regions.push_back(r);
    }

    // sort regions by area (descending order, largest first)
    std::sort(valid_regions.begin(), valid_regions.end(), compareRegionsByArea);

    return valid_regions;
}