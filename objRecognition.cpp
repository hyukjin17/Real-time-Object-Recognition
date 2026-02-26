/**
 * Hyuk Jin Chung
 * 2/16/2026
 *
 * Displays live video and creates a segmented binary image (background/foreground)
 * Collects training data using user input and saves labeled feature vectors into a csv file
 * Can classify objects based on training data in real time
 * -i flag (with an image filename) can be set to analyze an image instead of a video feed
 */

#include <map>
#include "features.hpp"
#include "csv.hpp"

// global database of known objects
std::vector<TrainingData> object_db;
// standard deviations for each feature dimension (calculated from DB)
std::vector<float> db_stdevs;
// Global color palette (so colors don't flicker/change randomly every frame)
std::vector<cv::Vec3b> color_palette;
// Map: Actual Label -> (Predicted Label -> Count)
std::map<std::string, std::map<std::string, int>> confusion_matrix;

// Returns the label of the nearest neighbor
std::string classify_object(const RegionFeatures &f)
{
    if (object_db.empty())
        return "Unknown";

    // Create a vector for the current object to match the DB format
    std::vector<float> current_vec = {
        (float)f.percent_filled,
        (float)f.aspect_ratio,
        (float)f.hu_moments[0],
        (float)f.hu_moments[1]};

    std::string best_label = "Unknown";
    float min_dist = 1e9; // start with a huge distance

    for (const auto &obj : object_db)
    {
        float dist = 0.0f;

        // calculate scaled Euclidean distance
        // d = sqrt(sum(((x_i - y_i) / stdev_i)^2))
        for (size_t i = 0; i < current_vec.size(); i++)
        {
            float diff = (current_vec[i] - obj.feature_vector[i]);
            // divide by the standard deviation to scale the distance values properly
            float weighted_diff = diff / db_stdevs[i];
            dist += weighted_diff * weighted_diff;
        }

        dist = std::sqrt(dist);

        if (dist < min_dist)
        {
            min_dist = dist;
            best_label = obj.label;
        }
    }

    // set a threshold for a distance
    // if the closest match is still too far, label as "Unknown"
    if (min_dist > 1.0f)
    {
        return "Unknown";
    }

    return best_label;
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
    char bin_filename[256];
    char reg_filename[256];
    char box_filename[256];

    bool classification_mode = false;
    load_db("object_data.csv"); // load existing data on startup (if exists)

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

        if (classification_mode && !object_db.empty())
        {
            for (auto &region : regions)
            {
                std::string name = classify_object(region.features);

                cv::putText(vis, name, region.centroid - cv::Point2d(50, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            }
        }

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
        else if (key == 'c') // toggle classification mode on/off
        {
            classification_mode = !classification_mode;
            printf("Classification Mode: %s\n", classification_mode ? "ON" : "OFF");
        }
        else if (key == 'n') // save new training data
        {
            if (regions.size() == 1) // only save data if there is one object on screen
            {
                printf("Enter label for this object: ");
                std::string label;
                std::cin >> label; // type the label name in the terminal to tag object

                // pass the precomputed features to the DB
                save_to_db("object_data.csv", label, regions[0].features);
                load_db("object_data.csv"); // reload immediately to update
            }
            else
            {
                printf("Error: Please make sure only one object is on screen\nCurrently seeing %zu objects\n", regions.size());
            }
        }
        else if (key == 'e') // classify object on screen and compare to actual label provided by user (updates confusion matrix)
        {
            if (regions.size() == 1)
            {
                // classify object on screen
                std::string predicted_label = classify_object(regions[0].features);

                // prompt user for the actual label
                printf("\n--- EVALUATION MODE ---\n");
                printf("System predicted: [%s]\n", predicted_label.c_str());
                printf("Enter ACTUAL label: ");

                std::string actual_label;
                std::cin >> actual_label;

                // update confusion matrix
                confusion_matrix[actual_label][predicted_label]++;
                printf("Recorded: Actual [%s] -> Predicted [%s]\n", actual_label.c_str(), predicted_label.c_str());
            }
            else
            {
                printf("Error: Please make sure only one object is on screen\n");
            }
        }
        else if (key == 'p') // print out confusion matrix
        {
            // Print the Confusion Matrix to the terminal
            printf("\nCONFUSION MATRIX\n");
            printf("%-15s", "Actual / Pred"); // Top-left corner

            // Extract all unique labels we've seen (both actual and predicted) to build the headers
            std::vector<std::string> all_labels;
            for (const auto &row : confusion_matrix)
            {
                if (std::find(all_labels.begin(), all_labels.end(), row.first) == all_labels.end())
                    all_labels.push_back(row.first);
                for (const auto &col : row.second)
                {
                    if (std::find(all_labels.begin(), all_labels.end(), col.first) == all_labels.end())
                        all_labels.push_back(col.first);
                }
            }

            // Print Column Headers
            for (const auto &label : all_labels)
            {
                printf("%-15s", label.c_str());
            }
            printf("\n");

            // Print Rows
            for (const auto &actual : all_labels)
            {
                printf("%-15s", actual.c_str()); // Row header
                for (const auto &predicted : all_labels)
                {
                    // Print the count, defaulting to 0 if it doesn't exist
                    printf("%-15d", confusion_matrix[actual][predicted]);
                }
                printf("\n");
            }
        }
    }

    if (capdev)
        delete capdev;
    cv::destroyAllWindows();
    return (0);
}