/**
 * Hyuk Jin Chung
 * 2/22/2026
 *
 * Utility function to read/write to a csv file for labeled feature vectors
 */

#include "csv.hpp"

// Save (append) the object label and feature vector to a csv
void save_to_db(const std::string &filename, const std::string &label, const RegionFeatures &f)
{
    std::ofstream file;
    // open file in append mode ("app") to avoid overwriting previous data
    file.open(filename, std::ios_base::app);

    if (file.is_open())
    {
        // add user provided label
        file << label << ",";
        // add feature vector
        file << f.percent_filled << ",";
        file << f.aspect_ratio << ",";
        file << f.hu_moments[0] << ",";
        file << f.hu_moments[1] << "\n"; // only using the first 2 Hu moments

        file.close();
        printf("Saved '%s' to database\n", label.c_str());
    }
    else
    {
        printf("Error: Could not open database file\n");
    }
}

// Load the feature vectors from a csv file into the local database
void load_db(const std::string &filename)
{
    // clear the database first
    object_db.clear();
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open())
    {
        printf("No database found\n");
        return;
    }

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line); // turns every line in the csv as a stream
        std::string segment;        // buffer for the feature vector values (floats)
        TrainingData data;

        std::getline(ss, data.label, ','); // reads the label first
        while (std::getline(ss, segment, ','))
        {
            data.feature_vector.push_back(std::stof(segment)); // adds each subsequent number into the feature vector
        }

        object_db.push_back(data); // saves the TrainingData object into the DB
    }
    printf("Loaded %lu vectors from database.\n", object_db.size());

    // return if nothing was loaded (e.g. csv was empty)
    if (object_db.empty())
        return;

    int num_features = object_db[0].feature_vector.size();
    db_stdevs.assign(num_features, 0.0f); // fill with 0s initially

    // loop through each feature dimension
    for (int i = 0; i < num_features; i++)
    {
        // extract all values for this specific feature column across all objects
        std::vector<float> feature_column;
        for (const auto &obj : object_db)
        {
            feature_column.push_back(obj.feature_vector[i]);
        }

        // calculate mean and stdev using OpenCV command
        cv::Mat mean, stdev;
        cv::meanStdDev(feature_column, mean, stdev);

        // store the result in the global stdev vector
        db_stdevs[i] = (float)stdev.at<double>(0, 0); // convert double to float first

        // prevent division by zero during classification
        // if there is only 1 training sample, stdev will be 0.
        if (db_stdevs[i] < 1e-5f)
        {
            db_stdevs[i] = 1.0f;
        }
    }
}