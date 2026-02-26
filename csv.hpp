/**
 * Hyuk Jin Chung
 * 2/22/2026
 * 
 * Utility function to read/write to a csv file for labeled feature vectors
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "features.hpp"

// A struct to hold a labeled training example
struct TrainingData {
    std::string label;
    std::vector<float> feature_vector; // for the feature-based classification
    cv::Mat dnn_embedding;             // for the CNN one-shot classification
};

// global database of known objects
extern std::vector<TrainingData> object_db;
// standard deviations for each feature dimension (calculated from DB)
extern std::vector<float> db_stdevs;

// Save (append) the object label and feature vector to a csv
void save_to_db(const std::string &filename, const std::string &label, const RegionFeatures &f);

// Load the feature vectors from a csv file into the local database
void load_db(const std::string& filename);