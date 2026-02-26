/*
  Bruce A. Maxwell

  Set of utility functions for computing features and embeddings
*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy,
                        float theta, float minE1, float maxE1, float minE2, float maxE2, int debug);
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);