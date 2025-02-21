// =========================================================================================================
// Saideep Arikontham
// February 2025
// CS 5330 Project 3
// =========================================================================================================


#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

#define SSD(a, b) ( ((int)a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) )

//KMeans
int kmeans( std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations=10, int stopThresh=0 );
int threshold_using_KMeans(cv::Mat &frame, cv::Mat &binaryOutput);

//Morphological filtering
int perform_erosion(cv::Mat &binary, cv::Mat &erosion_op, cv::Mat &kernel);
int perform_dilation(cv::Mat &erosion_op, cv::Mat &dialation_op, cv::Mat &kernel);
int perform_morphological_filter(cv::Mat &binary, cv::Mat &binary_filtered, int rounds);

//Segmentation
int analyze_connected_components(const cv::Mat &binary_image, cv::Mat &region_map, cv::Mat &stats, cv::Mat &centroids, std::vector<std::pair<int, int>> &region_label_pairs);
int display_segmented_regions(cv::Mat &region_map, int num_regions, std::vector<std::pair<int, int>> &region_sizes, int size_threshold, int max_regions, cv::Mat &color_region_map);

//Features
int compute_bounding_box_features(const cv::Mat &region_mask, double &height_width_ratio, double &percent_filled, double &orientation, double &eccentricity, std::vector<double> &hu_features, cv::Rect &bounding_box);
int label_region_with_features(const cv::Mat &region_map, int region_id, const cv::Mat &binary_image, cv::Mat &output_image, std::function<std::string(std::vector<float>)> label_function);
int get_all_region_data(cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, cv::Mat &binary_image, int max_regions, cv::Mat &output_image, std::function<std::string(std::vector<float>)> label_function);
int get_largest_region_data(cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, cv::Mat &binary_image, int max_regions, cv::Mat &output_image, std::function<std::string(std::vector<float>)> label_function);

//Training features
int write_training_feature(const std::string &feature_file_path, cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, int max_regions);
int write_results(std::string &img_name, const std::string &feature_file_path, std::string &closest_label);

//Distance metrics and label functions
std::vector<float> compute_std_dev(const std::vector<std::vector<float>>& data);
float scaled_euclidean_distance(const std::vector<float>& point1, const std::vector<float>& point2, const std::vector<float>& std_dev);
std::string get_scaled_euclidean_label(std::vector<float> query_point);
std::string get_decision_tree_predict_label(std::vector<float> query_point);
#endif
