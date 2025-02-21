// =========================================================================================================
// Saideep Arikontham
// February 2025
// CS 5330 Project 3
// =========================================================================================================


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "csv_util.h"
#include <algorithm>


// =========================================================================================================
// Functions to implement Thresholding
// =========================================================================================================

//K-means clustering
int kmeans( std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations, int stopThresh ) {

  // error checking
  if( K > data.size() ) {
    printf("error: K must be less than the number of data points\n");
    return(-1);
  }

  // clear the means vector
  means.clear();

  // initialize the K mean values
  // use comb sampling to select K values
  int delta = data.size() / K;
  int istep = rand() % (data.size() % K);
  for(int i=0;i<K;i++) {
    int index = (istep + i*delta) % data.size();
    means.push_back( data[index] );
  }
  // have K initial means

  // loop the E-M steps
  for(int i=0;i<maxIterations;i++) {

    // classify each data point using SSD
    for(int j=0;j<data.size();j++) {
      int minssd = SSD( means[0], data[j] );
      int minidx = 0;
      for(int k=1;k<K;k++) {
	int tssd = SSD( means[k], data[j] );
	if( tssd < minssd ) {
	  minssd = tssd;
	  minidx = k;
	}
      }
      labels[j] = minidx;
    }

    // calculate the new means
    std::vector<cv::Vec4i> tmeans(means.size(), cv::Vec4i(0, 0, 0, 0) ); // initialize with zeros
    for(int j=0;j<data.size();j++) {
      tmeans[ labels[j] ][0] += data[j][0];
      tmeans[ labels[j] ][1] += data[j][1];
      tmeans[ labels[j] ][2] += data[j][2];
      tmeans[ labels[j] ][3] ++; // counter
    }
    
    int sum = 0;
    for(int k=0;k<tmeans.size();k++) {
      tmeans[k][0] /= tmeans[k][3];
      tmeans[k][1] /= tmeans[k][3];
      tmeans[k][2] /= tmeans[k][3];

      // compute the SSD between the new and old means
      sum += SSD( tmeans[k], means[k] );

      means[k][0] = tmeans[k][0]; // update the mean
      means[k][1] = tmeans[k][1]; // update the mean
      means[k][2] = tmeans[k][2]; // update the mean
    }

    // check if we can stop early
    if( sum <= stopThresh ) {
      break;
    }
  }

  // the labels and updated means are the final values

  return(0);
}


// Function to apply thresholds using KMeans
int threshold_using_KMeans(cv::Mat &frame, cv::Mat &binaryOutput) {
    //Random Sampling of Pixels
    std::vector<cv::Vec3b> sampledPixels;
    int totalPixels = frame.rows * frame.cols;
    int sampleCount = totalPixels / 32;

    srand(time(0));
    for (int i = 0; i < sampleCount; i++) {
        int randomIndex = rand() % totalPixels;
        int row = randomIndex / frame.cols;
        int col = randomIndex % frame.cols;
        sampledPixels.push_back(frame.at<cv::Vec3b>(row, col));
    }

    //k-means clustering with K=2
    std::vector<cv::Vec3b> means;
    int *labels = new int[sampledPixels.size()];
    int K = 2, maxIterations = 500, stopThresh = 1;

    if (kmeans(sampledPixels, means, labels, K, maxIterations, stopThresh) != 0) {
        printf("KMeans Failed...");
        delete[] labels;
        return -1;
    }

    delete[] labels;

    //threshold value
    int mean1Intensity = (means[0][0] + means[0][1] + means[0][2]) / 3;
    int mean2Intensity = (means[1][0] + means[1][1] + means[1][2]) / 3;
    int thresholdValue = (mean1Intensity + mean2Intensity) / 2;

    //using threshold value
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::threshold(grayFrame, binaryOutput, thresholdValue, 255, cv::THRESH_BINARY_INV);

    return 0;
}


// =========================================================================================================
// Functions to implement Morphological filtering
// =========================================================================================================


int perform_erosion(cv::Mat &binary, cv::Mat &erosion_op, cv::Mat &kernel){

    cv::copyMakeBorder(binary, binary, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    binary.copyTo(erosion_op);

    for(int i=1; i<binary.rows+1; i++){
        for(int j=1; j<binary.cols+1; j++){
            if(binary.at<uchar>(i, j) == 0){
                erosion_op.at<uchar>(i-1, j-1) = std::max(binary.at<uchar>(i-1, j-1) - kernel.at<uchar>(0, 0), 0);
                erosion_op.at<uchar>(i-1, j) = std::max(binary.at<uchar>(i-1, j) - kernel.at<uchar>(0, 1), 0);
                erosion_op.at<uchar>(i-1, j+1) = std::max(binary.at<uchar>(i-1, j+1) - kernel.at<uchar>(0, 2), 0);

                erosion_op.at<uchar>(i, j-1) = std::max(binary.at<uchar>(i, j-1) - kernel.at<uchar>(1, 0), 0);
                erosion_op.at<uchar>(i, j) = std::max(binary.at<uchar>(i, j) - kernel.at<uchar>(1, 1), 0);
                erosion_op.at<uchar>(i, j+1) = std::max(binary.at<uchar>(i, j+1) - kernel.at<uchar>(1, 2), 0);

                erosion_op.at<uchar>(i+1, j-1) = std::max(binary.at<uchar>(i+1, j-1) - kernel.at<uchar>(2, 0), 0);
                erosion_op.at<uchar>(i+1, j) = std::max(binary.at<uchar>(i+1, j) - kernel.at<uchar>(2, 1), 0);
                erosion_op.at<uchar>(i+1, j+1) = std::max(binary.at<uchar>(i+1, j+1) - kernel.at<uchar>(2, 2), 0);
            }
        }
    }

    erosion_op(cv::Rect(1, 1,erosion_op.cols - 2 ,erosion_op.rows - 2));
    return 0;
}


int perform_dialation(cv::Mat &erosion_op, cv::Mat &dialation_op, cv::Mat &kernel){

    cv::copyMakeBorder(erosion_op, erosion_op, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    erosion_op.copyTo(dialation_op);

    for(int i=1; i<erosion_op.rows+1; i++){
        for(int j=1; j<erosion_op.cols+1; j++){
            if(erosion_op.at<uchar>(i, j) == 1){
                dialation_op.at<uchar>(i-1, j-1) = std::min(erosion_op.at<uchar>(i-1, j-1) + kernel.at<uchar>(0, 0), 1);
                dialation_op.at<uchar>(i-1, j) = std::min(erosion_op.at<uchar>(i-1, j) + kernel.at<uchar>(0, 1), 1);
                dialation_op.at<uchar>(i-1, j+1) = std::min(erosion_op.at<uchar>(i-1, j+1) + kernel.at<uchar>(0, 2), 1);

                dialation_op.at<uchar>(i, j-1) = std::min(erosion_op.at<uchar>(i, j-1) + kernel.at<uchar>(1, 0), 1);
                dialation_op.at<uchar>(i, j) = std::min(erosion_op.at<uchar>(i, j) + kernel.at<uchar>(1, 1), 1);
                dialation_op.at<uchar>(i, j+1) = std::min(erosion_op.at<uchar>(i, j+1) + kernel.at<uchar>(1, 2), 1);

                dialation_op.at<uchar>(i+1, j-1) = std::min(erosion_op.at<uchar>(i+1, j-1) + kernel.at<uchar>(2, 0), 1);
                dialation_op.at<uchar>(i+1, j) = std::min(erosion_op.at<uchar>(i+1, j) + kernel.at<uchar>(2, 1), 1);
                dialation_op.at<uchar>(i+1, j+1) = std::min(erosion_op.at<uchar>(i+1, j+1) + kernel.at<uchar>(2, 2), 1);
            }
        }
    }

    dialation_op(cv::Rect(1, 1,dialation_op.cols - 2 ,dialation_op.rows - 2));
    return 0;
}


// Morphological filtering using Opening Method
int perform_morphological_filter(cv::Mat &binary, cv::Mat &binary_filtered, int rounds){

    binary = binary / 255;
    //binary has thresholded info. 0 for background and 1 for foreground.
    cv::Mat c4_kernel = (cv::Mat_<uchar>(3, 3) <<
        0, 1, 0,
        1, 1, 1,
        0, 1, 0);

     cv::Mat c8_kernel = (cv::Mat_<uchar>(3, 3) <<
        1, 1, 1,
        1, 1, 1,
        1, 1, 1);   

    //Erosion
    cv::Mat erosion_op;
    for (int i=0; i<rounds; i++){
        // perform_erosion(binary, erosion_op, c4_kernel);
        cv::morphologyEx(binary, erosion_op, cv::MORPH_ERODE, c4_kernel);
    }
    
    //Dialation
    for (int i=0; i<rounds; i++){
        // perform_dialation(erosion_op, binary_filtered, c8_kernel);
        cv::morphologyEx(erosion_op, binary_filtered, cv::MORPH_DILATE, c8_kernel);
    }

    binary_filtered = binary_filtered * 255;

    return 0;
}


// =========================================================================================================
// Functions to implement Segmentation
// =========================================================================================================


// Function to perform connected components analysis
int analyze_connected_components(const cv::Mat &binary_image, cv::Mat &region_map, cv::Mat &stats, cv::Mat &centroids, std::vector<std::pair<int, int>> &region_label_pairs) {
    // Perform connected components analysis
    int num_labels = cv::connectedComponentsWithStats(binary_image, region_map, stats, centroids, 8, CV_32S);

    // Collect region sizes and their labels
    for (int i = 0; i < num_labels; ++i) {
        int region_size = stats.at<int>(i, cv::CC_STAT_AREA);
        region_label_pairs.emplace_back(region_size, i);
    }

    // Sort regions by size in descending order
    std::sort(region_label_pairs.begin(), region_label_pairs.end(), std::greater<>());

    return 0;
}

// Function to display segmented regions
int display_segmented_regions(cv::Mat &region_map, int num_regions, std::vector<std::pair<int, int>> &region_sizes, int size_threshold, int max_regions, cv::Mat &color_region_map) {
    // Hardcoded color palette for regions (up to 5 colors)
    std::vector<cv::Vec3b> colors = {
        cv::Vec3b(0, 0, 0),        // Black background
        cv::Vec3b(0, 0, 255),      // 1st largest (red)
        cv::Vec3b(0, 255, 0),      // 2nd largest (green)
        cv::Vec3b(40, 93, 173),      // 3rd largest (blue)
        cv::Vec3b(255, 255, 0),    // 4th largest (yellow)
        cv::Vec3b(255, 0, 255)     // 5th largest (magenta)
    };

    int processed_regions = 0; // Counter for regions processed
    for (int i = 0; i < region_sizes.size(); i++) {
        if (region_sizes[i].second == 0) {
            // Background: Assign black color
            for (int x = 0; x < region_map.rows; x++) {
                for (int y = 0; y < region_map.cols; y++) {
                    if (region_map.at<int>(x, y) == 0) {
                        color_region_map.at<cv::Vec3b>(x, y) = colors[0];
                    }
                }
            }
        } else if (region_sizes[i].first > size_threshold) {
            // Foreground regions above the size threshold
            for (int x = 0; x < region_map.rows; x++) {
                for (int y = 0; y < region_map.cols; y++) {
                    if (region_map.at<int>(x, y) == region_sizes[i].second) {
                        color_region_map.at<cv::Vec3b>(x, y) = colors[processed_regions + 1]; // Cycle through region colors       
                    }
                }
            }
            processed_regions++; // Increment regions processed
        }

        if (processed_regions >= max_regions) {
            break; // Exit when the maximum number of regions is reached
        }
    }

    return 0;
}


// =========================================================================================================
// Functions to write training features
// =========================================================================================================


int write_training_feature(const std::string &feature_file_path, cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, int max_regions) {
    // Prompt for label
    std::string label;
    std::cout << "Enter label for this object: ";
    std::cin >> label;

    // Open file for appending training data
    FILE *featureFile = fopen(feature_file_path.c_str(), "a");
    if (!featureFile) {
        std::cerr << "Error opening file: " << feature_file_path << " for saving features.\n";
        return -1;
    }

    // Extract features
    for (size_t i = 0; i <= max_regions; i++) {
        int region_id = region_label_pairs[i].second;
        if (region_id == 0) continue; // Ignore background
        
        // Compute features
        cv::Mat region_mask = (region_map == region_id);
        cv::Rect bounding_box;
        double height_width_ratio, percent_filled;
        compute_bounding_box_features(region_mask, height_width_ratio, percent_filled, bounding_box);

        // Save to file using fprintf
        fprintf(featureFile, "%s,%.2f,%.2f\n", label.c_str(), height_width_ratio, percent_filled);
        fflush(featureFile); // Ensure data is written immediately
        break; // Writing only the largest region that is not background
    }
    
    fclose(featureFile); // Close the file before exiting
    return 0;
}


int write_results(const std::string &feature_file_path, std::string &closest_label) {
    // Prompt for label
    std::string label;
    std::cout << "Enter label for this object: ";
    std::cin >> label;

    // Open file for appending training data
    FILE *featureFile = fopen(feature_file_path.c_str(), "a");
    if (!featureFile) {
        std::cerr << "Error opening file: " << feature_file_path << " for saving features.\n";
        return -1;
    }

    // Extract features
    // Save to file using fprintf
    fprintf(featureFile, "%s,%s\n", label.c_str(), closest_label.c_str());
    fflush(featureFile); // Ensure data is written immediately

    
    fclose(featureFile); // Close the file before exiting
    return 0;
}

// =========================================================================================================
// Functions to label objects
// =========================================================================================================


std::vector<float> compute_std_dev(const std::vector<std::vector<float>>& data) {
    int num_features = data[0].size();
    int num_samples = data.size();
    std::vector<float> mean(num_features, 0.0f);
    std::vector<float> std_dev(num_features, 0.0f);

    for (const auto& row : data) {
        for (int i = 0; i < num_features; i++) {
            mean[i] += row[i];
        }
    }
    for (float& m : mean) {
        m /= num_samples;
    }

    for (const auto& row : data) {
        for (int i = 0; i < num_features; i++) {
            std_dev[i] += pow(row[i] - mean[i], 2);
        }
    }
    for (float& s : std_dev) {
        s = sqrt(s / (num_samples - 1)); // Unbiased standard deviation (n-1)
        if (s == 0) s = 1; // Avoid division by zero
    }

    return std_dev;
}


float scaled_euclidean_distance(const std::vector<float>& point1, const std::vector<float>& point2, const std::vector<float>& std_dev) {
    float sum = 0.0f;
    for (size_t i = 0; i < point1.size(); i++) {
        float scaled_diff = (point1[i] - point2[i]) / std_dev[i];
        sum += scaled_diff * scaled_diff;
    }
    return sqrt(sum);
}


int get_closest_label(std::vector<float> query_point, std::string &closest_label){

    char filename[] = "./features/features.csv";
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;

    // Read the CSV file
    if (read_image_data_csv(filename, filenames, data, 0) != 0) {
        return 1; // Exit if file reading fails
    }

    // Convert filenames (char*) to std::vector<std::string>
    std::vector<std::string> labels;
    for (char* fname : filenames) {
        labels.push_back(std::string(fname));
    }

    // Compute standard deviation for each feature
    std::vector<float> std_dev = compute_std_dev(data);

    // Find the closest point using Scaled Euclidean Distance
    float min_distance = std::numeric_limits<float>::max();

    for (size_t i = 0; i < data.size(); i++) {
        float distance = scaled_euclidean_distance(query_point, data[i], std_dev);

        if (distance < min_distance) {
            min_distance = distance;
            closest_label = labels[i];
        }
    }
    return 0;
}


std::string decision_tree_predict_label(float Feature_1, float Feature_2) {
    if (Feature_2 <= 0.48) {
        return "mouse";
    } 
    else {
        if (Feature_2 <= 0.535) {
            if (Feature_1 <= 0.535) {
                if (Feature_2 <= 0.835) {
                    return "box";
                } else {
                    return "perfume";
                }
            } else {
                return "box";
            }
        } else {
            if (Feature_1 <= 0.7) {
                if (Feature_1 <= 0.535) {
                    return "box";
                } else {
                    return "beanie";
                }
            } else {
                if (Feature_1 <= 1.83) {
                    if (Feature_2 <= 0.605) {
                        return "box";
                    } else {
                        if (Feature_1 <= 1.265) {
                            if (Feature_1 <= 0.95) {
                                if (Feature_2 <= 0.66) {
                                    return "beanie";
                                } else {
                                    return "book";
                                }
                            } else {
                                return "beanie";
                            }
                        } else {
                            if (Feature_2 <= 0.845) {
                                return "perfume";
                            } else {
                                return "book";
                            }
                        }
                    }
                } else {
                    return "beanie";
                }
            }
        }
    }
}


// =========================================================================================================
// Functions to implement features
// =========================================================================================================


// Function to compute bounding box features
int compute_bounding_box_features(const cv::Mat &region_mask, double &height_width_ratio, double &percent_filled, cv::Rect &bounding_box) {
    bounding_box = cv::boundingRect(region_mask);
    height_width_ratio = (double)bounding_box.height / bounding_box.width;
    percent_filled = (double)cv::countNonZero(region_mask) / bounding_box.area();

    return 0;
}

// Function to compute region features
int compute_region_features(const cv::Mat &region_map, int region_id, const cv::Mat &binary_image, cv::Mat &output_image, std::vector<float> &query_point) {
    // Mask for the specific region
    cv::Mat region_mask = (region_map == region_id);

    // Check if the region touches the border
    for (int row = 0; row < region_mask.rows; ++row) {
        if (region_mask.at<uchar>(row, 0) > 0 || region_mask.at<uchar>(row, region_mask.cols - 1) > 0) {
            return -1; // Skip regions touching the left or right border
        }
    }
    for (int col = 0; col < region_mask.cols; ++col) {
        if (region_mask.at<uchar>(0, col) > 0 || region_mask.at<uchar>(region_mask.rows - 1, col) > 0) {
            return -1; // Skip regions touching the top or bottom border
        }
    }

    // Ensure the region has enough points
    std::vector<cv::Point> points;
    cv::findNonZero(region_mask, points);

    // Compute moments for the region
    cv::Moments moments = cv::moments(region_mask, true);

    // Calculate centroid
    double cx = moments.m10 / moments.m00;
    double cy = moments.m01 / moments.m00;

    // Compute bounding box features
    double height_width_ratio, percent_filled;
    cv::Rect bounding_box;
    compute_bounding_box_features(region_mask, height_width_ratio, percent_filled, bounding_box);

    // Draw centroid on the output image
    cv::circle(output_image, cv::Point(cx, cy), 3, cv::Scalar(0, 255, 255), -1);

    // Draw bounding box
    cv::rectangle(output_image, bounding_box, cv::Scalar(255, 0, 0), 2);

    // Format the feature text
    std::ostringstream feature_text;
    feature_text << "H/W: " << std::fixed << std::setprecision(2) << height_width_ratio
                 << ", Filled: " << std::fixed << std::setprecision(2) << percent_filled * 100 << "%";

    // Display the text above or below the bounding box
    int text_y = std::max(10, bounding_box.y - 10); // Ensure the text stays within the image
    cv::putText(output_image, feature_text.str(), cv::Point(bounding_box.x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    // Display label of the object on the screen
    query_point.push_back(height_width_ratio);
    query_point.push_back(percent_filled);

    return 0;
}

// Function to compute features for multiple regions
int get_region_features(cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, cv::Mat &binary_image, int max_regions, cv::Mat &output_image, std::vector<float> &query_point) {
    // Compute and display features for each major region
    int counter = 0;
    for (size_t i = 0; i <= max_regions; i++) {
        if (region_label_pairs[i].second == 0) {
            counter++;
            continue; // Skip background
        }
        int k = compute_region_features(region_map, region_label_pairs[i].second, binary_image, output_image, query_point);
        if (k == -1) {
            counter++;
            continue; // Skip regions touching the border
        }
        else if(k==0){
            break;
        }
    }
    
    if(counter >= region_label_pairs.size() || counter>=max_regions){
        query_point.push_back(-1.0f);
        query_point.push_back(-1.0f);
    }
    return 0;
}

