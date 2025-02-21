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


int perform_dilation(cv::Mat &erosion_op, cv::Mat &dilation_op, cv::Mat &kernel){

    cv::copyMakeBorder(erosion_op, erosion_op, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    erosion_op.copyTo(dilation_op);

    for(int i=1; i<erosion_op.rows+1; i++){
        for(int j=1; j<erosion_op.cols+1; j++){
            if(erosion_op.at<uchar>(i, j) == 1){
                dilation_op.at<uchar>(i-1, j-1) = std::min(erosion_op.at<uchar>(i-1, j-1) + kernel.at<uchar>(0, 0), 1);
                dilation_op.at<uchar>(i-1, j) = std::min(erosion_op.at<uchar>(i-1, j) + kernel.at<uchar>(0, 1), 1);
                dilation_op.at<uchar>(i-1, j+1) = std::min(erosion_op.at<uchar>(i-1, j+1) + kernel.at<uchar>(0, 2), 1);

                dilation_op.at<uchar>(i, j-1) = std::min(erosion_op.at<uchar>(i, j-1) + kernel.at<uchar>(1, 0), 1);
                dilation_op.at<uchar>(i, j) = std::min(erosion_op.at<uchar>(i, j) + kernel.at<uchar>(1, 1), 1);
                dilation_op.at<uchar>(i, j+1) = std::min(erosion_op.at<uchar>(i, j+1) + kernel.at<uchar>(1, 2), 1);

                dilation_op.at<uchar>(i+1, j-1) = std::min(erosion_op.at<uchar>(i+1, j-1) + kernel.at<uchar>(2, 0), 1);
                dilation_op.at<uchar>(i+1, j) = std::min(erosion_op.at<uchar>(i+1, j) + kernel.at<uchar>(2, 1), 1);
                dilation_op.at<uchar>(i+1, j+1) = std::min(erosion_op.at<uchar>(i+1, j+1) + kernel.at<uchar>(2, 2), 1);
            }
        }
    }

    dilation_op(cv::Rect(1, 1,dilation_op.cols - 2 ,dilation_op.rows - 2));
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
    
    //Dilation
    for (int i=0; i<rounds; i++){
        // perform_dilation(erosion_op, binary_filtered, c8_kernel);
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
        cv::Vec3b(40, 93, 173),      // 3rd largest (Brown)
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
        double height_width_ratio, percent_filled, orientation, eccentricity;
        std::vector<double> hu_moments;  // Store Hu moments as double initially
        cv::Rect bounding_box;

        // Compute bounding box features and Hu Moments
        compute_bounding_box_features(region_mask, height_width_ratio, percent_filled, orientation, eccentricity, hu_moments, bounding_box);

        // Write features to file
        fprintf(featureFile, "%s,%.6f,%.6f,%.6f,%.6f", 
            label.c_str(), 
            static_cast<float>(height_width_ratio),   // Feature 1: Height/Width Ratio
            static_cast<float>(percent_filled),       // Feature 2: Percent Filled
            static_cast<float>(orientation),          // Feature 3: Orientation (Axis of Least Central Moment)
            static_cast<float>(eccentricity)          // Feature 4: Eccentricity (Elongation)
        );

        // Append Hu moments to the file
        for (size_t j = 0; j < hu_moments.size(); j++) {
            fprintf(featureFile, ",%.6f", static_cast<float>(hu_moments[j])); // Features 5-11: Hu Moments (7 values)
        }

        // Newline after writing all features
        fprintf(featureFile, "\n");
        fflush(featureFile);

        // Writing only the first valid region (largest one that is not background)
        break; 
    }

    fclose(featureFile); // Close the file before exiting
    return 0;
}


int write_results(std::string &img_name, const std::string &feature_file_path, std::string &closest_label) {
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
    fprintf(featureFile, "%s,%s,%s\n", img_name.c_str(), label.c_str(), closest_label.c_str());
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


std::string get_scaled_euclidean_label(std::vector<float> query_point){

    char filename[] = "./features/training_features.csv";
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;

    // Read the CSV file
    if (read_image_data_csv(filename, filenames, data, 0) != 0) {
        printf("Unable to read training file"); // Exit if file reading fails
        exit(-1);
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

    std::string closest_label;
    for (size_t i = 0; i < data.size(); i++) {
        float distance = scaled_euclidean_distance(query_point, data[i], std_dev);

        if (distance < min_distance) {
            min_distance = distance;
            closest_label = labels[i];
        }
    }
    return closest_label;
}


std::string get_decision_tree_predict_label(std::vector<float> query_point) {
    // Writing conditional statements for the decision tree built
    if (query_point[3] <= 0.585) {  // Feature_4
        return "beanie";
    } else {
        if (query_point[1] <= 0.444) {  // Feature_2
            return "mouse";
        } else {
            if (query_point[7] <= -4.496) {  // Feature_8
                if (query_point[6] <= -4.355) {  // Feature_7
                    return "book";
                } else {
                    return "box";
                }
            } else {
                if (query_point[0] <= 1.789) {  // Feature_1
                    return "perfume";
                } else {
                    return "book";
                }
            }
        }
    }
}


// =========================================================================================================
// Functions to implement features
// =========================================================================================================


// Function to compute region-based features
int compute_bounding_box_features(const cv::Mat &region_mask, double &height_width_ratio, double &percent_filled, double &orientation, double &eccentricity, std::vector<double> &hu_features, cv::Rect &bounding_box) {
    // Compute bounding box
    bounding_box = cv::boundingRect(region_mask);
    height_width_ratio = static_cast<double>(bounding_box.height) / bounding_box.width;
    percent_filled = static_cast<double>(cv::countNonZero(region_mask)) / bounding_box.area();

    // Compute moments
    cv::Moments moments = cv::moments(region_mask, true);
    if (moments.m00 == 0) {
        return -1; // Avoid division by zero if no valid region
    }

    // Compute Axis of Least Central Moment (Orientation)
    orientation = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

    // Compute Eccentricity (Elongation)
    double a = moments.mu20 / moments.m00;
    double b = moments.mu11 / moments.m00;
    double c = moments.mu02 / moments.m00;

    double lambda1 = (a + c) / 2 + sqrt(4 * b * b + (a - c) * (a - c)) / 2;
    double lambda2 = (a + c) / 2 - sqrt(4 * b * b + (a - c) * (a - c)) / 2;
    eccentricity = sqrt(1 - lambda2 / lambda1);

    // Compute Hu Moments (Scale, Rotation, and Translation Invariant Features)
    double hu_moments[7];
    cv::HuMoments(moments, hu_moments);

    // Log transform Hu moments to normalize the scale
    hu_features.clear();
    for (int i = 0; i < 7; i++) {
        hu_features.push_back(-1 * copysign(log10(abs(hu_moments[i]) + 1e-10), hu_moments[i])); // Avoid log(0)
    }

    return 0;
}



int label_region_with_features(const cv::Mat &region_map, int region_id, const cv::Mat &binary_image, cv::Mat &output_image, std::function<std::string(std::vector<float>)> label_function) {
    // Create mask for the specific region
    cv::Mat region_mask = (region_map == region_id);

    // Skip regions touching the border
    for (int row = 0; row < region_mask.rows; ++row) {
        if (region_mask.at<uchar>(row, 0) > 0 || region_mask.at<uchar>(row, region_mask.cols - 1) > 0) {
            return -1;
        }
    }
    for (int col = 0; col < region_mask.cols; ++col) {
        if (region_mask.at<uchar>(0, col) > 0 || region_mask.at<uchar>(region_mask.rows - 1, col) > 0) {
            return -1;
        }
    }

    // Compute features using the updated function
    double height_width_ratio, percent_filled, orientation, eccentricity;
    std::vector<double> hu_moments;  // Store Hu moments as double initially
    cv::Rect bounding_box;

    int status = compute_bounding_box_features(region_mask, height_width_ratio, percent_filled, orientation, eccentricity, hu_moments, bounding_box);
    if (status == -1) {
        return -1;
    }

    // Compute centroid
    cv::Moments moments = cv::moments(region_mask, true);
    double cx = moments.m10 / moments.m00;
    double cy = moments.m01 / moments.m00;

    // Prepare feature vector (Explicitly cast `double` to `float` to avoid narrowing conversion errors)
    std::vector<float> query_point = {static_cast<float>(height_width_ratio), static_cast<float>(percent_filled), static_cast<float>(orientation), static_cast<float>(eccentricity)};

    // Convert Hu moments to float and add to query_point
    for (double hu_value : hu_moments) {
        query_point.push_back(static_cast<float>(hu_value));
    }

    // Draw Features
    cv::circle(output_image, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), 3, cv::Scalar(0, 255, 255), -1);  // Centroid
    cv::rectangle(output_image, bounding_box, cv::Scalar(255, 0, 0), 2);  // Bounding box

    // Draw orientation axis
    cv::Point2f axis_end(cx + 50 * cos(orientation), cy + 50 * sin(orientation));
    cv::line(output_image, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), axis_end, cv::Scalar(0, 255, 0), 2);

    // Display feature values
    std::ostringstream feature_text;
    feature_text << "Filled: " << std::fixed << std::setprecision(2) << percent_filled * 100 << "%"
    << ", Ecc: " << std::fixed << std::setprecision(2) << eccentricity;

    int text_y = std::max(10, bounding_box.y - 10);
    cv::putText(output_image, feature_text.str(), cv::Point(bounding_box.x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    // Get label using provided function
    std::string label = label_function(query_point);

    // Display label
    cv::putText(output_image, label, cv::Point(bounding_box.x, text_y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    std::vector<std::string> label_names = {"beanie", "mouse", "book", "perfume", "box"};
    if(label==label_names[0]){
        return 0;
    }
    else if(label==label_names[1]){
        return 1;
    }
    else if(label==label_names[2]){
        return 2;
    }
    else if(label==label_names[3]){
        return 3;
    }
    else if(label==label_names[4]){
        return 4;
    }
    else{
        return -1;
    }
}


int get_all_region_data(cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, cv::Mat &binary_image, int max_regions, cv::Mat &output_image, std::function<std::string(std::vector<float>)> label_function) {
    int counter = 0;
    int k;
    int largest_region_label;
    bool is_largest_region_label_set = false;
    for (size_t i = 0; i < region_label_pairs.size() && i <= static_cast<size_t>(max_regions); i++) {
        // Skip background regions (assuming a label value of 0 indicates background)
        if (region_label_pairs[i].second == 0) {
            counter++;
            continue;
        }
        k = label_region_with_features(region_map, region_label_pairs[i].second, binary_image, output_image, label_function);
        if (k == -1) {
            counter++;
            continue; // Skip regions touching the border
        }
        else{
            if(!is_largest_region_label_set){
                largest_region_label = k;
                is_largest_region_label_set = true;
            }
        }
    }

    if (counter >= region_label_pairs.size() || counter >= max_regions) {
        return -1;
    }
    return largest_region_label;
}


int get_largest_region_data(cv::Mat &region_map, std::vector<std::pair<int, int>> &region_label_pairs, cv::Mat &binary_image, int max_regions, cv::Mat &output_image, std::function<std::string(std::vector<float>)> label_function) {
    int counter = 0;
    int k;
    for (size_t i = 0; i < region_label_pairs.size() && i <= static_cast<size_t>(max_regions); i++) {
        // Skip background regions (assuming a label value of 0 indicates background)
        if (region_label_pairs[i].second == 0) {
            counter++;
            continue;
        }
        k = label_region_with_features(region_map, region_label_pairs[i].second, binary_image, output_image, label_function);
        if (k == -1) {
            counter++;
            continue; // Skip regions touching the border
        }
        else{
            return k;
            break;
        }
    }

    if (counter >= region_label_pairs.size() || counter >= max_regions) {
        return -1;
    }
    return k;
}


