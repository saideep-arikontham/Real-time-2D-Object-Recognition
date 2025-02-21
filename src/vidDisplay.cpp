#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream> // For user input
#include <vector>
#include <algorithm>
#include "utils.h"

int main(int argc, char *argv[]) {
    if(argc < 2 ) {
        printf("usage: %s <single/multiple>\n", argv[0]); // argv[0] is the program name
        exit(-1);
    }

    char mode[256];
    strncpy(mode, argv[1], 255); // single object detection mode or multiple object detection mode

    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    std::string train_file_path = "./features/training_features.csv";
    std::string scaled_euclidean_result_file_path = "./features/scaled_euclidean_results.csv";
    std::string decision_tree_result_file_path = "./features/decision_tree_results.csv";

    cv::namedWindow("Video", 1);
    cv::Mat frame;
    char prev_key = 'E'; //Starting with Object detection with scaled euclidean distance

    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        cv::resize(frame, frame, cv::Size(640, 360));
        cv::imshow("Video", frame);

        // Apply Thresholding
        cv::Mat binary;
        threshold_using_KMeans(frame, binary);
        //cv::imshow("Binary after thresholding", binary);

        // Morphological Filtering
        cv::Mat binary_filtered;
        perform_morphological_filter(binary, binary_filtered, 8);
        cv::imshow("Morphological filtered", binary_filtered);

        // Segmentation
        cv::Mat region_map, stats, centroids;
        std::vector<std::pair<int, int>> region_label_pairs;
        analyze_connected_components(binary_filtered, region_map, stats, centroids, region_label_pairs);

        // Display Segmented Regions
        int max_regions = 5;
        cv::Mat color_region_map(region_map.size(), CV_8UC3, cv::Vec3b(0, 0, 0));
        display_segmented_regions(region_map, region_label_pairs.size(), region_label_pairs, 500, max_regions, color_region_map);
        cv::imshow("Colored Segmented regions", color_region_map);

        // Compute Features
        std::vector<float> query_point;
        cv::Mat output_image = frame.clone();
        

        // Choosing detection method for labelling
        int closest_label;
        if(strcmp(mode, "single")==0){
            if(prev_key=='E' || prev_key=='e'){
                cv::putText(output_image, "Using Scaled Euclidean Distance", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                closest_label = get_largest_region_data(region_map, region_label_pairs, binary_filtered, max_regions, output_image, get_scaled_euclidean_label);
                if(closest_label == -1){
                    cv::putText(output_image, "No Valid Region detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                }

                cv::imshow("Segmented regions with features", output_image);
            }

            else if(prev_key=='D' || prev_key=='d'){
                cv::putText(output_image, "Using Decision Tree classification", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                closest_label  = get_largest_region_data(region_map, region_label_pairs, binary_filtered, max_regions, output_image, get_decision_tree_predict_label);
                if(closest_label == -1){
                    cv::putText(output_image, "No Valid Region detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                }

                cv::imshow("Segmented regions with features", output_image);
            }
        }

        else if(strcmp(mode, "multiple")==0){
            if(prev_key=='E' || prev_key=='e'){
                cv::putText(output_image, "Using Scaled Euclidean Distance", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                closest_label = get_all_region_data(region_map, region_label_pairs, binary_filtered, max_regions, output_image, get_scaled_euclidean_label);
                if(closest_label == -1){
                    cv::putText(output_image, "No Valid Region detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                }

                cv::imshow("Segmented regions with features", output_image);
            }

            else if(prev_key=='D' || prev_key=='d'){
                cv::putText(output_image, "Using Decision Tree classification", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                closest_label  = get_all_region_data(region_map, region_label_pairs, binary_filtered, max_regions, output_image, get_decision_tree_predict_label);
                if(closest_label == -1){
                    cv::putText(output_image, "No Valid Region detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                }

                cv::imshow("Segmented regions with features", output_image);
            }            
        }

        // Handle Key Press
        char key = cv::waitKey(10);

        if (key == 'q') {
            break; // Quit the program
        } 
        else if (key == 'N' || key == 'n') { //write training features
            write_training_feature(train_file_path, region_map, region_label_pairs, max_regions);
        }
        else if (key == 'S' || key == 's'){ // write results with true labels
            std::vector<std::string> label_names = {"beanie", "mouse", "book", "perfume", "box"};
            std::string img_name = "./result_images/img_name_" + std::to_string(std::time(0)) + ".jpg";

            if(prev_key=='E' || prev_key=='e'){
                write_results(img_name, scaled_euclidean_result_file_path, label_names[closest_label]);
            }
            else if(prev_key=='D' || prev_key=='d'){
                write_results(img_name, decision_tree_result_file_path, label_names[closest_label]);
            }
            cv::imwrite(img_name, output_image);
        }
        
        if(key =='D' || key=='E' || key=='e' || key=='d'){
            prev_key=key;
        }
    }

    delete capdev;
    return 0;
}
