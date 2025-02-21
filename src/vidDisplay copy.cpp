#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream> // For user input
#include <vector>
#include <algorithm>
#include "utils.h"

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);
    cv::Mat frame;
    char prev_key = 'E';

    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        cv::imshow("Video", frame);

        // Apply Thresholding
        cv::Mat binary;
        threshold_using_KMeans(frame, binary);

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
        display_segmented_regions(region_map, region_label_pairs.size(), region_label_pairs, 100, max_regions, color_region_map);
        cv::imshow("Colored Segmented regions", color_region_map);

        // Compute Features
        std::vector<float> query_point;
        cv::Mat output_image = frame.clone();
        get_region_features(region_map, region_label_pairs, binary_filtered, max_regions, output_image, query_point);

        // Choosing detection method for labelling
        std::string closest_label;
        if(prev_key=='D' || prev_key=='d'){
            if(query_point[0] == -1.0f && query_point[1] == -1.0f){
                cv::putText(output_image, "No Object detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            else{
                closest_label = decision_tree_predict_label(query_point[0], query_point[1]);
                cv::putText(output_image, closest_label, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            cv::putText(output_image, "Using Decision Tree classification", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::imshow("Segmented regions with features", output_image);
        }
        else if(prev_key=='E' || prev_key=='e'){
            // printf("%f, %f", query_point[0], query_point[1]);
            if(query_point[0] == -1.0f && query_point[1] == -1.0f){
                cv::putText(output_image, "No Object detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            else{
                get_closest_label(query_point, closest_label);
                cv::putText(output_image, closest_label, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            cv::putText(output_image, "Using Scaled Euclidean Distance", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::imshow("Segmented regions with features", output_image);
        }


        // Handle Key Press
        char key = cv::waitKey(10);

        if (key == 'q') {
            break; // Quit the program
        } 
        else if (key == 'N' || key == 'n') {
            std::string file_path = "./features/features.csv";
            write_training_feature(file_path, region_map, region_label_pairs, max_regions);
        }
        else if (key == 'S' || key == 's'){
            std::string file_path = "./features/results.csv";
            write_results(file_path, closest_label);
        }
        
        if(key =='D' || key=='E' || key=='e' || key=='d'){
            prev_key=key;
        }
    }

    delete capdev;
    return 0;
}
