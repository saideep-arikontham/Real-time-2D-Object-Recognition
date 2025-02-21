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

    cv::namedWindow("Video", 1);
    cv::Mat frame;

    frame = cv::imread("./sample_images/img5P3.png");

    cv::resize(frame, frame, cv::Size(640, 360));
    cv::imshow("Video", frame);

    // Apply Thresholding
    cv::Mat binary;
    threshold_using_KMeans(frame, binary);
    cv::imshow("Binary after thresholding", binary);

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

    int closest_label;
    std::vector<float> query_point;
    cv::Mat output_image = frame.clone();

    cv::putText(output_image, "Using Scaled Euclidean Distance", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    closest_label = get_largest_region_data(region_map, region_label_pairs, binary_filtered, max_regions, output_image, get_scaled_euclidean_label);
    if(closest_label == -1){
        cv::putText(output_image, "No Valid Region detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    cv::imshow("Segmented regions with features", output_image);

    printf("\ndsadascaxasx\n");
    char key_pressed = cv::waitKey(10);

    // Entering loop to wait for a key press - "q" to quit
    while(1){
        key_pressed = cv::waitKey(0); // returns ASCII for pressed key
        if(key_pressed == 113 || key_pressed == 81){ // ASCII for 'q' (113) and 'Q' (81)
            printf("key pressed: %c, terminating\n", static_cast<char>(key_pressed));
            exit(0); // exit the loop and terminate the program
        } 
        else{
            printf("key pressed: %c, continuing\n", static_cast<char>(key_pressed));
        }
    }
}
