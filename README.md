# Real-time-2D-Object-Recognition - Project #3

## Overview
This project demonstrates a real-time 2D object detection system that can detect upto 5 different objects using a training data and display the respective label, bouding box, orientation axis, filled percentage, eccentricity and the method of detection as output.

---

## Process for object detection
- Thresholding to generate binary image (black for background and white for foreground)
- Morphological filtering for binary image to remove noise and fill spaces.
- Segmentation to identify different regions in the frame of the video stream.
- Calculating features like height-width ratio, filled percentage, orientation axis, eccentricity, Hu moments.
- Build a training data with above features along with the object label assigned by the user.
- Use object detection method of choice to predict a label for the object in focus.

## Object detection methods
- Nearest neighbor method using scaled euclidean method.
- Decision tree classifier using conditional statements.

## Object detection modes
- `Single object detection`: To predict the object label for the largest valid segmented region.
- `Multiple object detection`: To predict the object labels for multiple valid segmented regions.
---

## Project Structure

```
├── bin/
│   ├── #Executable binaries
│
├── features
│   ├── # To store training data feature vectors and result files as well
│
├── include/                                
│   ├── # Includes for external libraries (if any)
│
├── result_images/
│   ├── # The images picked from the video stream for testing object detection. Also has Confusion matrices.
│
├── src/                                    # Source files
│   ├── csv_util.cpp
│   ├── csv_util.h
│   ├── vidDisplay.cpp
│   ├── test.cpp
│   ├── utils.cpp
│   └── utils.h
│
├── .gitignore                              # Git ignore file
├── makefile                                # Build configuration
├── Project3_Report.pdf                     # Project report
```

---

## Tools used
- `OS`: MacOS
- `C++ Compiler`: Apple clang version 16.0.0
- `IDE`: Visual Studio code
- `Camera source`: Iphone (Continuity camera)

---

## Dependencies
- OpenCV

**Note:** Update the dependency paths in the makefile after installation.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

3. Compile the project:
   ```bash
   make vidDisplay
   ```

---

## Running the code

Run the `vidDisplay` file to perform object detection. Below are few modes that can be selected for this system.

### 1. Single object detection mode

```bash
./bin/vidDisplay single
```

### 2. Multiple object detection mode
```bash
./bin/vidDisplay multiple
```

---

## Usage

Key press functionality of OpenCV plays a very important role in this project. Below is all the important key press information required to understand the project.

- `E or e`: To use Nearest neighbor method with scaled euclidean distance metric for object detection. This is the default technique
- `D or d`: To use decision tree classifier method for object detection.
- `N or n`: To create or add to training dataset. I already have a training dataset in the repository (features/training_features.csv). Code modifications have to be made to use your own objects and training data.
- `S or s`: To save the predictions to respective csv file. decision_tree_results.csv for Decision tree classification method and scaled_euclidean_results.csv for Nearest neighbot method. Given there are objects, you can simply create your own results data without any code modification.
- `q`: To stop the program. 

More information about the internal implementation along with outputs is included in **[Project3_Report.pdf](https://github.com/saideep-arikontham/Real-time-2D-Object-Recognition/blob/main/Project3_Report.pdf)**

---

## Highlights
- The `utils.cpp` file includes multiple utility functions like:
    - Thresholding functions
    - Morphological filtering functions
    - Region segmentation functions
    - Feature computing functions
    - Feature displaying functions
    - Data writing functions

- The `csv_util.cpp` file includes utility functions to handle csv files. But the function used is:
    - Utility function to read csv

- The file `test.cpp` is just for testing purposes.

---

## Note

Not using any time travel days.

---

## Contact
- **Name**: Saideep Arikontham
- **Email**: arikontham.s@northeastern,edu