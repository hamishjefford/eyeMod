#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

<<<<<<< Updated upstream
=======
#include "utils.hpp"
#include "ezOptionParser.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

>>>>>>> Stashed changes
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <time.h>

<<<<<<< Updated upstream
=======
#include <thread>
#include <list>
#include <algorithm>

>>>>>>> Stashed changes
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

<<<<<<< Updated upstream
//#include <curses.h>



using namespace cv;

=======


using namespace cv;
>>>>>>> Stashed changes
/** Constants **/


/** Function Headers */
void detectAndDisplay(cv::Mat frame);
<<<<<<< Updated upstream
void paintCalibration(cv::Mat framed);
=======
void paintCalibration();
>>>>>>> Stashed changes
void paintDepthDetails(cv::Mat frame);
void toggleSetup();
void fullScreenCalibration();
void drawCalibCircle();
void moveMouse();
void showEyeGazePoints();
void paintGazeLocation(cv::Mat framed);
<<<<<<< Updated upstream
=======
int displayVideo();
void create_kernel();
>>>>>>> Stashed changes

#pragma mark - initVariables

/** Global variables */

cv::String face_cascade_name = "/Users/hamishjefford/Dropbox/eyeMod/eyeMod/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "eyeMod - Hamish Jefford";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
<<<<<<< Updated upstream
cv::Mat debugImage;
cv::Mat overlayImage;
std::vector<cv::Rect> faces;
=======
cv::Mat *faceImage, videoImage, *firstFrame;
IplImage* backgroundImage;
cv::Mat overlayImage;
std::vector<cv::Rect> faces;
Point detectedFaceLocation;

int screenWidth, screenHeight;
>>>>>>> Stashed changes


cv::Point leftPupil, rightPupil, eyeMiddle;
cv::Point av1, av2, av3, av4, av5, av6, av7, av8, av9;

cv::Point topAv, bottomAv, rightAv, leftAv;

int step = 1;

time_t now;

<<<<<<< Updated upstream
=======
Size fullScreen = Size(screenWidth, screenHeight);
>>>>>>> Stashed changes

int calibStatus = 0;
// 0 = Initial;
// 1 = Too Close;
// 2 = Too Far;
<<<<<<< Updated upstream
// 10 = Calibrated;

float detectedFaceWidth, detectedFaceHeight;

=======
// 10 = DepthCalibrated;
// 11 = GazeCalibrated;

float detectedFaceWidth, detectedFaceHeight;

float scalarLeft, scalarRight, scalarUp, scalarDown, scalarH, scalarV;

>>>>>>> Stashed changes
float scaleX, scaleY;

Vector<Point> calibP1, calibP2, calibP3, calibP4, calibP5, calibP6, calibP7, calibP8, calibP9;

<<<<<<< Updated upstream
=======
Vector<Point> gazePath;

>>>>>>> Stashed changes
float leftDiff = 0;
float rightDiff = 0;
float upDiff = 0;
float bottomDiff = 0;
<<<<<<< Updated upstream

=======
float depthCalc = 0;


int framecount = 0;
CvCapture* videoCap = cvCreateFileCapture("/Users/hamishjefford/Dropbox/eyeMod/12_9 tracking.avi");
IplImage* vidFrame = NULL;
>>>>>>> Stashed changes

//Screen Size
cv::Size s;

cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

<<<<<<< Updated upstream
bool trackingMode, calibrationStep, gazeTrackingStep, fullScreenMode, gazeTracking = false;

std::string calibrationInitText = "Centre your face within the area and press space";

=======
bool trackingMode, calibrationStep, gazeTrackingSetup, gazeTracking, pathShow, vidTime, pathConfig, endVid, faceOffset = false;

std::string calibrationInitText = "Centre your face within the area and press space";

/*-- HEATMAP OPTIONS --*/
int g_kernel_size = 175; //75;
float g_fade_time = 120.0; //2.0;
float g_kernel_intensity = 0.1;
float g_base_intensity = 0.1;
float g_max_transparency = 0.3;
bool g_linear_kernel = false;
bool g_realtime_playback = true;
bool g_print_progress = false;
bool g_show_video = true;
bool g_out_video = false;
string g_out_file;
string g_out_four_cc = "MJPG";

// Required arguments
string g_in_video = "/Users/hamishjefford/Dropbox/eyeMod/12_9 tracking.avi";
string g_data_file;

Mat g_heatmap;
Mat g_kernel;
Mat g_ones;
Mat g_zeros;
Mat g_fade_mat;

//vector<DataPoint> g_data;

// Following are in HSV format
Vec3b g_heat_color1 = Vec3b(0, 255, 255); // Red
Vec3b g_heat_color2 = Vec3b(170, 255, 255); // Blue

>>>>>>> Stashed changes


#pragma mark - functions


void resetCalibration(){

    now = time(0);
<<<<<<< Updated upstream
    char* dt = ctime(&now);
    
    std::cout << "Number of sec since January 1,1970:" << now << std::endl;
    
    printf(dt);
    
    gazeTrackingStep = !gazeTrackingStep;
=======
    
    //char* dt = ctime(&now);
    
    
    gazeTrackingSetup = !gazeTrackingSetup;
>>>>>>> Stashed changes
    
    step = 1;
    
    calibP1.release();
    calibP2.release();
    calibP3.release();
    calibP4.release();
    calibP5.release();
    calibP6.release();
    calibP7.release();
    calibP8.release();
    calibP9.release();
    
<<<<<<< Updated upstream
    
}


void fullScreenCalibration(){
    

      //(2880 x 1800) / 2 = (1440 x 900)
    

//      cv::resizeWindow(main_window_name, 1440, 900);
//      cv::moveWindow(main_window_name, 0, 0);
//    
//      cv::resize(debugImage, debugImage, Size(1440, 900));
    
      fullScreenMode = true;
    
      debugImage.setTo(cv::Scalar(0,0,0));
    
      drawCalibCircle();
    
      imshow(main_window_name, debugImage);
    
    
}


void showEyeGazePoints(){
    
    circle(debugImage, av1, 3, 1234);
    circle(debugImage, av2, 3, 1234);
    circle(debugImage, av3, 3, 1234);
    circle(debugImage, av4, 3, 1234);
    circle(debugImage, av5, 3, 1234);
    circle(debugImage, av6, 3, 1234);
    circle(debugImage, av7, 3, 1234);
    circle(debugImage, av8, 3, 1234);
    circle(debugImage, av9, 3, 1234);
    
    
}

void calculateGazeScale(){
    
    //PRINTS ALL GAZE POSITIONS
    int x1 = 0;
    int y1 = 0;
    
   
    
    //printf("POSITION 1: \n");
    
    auto n = calibP1.size() ;
    auto m = calibP1.size() ;
=======
    gazePath.release();
    
    
}

void fullScreenCalibration(){
    
    
      faceImage->setTo(cv::Scalar(0,0,0));
    
      drawCalibCircle();
    
      imshow(main_window_name, *faceImage);
    
    
}

void showEyeGazePoints(){
    
    circle(*faceImage, av1, 3, 1234);
    circle(*faceImage, av2, 3, 1234);
    circle(*faceImage, av3, 3, 1234);
    circle(*faceImage, av4, 3, 1234);
    circle(*faceImage, av5, 3, 1234);
    circle(*faceImage, av6, 3, 1234);
    circle(*faceImage, av7, 3, 1234);
    circle(*faceImage, av8, 3, 1234);
    circle(*faceImage, av9, 3, 1234);
    
    
}


void heat_point(int x, int y)
{
    // Make sure the coordinates are in bounds
    if (x < 0 || y < 0 || x >= g_heatmap.cols || y >= g_heatmap.rows)
    {
        return;
    }
    
    // Only update a small portion of the matrix
    const int g_kernel_half = g_kernel_size / 2;
    const int fixed_x = x - g_kernel_half;
    const int fixed_y = y - g_kernel_half;
    const int roi_l = max(fixed_x, 0);
    const int roi_t = max(fixed_y, 0);
    const int roi_w = min(fixed_x + g_kernel_size, g_heatmap.cols) - roi_l;
    const int roi_h = min(fixed_y + g_kernel_size, g_heatmap.rows) - roi_t;
    
    Mat roi(g_heatmap(Rect(roi_l, roi_t, roi_w, roi_h)));
    
    const int groi_l = roi_l - fixed_x;
    const int groi_t = roi_t - fixed_y;
    const int groi_w = roi_w;
    const int groi_h = roi_h;
    
    Mat roi_gauss(g_kernel(Rect(groi_l, groi_t, groi_w, groi_h)));
    roi += roi_gauss;
}

//============================================================================
// decrease_heatmap
//============================================================================

/* Fades the entire heatmap by g_fade_mat amount.
 */
void decrease_heatmap()
{
    // Fade some of the values in the matrix
    g_heatmap -= g_fade_mat;
    g_heatmap = max(g_zeros, g_heatmap);

}

//============================================================================
// overlay_heatmap
//============================================================================

/* Draws the heatmap on top of a frame. The frame must be the same size as
 * the heatmap.
 */
void overlay_heatmap(Mat *frame)
{
    

    
    // Make sure all values are capped at one
    g_heatmap = min(g_ones, g_heatmap);
    
    Mat temp_map;
    blur(g_heatmap, temp_map, Size(15, 15));
    
    for (int r = 0; r < frame->rows; ++r)
    {
        Vec3b* f_ptr = frame->ptr<Vec3b>(r);
        float* h_ptr = temp_map.ptr<float>(r);
        for (int c = 0; c < frame->cols; ++c)
        {
            const float heat_mix = h_ptr[c];
            if (heat_mix > 0.0)
            {
                // in BGR
                const Vec3b i_color = f_ptr[c];
                
                const Vec3b heat_color =
                hsv_to_bgr(interpolate_hsv(g_heat_color2, g_heat_color1, heat_mix));
                
                const float heat_mix2 = std::min(heat_mix, g_max_transparency);
                
                const Vec3b final_color = interpolate(i_color, heat_color, heat_mix2);
                
                f_ptr[c] = final_color;
            }
        }
    }
}

//============================================================================
// create_kernel
//============================================================================

/* Create the heatmap kernel. This is applied when heat_point() is called.
 */
void create_kernel()
{
    if (g_linear_kernel)
    {
        // Linear kernel
        const float max_val = 1.0 * g_base_intensity;
        const float min_val = 0.0;
        const float interval = max_val - min_val;
        
        const int center = g_kernel_size / 2 + 1;
        const float radius = g_kernel_size / 2;
        
        g_kernel = Mat::zeros(g_kernel_size, g_kernel_size, CV_32F);
        for (int r = 0; r < g_kernel_size; ++r)
        {
            float* ptr = g_kernel.ptr<float>(r);
            for (int c = 0; c < g_kernel_size; ++c)
            {
                // Calculate the distance from the center
                const float diff_x = static_cast<float>(abs(r - center));
                const float diff_y = static_cast<float>(abs(c - center));
                const float length = sqrt(diff_x*diff_x + diff_y*diff_y);
                if (length <= radius)
                {
                    const float b = 1.0 - (length / radius);
                    const float val = b*interval + min_val;
                    ptr[c] = val;
                }
            }
        }
    }
    else
    {
        // Gaussian kernel
        Mat coeffs = getGaussianKernel(g_kernel_size, 0.0, CV_32F)*150*g_base_intensity;
        g_kernel = coeffs * coeffs.t();
        
    }

}




void calculateGazeScale(){
    
    //PRINTS ALL GAZE POSITIONS
    float x1 = 0;
    float y1 = 0;
    

    
    unsigned long n = calibP1.size() ;
    unsigned long m = calibP1.size() ;
>>>>>>> Stashed changes
    
    for( auto iter = std::begin(calibP1) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av1.x = x1 / m;
    av1.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av1.x, av1.y);
    
    
    //printf("POSITION 2: \n");
=======
    printf("POSITION 1: X:%d, Y:%d \n", av1.x, av1.y);
    
    
>>>>>>> Stashed changes
    
    n = calibP2.size() ;
    m = calibP2.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP2) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av2.x = x1 / m;
    av2.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av2.x, av2.y);
    
    
    //printf("POSITION 3: \n");
=======
    printf("POSITION 2: X:%d, Y:%d \n", av2.x, av2.y);
    
    
    
>>>>>>> Stashed changes
    
    n = calibP3.size() ;
    m = calibP3.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP3) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av3.x = x1 / m;
    av3.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av3.x, av3.y);
    
    
    //printf("POSITION 4: \n");
=======
    printf("POSITION 3: X:%d, Y:%d \n", av3.x, av3.y);
    
    
    
>>>>>>> Stashed changes
    
    n = calibP4.size() ;
    m = calibP4.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP4) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av4.x = x1 / m;
    av4.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av4.x, av4.y);
    
    
    //printf("POSITION 5: \n");
=======
    printf("POSITION 4: X:%d, Y:%d \n", av4.x, av4.y);
    
    
    
>>>>>>> Stashed changes
    
    n = calibP5.size() ;
    m = calibP5.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP5) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av5.x = x1 / m;
    av5.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av5.x, av5.y);
    
    
    //printf("POSITION 6: \n");
=======
    printf("POSITION 5: X:%d, Y:%d \n", av5.x, av5.y);
    
    
    
>>>>>>> Stashed changes
    
    n = calibP6.size() ;
    m = calibP6.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP6) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av6.x = x1 / m;
    av6.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av6.x, av6.y);
    
    
    //printf("POSITION 7: \n");
=======
    printf("POSITION 6: X:%d, Y:%d \n", av6.x, av6.y);
    
    
    
>>>>>>> Stashed changes
    
    n = calibP7.size() ;
    m = calibP7.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP7) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av7.x = x1 / m;
    av7.y = y1 / m;
    
<<<<<<< Updated upstream
   // printf("X:%d, Y:%d \n", av7.x, av7.y);
    
    
    //printf("POSITION 8: \n");
=======
    printf("POSITION 7: X:%d, Y:%d  \n", av7.x, av7.y);
    

>>>>>>> Stashed changes
    
    n = calibP8.size() ;
    m = calibP8.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP8) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av8.x = x1 / m;
    av8.y = y1 / m;
    
<<<<<<< Updated upstream
    //printf("X:%d, Y:%d \n", av8.x, av8.y);
    
    
    
    //printf("POSITION 9: \n");
=======
    printf("POSITION 8: X:%d, Y:%d \n", av8.x, av8.y);

>>>>>>> Stashed changes
    
    n = calibP9.size() ;
    m = calibP9.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP9) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av9.x = x1 / m;
    av9.y = y1 / m;
<<<<<<< Updated upstream
    printf("X:%d, Y:%d \n", av9.x, av9.y);
    

    topAv.y = ((av2.y + av3.y + av6.y) / 3);
    
    bottomAv.y = ((av4.y + av5.y + av9.y) / 3);
    
    rightAv.x = ((av3.x + av5.x + av8.x) / 3);
    
    leftAv.x = ((av2.x + av4.x + av7.x) / 3);
    
=======
    
    printf("POSITION 9: X:%d, Y:%d \n", av9.x, av9.y);
    

    topAv.y = ((av2.y + av3.y + av6.y) / 3);
    bottomAv.y = ((av4.y + av5.y + av9.y) / 3);
    rightAv.x = ((av3.x + av5.x + av8.x) / 3);
    leftAv.x = ((av2.x + av4.x + av7.x) / 3);
    
    printf("Top: %d \n", topAv.y);
    printf("Bottom: %d \n", bottomAv.y);
    printf("Right: %d \n", rightAv.x);
    printf("Left: %d \n", leftAv.x);
    
    printf("Left Dif: %d \n", (av1.x - leftAv.x));
    printf("Right Dif: %d \n", (rightAv.x - av1.x));
    printf("Up Dif: %d \n", (av1.y - topAv.y));
    printf("Down Dif: %d \n", (bottomAv.y - av1.y));
     
    scalarH = (screenWidth) / (av1.x - leftAv.x);
    scalarV = (screenHeight) / (av1.y - topAv.y);
    
    scalarLeft = (screenWidth / 2) / (av1.x - leftAv.x);
    scalarRight = (screenWidth / 2) / (rightAv.x - av1.x);
  
    scalarUp = (screenHeight / 2) / (av1.y - topAv.y);
    scalarDown = (screenHeight / 2) / (bottomAv.y - av1.y);
    
    printf("Scalar Left: %f \n", scalarLeft);
    printf("Scalar Right: %f \n", scalarRight);
    printf("Scalar Up: %f \n", scalarUp);
    printf("Scalar Down: %f \n", scalarDown);
    
    
}


void breaker(){
    
    
    printf("broke");
    
>>>>>>> Stashed changes
    
}

void drawCalibCircle(){
    
<<<<<<< Updated upstream
    s = debugImage.size();
=======
    s = faceImage->size();
>>>>>>> Stashed changes
    
    
    
    switch (step) {
        case 1:
<<<<<<< Updated upstream
            circle(debugImage, Point(s.width/2, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width/2, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width/2, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 2:
            circle(debugImage, Point(30, 130), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(30, 130), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(30, 130), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 3:
            circle(debugImage, Point(s.width - 30, 130), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width - 30, 130), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width - 30, 130), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 4:
            circle(debugImage, Point(30, s.height - 35), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(30, s.height - 35), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(30, s.height - 35), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 5:
            circle(debugImage, Point(s.width - 30, s.height - 35), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width - 30, s.height - 35), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width - 30, s.height - 35), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 6:
            circle(debugImage, Point(s.width / 2, 130), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width / 2, 130), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width / 2, 130), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 7:
            circle(debugImage, Point(30, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(30, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(30, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 8:
            circle(debugImage, Point(s.width - 30, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width - 30, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width - 30, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 9:
            circle(debugImage, Point(s.width / 2, s.height - 35), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width / 2, s.height - 35), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(debugImage, Point(s.width / 2, s.height - 35), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
        
        case 10:
            calculateGazeScale();
            gazeTrackingStep = false;
            gazeTracking = true;
=======
            circle(*faceImage, Point(s.width/2, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width/2, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width/2, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 2:
            circle(*faceImage, Point(30, 130), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(30, 130), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(30, 130), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 3:
            circle(*faceImage, Point(s.width - 30, 130), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width - 30, 130), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width - 30, 130), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 4:
            circle(*faceImage, Point(30, s.height - 35), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(30, s.height - 35), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(30, s.height - 35), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 5:
            circle(*faceImage, Point(s.width - 30, s.height - 35), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width - 30, s.height - 35), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width - 30, s.height - 35), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 6:
            circle(*faceImage, Point(s.width / 2, 130), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width / 2, 130), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width / 2, 130), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 7:
            circle(*faceImage, Point(30, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(30, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(30, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 8:
            circle(*faceImage, Point(s.width - 30, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width - 30, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width - 30, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
            
        case 9:
            circle(*faceImage, Point(s.width / 2, s.height - 35), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width / 2, s.height - 35), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width / 2, s.height - 35), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            break;
        
        case 10:
            circle(*faceImage, Point(s.width/2, s.height/2), 30, Scalar(255,255,255),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width/2, s.height/2), 20, Scalar(0,0,0),CV_FILLED, 8,0);
            circle(*faceImage, Point(s.width/2, s.height/2), 10, Scalar(255,255,255),CV_FILLED, 8,0);
            
        case 11:
            calculateGazeScale();
            gazeTrackingSetup = false;
            gazeTracking = true;
            calibStatus = 11;
>>>>>>> Stashed changes
            
        default:
            break;
    }
    
    
    
}

void paintGazeLocation(  cv::Mat framed   ){
    
    cv::Point gazePoint;
    
<<<<<<< Updated upstream
    float scalarLeft = 720 / (av1.x - leftAv.x);
    float scalarRight = 720 / (rightAv.x - av1.x);
    float scalarUp = 450 / (av1.y - topAv.y);
    float scalarDown = 450 / (bottomAv.y - av1.y);
    
    if(eyeMiddle.x > av1.x){
        
        //Right
        
       gazePoint.x = (eyeMiddle.x - av1.x) * scalarRight;
        
        
    }
    if(eyeMiddle.x < av1.x){
        
        //Left
        
        gazePoint.x = (av1.x - eyeMiddle.x) * scalarLeft;
        
    }
    
    if(eyeMiddle.y > av1.y){
        
        //Bottom
        gazePoint.y = (eyeMiddle.y - av1.y) * scalarDown;
        
    }
    
    if(eyeMiddle.y < av1.y){
        
        //Top
        gazePoint.y = (av1.y - eyeMiddle.y) * scalarUp;

=======

    gazePoint.x = (screenWidth - ((av1.x - eyeMiddle.x)  * scalarLeft));
    gazePoint.y = (screenHeight - ((av1.y - eyeMiddle.y) * scalarDown));
    
    
//    if(eyeMiddle.x > av1.x){
//        
//        //Right
//        
//        gazePoint.x = (screenWidth - ((eyeMiddle.x - av1.x) * scalarRight));
//        
//        //(0 + ((eyeMiddle.x - av1.x) * scalarRight));
//        
//
//        
//        
//    }
//    if(eyeMiddle.x < av1.x){
//        
//        //Left
//        gazePoint.x = (screenWidth - ((av1.x - eyeMiddle.x)  * scalarLeft));
//
//        
//    }
//    
//    if(eyeMiddle.x == av1.x && eyeMiddle.y == av1.y){
//        
//        //Middle
//        
//        gazePoint.x = av1.x;
//        gazePoint.y = av1.y;
//
//        
//    }
//    
//
//    
//    if(eyeMiddle.y > av1.y){
//        
//        //Bottom
//        
//        gazePoint.y = (screenHeight - ((eyeMiddle.y - av1.y) * scalarDown));
//        //gazePoint.y = (0 + ((eyeMiddle.y - av1.y) * scalarDown));
//
//        
//    }
//    
//    if(eyeMiddle.y < av1.y){
//        
//        //Top
//        
//        gazePoint.y = (screenHeight - ((av1.y - eyeMiddle.y) * scalarUp));
//        //gazePoint.y = (eyeMiddle.y - av1.y) * scalarDown;
//
//    }
    
    
    
    if(gazePoint.x >  screenWidth){
        
        gazePoint.x = screenWidth;
        
    }
    
    if(gazePoint.x <  0 && eyeMiddle.x < av1.x){
        
        gazePoint.x = 0;
        
    }
    
    if(gazePoint.y >  screenHeight){
        
        gazePoint.y = screenHeight;
        
    }
    
    if(gazePoint.y <  0 && eyeMiddle.y < av1.y){
        
        gazePoint.y = 100;
        
>>>>>>> Stashed changes
    }
    
    
    Point pt1 =  Point(gazePoint.x - 20, gazePoint.y);
    Point pt2 =  Point(gazePoint.x + 20, gazePoint.y);
    
    //Vertical
    Point pt3 =  Point(gazePoint.x, gazePoint.y - 20);
    Point pt4 =  Point(gazePoint.x, gazePoint.y + 20);
    
<<<<<<< Updated upstream
    line(debugImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
    line(debugImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
    
    circle(debugImage, gazePoint, 3, 1234);
    //circle(debugImage, eyeMiddle, 3, 1234);
    
}

=======
    line(*faceImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
    line(*faceImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
    
    if(!vidTime){
        line(*faceImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
        line(*faceImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
        circle(*faceImage, gazePoint, 3, 1234);
    }
    else{
        line(videoImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
        line(videoImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
        circle(videoImage, gazePoint, 3, 1234);
    }
    
    circle(*faceImage, eyeMiddle, 3, 1234);
    
    circle(*faceImage, av1, 3, 4444);
    
    gazePath.push_back(gazePoint);
    
//    DataPoint newEntry;
//    newEntry.px = gazePoint.x;
//    newEntry.py = gazePoint.y;
//    
//    g_data.push_back(newEntry);
    

    printf("EYE MIDDLE: X: %d, Y: %d \n", eyeMiddle.x, eyeMiddle.y);
    printf("GAZE POINT: X: %d, Y: %d \n", gazePoint.x, gazePoint.y);
    
}
>>>>>>> Stashed changes

void paintDepthDetails(cv::Mat frame){
    
    s = frame.size();
<<<<<<< Updated upstream
    float depthCalc = 0;
    
    cv::Mat calibration_Area = cv::Mat::zeros( s.height, s.width, CV_8UC3 );
=======
    
>>>>>>> Stashed changes
    
    if(faces.size() != 0){
        
      depthCalc =  (faces[0].width * faces[0].height) / (detectedFaceHeight * detectedFaceWidth);
        
    }
    
    std::string depth_message = "Depth: " + std::to_string(depthCalc);
    
<<<<<<< Updated upstream
    putText(debugImage, depth_message, cvPoint(0 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
=======
    putText(*faceImage, depth_message, cvPoint(0 ,150), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
>>>>>>> Stashed changes
    
}


<<<<<<< Updated upstream
void paintCalibration( cv::Mat framed  ){
    
    s = framed.size();
    
    cv::Mat calibration_Area = cv::Mat::zeros( s.height, s.width, CV_8UC3 );
    
    //Horizontal
    Point pt1 =  Point((s.width / 2) - ((s.width / 8)), (s.height / 2));
    Point pt2 =  Point((s.width / 2) + ((s.width / 8)), (s.height / 2));
    
    //Vertical
    Point pt3 =  Point((s.width / 2), (s.height / 2) - (s.height / 5) );
    Point pt4 =  Point((s.width / 2), (s.height / 2) + (s.height / 5));
    
    line(debugImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
    line(debugImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
=======
void paintGazePath(cv::Mat *framePass){
    
    
    if(!pathConfig){
        
    g_heatmap = Mat::zeros(framePass->rows, framePass->cols, CV_32FC1);
    g_ones = Mat::ones(framePass->rows, framePass->cols, CV_32F);
    g_zeros = Mat::zeros(framePass->rows, framePass->cols, CV_32F);
    
    g_fade_mat = Mat::ones(framePass->rows, framePass->cols, CV_32F);
    // Determine how much to fade the heatmap values by each frame
    g_fade_mat.setTo((1.0 / 10) / g_fade_time);
        
    // Create heatmap kernel
    create_kernel();
        
    pathConfig = true;
        
    }
    
    for(int i = 0; i < gazePath.size() - 1; ++i)
    {
        
        
        heat_point(gazePath[i].x, gazePath[i].y);
        decrease_heatmap();
        
    }
    
    
    overlay_heatmap(framePass);
    //decrease_heatmap();
    imshow(main_window_name, *framePass);

    
    
}

void paintCalibration(  ){
    
    s = faceImage->size();
    
    //cv::Mat calibration_Area = cv::Mat::zeros( s.height, s.width, CV_8UC3 );
    
    //Horizontal
    Point pt1 =  Point((s.width / 2) - ((s.width / 5)), (s.height / 2));
    Point pt2 =  Point((s.width / 2) + ((s.width / 5)), (s.height / 2));
    
    //Vertical
    Point pt3 =  Point((s.width / 2), (s.height / 2) - (s.height / 3) );
    Point pt4 =  Point((s.width / 2), (s.height / 2) + (s.height / 3));
    
    line(*faceImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
    line(*faceImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
>>>>>>> Stashed changes
    
    
    //Text Displayed
    switch (calibStatus) {
<<<<<<< Updated upstream
        case 0:
            putText(debugImage, "Centre your face within the area and press space", cvPoint(s.width/2 - s.width /4 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
        
        case 1:
            putText(debugImage, "You're too close. Move back and try again.", cvPoint(s.width/2 - s.width /4 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
            
        case 2:
            putText(debugImage, "You're too far. Move closer and try again.", cvPoint(s.width/2 - s.width /4 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
=======
        case 1:
            putText(*faceImage, "Centre your face within the area and press space", cvPoint(s.width/2  - ((s.width / 8)) ,200), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
        
        case 2:
            putText(*faceImage, "You're too close. Move back and try again.", cvPoint(s.width/2 ,200), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
            
        case 3:
            putText(*faceImage, "You're too far. Move closer and try again.", cvPoint(s.width/2,200), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
>>>>>>> Stashed changes
            break;
            
            
        default:
            break;
    
    }
    
    
<<<<<<< Updated upstream
    imshow(main_window_name, debugImage);
    
}

int main( int argc, const char** argv ) {
  CvCapture* capture;
  cv::Mat frame;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){
      
      printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
      return -1;
  
  };

  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 0, 0);
  cv::resizeWindow(main_window_name, 1440, 900);

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2), 43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
=======
    imshow(main_window_name, *faceImage);
    
}

void resetFaceLoc(){
    
    
    detectedFaceHeight = faces[0].height;
    detectedFaceWidth = faces[0].width;
    detectedFaceLocation.x = faces[0].x;
    detectedFaceLocation.y = faces[0].y;
    
    
}

void checkFaceDepth(){
    
    
    if (depthCalc > 1.3 || depthCalc < 0.8){
        
        faceOffset = true;
        
        
    }
    else{
        
        faceOffset = false;
    }
    
    
}


void toggleSetup(){
    

    if (calibStatus == 1){
        
        int calibWidth = ((s.width / 2) + (s.width / 3)) - ((s.width / 2) - (s.width / 3));
        int calibHeight = ((s.height / 2) + (s.height / 2)) - ((s.height / 2) - (s.height / 2));
        
        face_cascade.detectMultiScale( *faceImage, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
        
        if (faces.size() != 0) {
            
            
            if(((faces[0].height * faces[0].width) <=  (calibWidth * calibHeight)) ){
                
                
                //printf("SUITABLE SIZE \n");
                
                calibStatus = 10;
                
                trackingMode = true;
                calibrationStep = false;
                
                resetFaceLoc();
                

                
            }
            else if(((faces[0].height * faces[0].width) <  (calibWidth/2 * calibHeight/2)) ){
                
                //printf("Too far \n");
                calibStatus = 3;
                
            }
            else if ((faces[0].height * faces[0].width) >  (calibWidth/2 * calibHeight/2)){
                
                //printf("Too close \n");
                calibStatus = 2;
                
            }
            
        }
        
    }else if (calibStatus == 10){
        
        pathShow = false;
        endVid = false;
        resetFaceLoc();
        resetCalibration();
    }
    
    else if (calibStatus == 11){
        
        gazePath.release();
        vidTime = true;
    }
    
    else if (calibStatus == 12){
        
        pathShow = true;
        calibStatus = 13;
    }
    
    else if (calibStatus == 13){
        
        vidTime = false;
        resetCalibration();
    }
    
    if(calibStatus == 0 && !trackingMode){
        //calibrationStep = true;
        calibStatus = 1;
    }
    
    
}


int displayVideo(){
    

    CvCapture* videoCap = cvCreateFileCapture("/Users/hamishjefford/Dropbox/eyeMod/12_9 tracking.avi");
    
    IplImage* frame = NULL;
    
    if(!videoCap)
    {
        printf("Video Not Opened\n");
        return -1;
    }
    
    int width = (int)cvGetCaptureProperty(videoCap,CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)cvGetCaptureProperty(videoCap,CV_CAP_PROP_FRAME_HEIGHT);
    double fps = cvGetCaptureProperty(videoCap, CV_CAP_PROP_FPS);
    int frame_count = (int)cvGetCaptureProperty(videoCap,  CV_CAP_PROP_FRAME_COUNT);
    
    printf("Video Size = %d x %d\n",width,height);
    printf("FPS = %f\nTotal Frames = %d\n",fps,frame_count);
    
    while(1)
    {
        frame = cvQueryFrame(videoCap);
        
        if(!frame)
        {
            printf("Capture Finished\n");
            break;
        }
        
        cvShowImage("video",frame);
        cvWaitKey(10);
    }
    
    cvReleaseCapture(&videoCap);
    return 0;
    
    
}

void videoPlayback(){
    
                vidFrame = cvQueryFrame(videoCap);
    
    
    
    
                if (vidFrame){
    
    
                    backgroundImage = cvCreateImage(Size(screenWidth, screenHeight), vidFrame->depth, vidFrame->nChannels);
                    cvSet(backgroundImage, cvScalar(0,0,0));
                    cvSetImageROI(backgroundImage, cvRect((screenWidth/2) - vidFrame->width /2, (screenHeight / 2) - vidFrame->height /2, vidFrame->width, vidFrame->height));
    
                    cvAddWeighted(backgroundImage, 1, vidFrame, 1, 1, backgroundImage);
                    cvResetImageROI(backgroundImage);
                    videoImage = Mat(backgroundImage, true);
    
                    if (framecount == 0){
    
                        *firstFrame = Mat(backgroundImage, true);
    
                    }
    
                    if (gazeTracking){
    
                        paintGazeLocation(*faceImage);

                    
                        
                    }
    
                    imshow(main_window_name, videoImage);
    
    
    
                }else{
                    //Video finished remove window
    
                    vidTime = false;
                    endVid = true;
                    pathShow = true;
                    gazeTracking = false;
                    calibStatus = 13;
                    
                    printf("Vid Finished\n");
                    
                }
    
    
}


int main( int argc, const char** argv ) {
    
    string input;
    
    std::cout << "Please enter your screen width:\n>";
    std::getline (std::cin,input);
    std::istringstream myStream(input);
    myStream >> screenWidth;
    
    std::cout << "Please enter your screen height:\n>";
    std::getline (std::cin,input);
    std::istringstream myStream2(input);
    myStream2 >> screenHeight;
    
    
    CvCapture* capture;
    cv::Mat frame;
    
    
//    backgroundImage = cvCreateImage(Size(screenWidth, screenHeight), 8, 3);
//    cvSet(backgroundImage, cvScalar(0,0,0));

    
    faceImage = new cv::Mat;
    firstFrame = new cv::Mat;

  // Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){
      
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
  
    };

    cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
    cv::moveWindow(main_window_name, 0, 0);

    createCornerKernels();
    ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2), 43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
>>>>>>> Stashed changes



  // Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture ) {
    while( true ) {
        frame = cvQueryFrame( capture );
        
        // mirror it
        cv::flip(frame, frame, 1);
<<<<<<< Updated upstream
        frame.copyTo(debugImage);
        
        cv::resizeWindow(main_window_name, 1440, 900);
        cv::moveWindow(main_window_name, 0, 0);
        
        cv::resize(debugImage, debugImage, Size(1440, 900));

      // Apply the classifier to the frame
      if( !frame.empty() ) {

        if(trackingMode){

        detectAndDisplay(debugImage);
        paintDepthDetails(debugImage);
       
    
            
      }

      }
      else {
=======
        frame.copyTo(*faceImage);

        //Resize image to fit window
        cv::resize(*faceImage, *faceImage, Size(screenWidth, screenHeight));

        // Apply the classifier to the frame
        if( !frame.empty() ) {

            if(trackingMode){

                detectAndDisplay(*faceImage);
                paintDepthDetails(*faceImage);
       
    
            
            }

        }
        else {
>>>>>>> Stashed changes
          
        printf(" --(!) No captured frame -- Break!");
        break;
          
<<<<<<< Updated upstream
      }

      
        if(!gazeTrackingStep){
        
        if(calibrationStep){
            
            paintCalibration(debugImage);
=======
        }

    
        
        if(!gazeTrackingSetup){
        
        if(calibStatus == 1){
            
            paintCalibration();
>>>>>>> Stashed changes
        
        };
            
        if (gazeTracking){
            
<<<<<<< Updated upstream
            cv::resizeWindow(main_window_name, 1440, 900);
            cv::moveWindow(main_window_name, 0, 0);
            
            cv::resize(debugImage, debugImage, Size(1440, 900));
            paintGazeLocation(debugImage);
=======
            checkFaceDepth();
            if(faceOffset){
                
                putText(*faceImage, "TRACKING DISABLED. Please move back into your original position", cvPoint(s.width/2 - s.width /4 ,200), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
                imshow(main_window_name, *faceImage);
                
            }else{
                
             paintGazeLocation(*faceImage);
                
            }
           

>>>>>>> Stashed changes
            
        }
        };
        
<<<<<<< Updated upstream
        if (gazeTrackingStep){
            
            time_t here = time(0);
            
            if ((now - here) < (-5 * step)){
=======
        if (vidTime){
            
            videoPlayback();
            
        }
        
        if(endVid){
            
            printf("End video shown \n");
            paintGazePath(firstFrame);
            imshow(main_window_name, *firstFrame);
            
            calibStatus = 11;
            
        }
        
        if (gazeTrackingSetup){
            
            time_t here = time(0);
            
            if ((now - here) < (-2 * step)){
>>>>>>> Stashed changes
                
                step++;
                
            }
            
<<<<<<< Updated upstream
            detectAndDisplay(debugImage);
=======
//            if ((now - here) < (-5 * step)){
//                
//                step++;
//                
//            }
//            
            
            detectAndDisplay(*faceImage);
>>>>>>> Stashed changes
            fullScreenCalibration();
            switch (step) {
                case 1:
                    calibP1.push_back(eyeMiddle);
                    break;
                    
                case 2:
                    calibP2.push_back(eyeMiddle);
                    break;
                    
                case 3:
                    calibP3.push_back(eyeMiddle);
                    break;
                    
                case 4:
                    calibP4.push_back(eyeMiddle);
                    break;
                    
                case 5:
                    calibP5.push_back(eyeMiddle);
                    break;
                    
                case 6:
                    calibP6.push_back(eyeMiddle);
                    break;
                    
                case 7:
                    calibP7.push_back(eyeMiddle);
                    break;
                    
                case 8:
                    calibP8.push_back(eyeMiddle);
                    break;
                    
                case 9:
                    calibP9.push_back(eyeMiddle);
                    break;
                    
                case 10:
<<<<<<< Updated upstream
                    calculateGazeScale();
                    gazeTrackingStep = false;
=======
                    //calibP1.push_back(eyeMiddle);
                    break;
                    
                case 11:
                    calculateGazeScale();
                    gazeTrackingSetup = false;
>>>>>>> Stashed changes
                    gazeTracking = true;
                    
                default:
                    break;
            }
        }
<<<<<<< Updated upstream
        else{
                
                imshow(main_window_name, debugImage);
        
        };
      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == ' ' ) { toggleSetup();}
      //if( (char)c == 'p' ) { calculateGazeScale();}
      if( (char)c == 'j' ) { showEyeGazePoints();}
      if( (char)c == 't' ) { trackingMode = !trackingMode; }
      if( (char)c == 'g' ) { resetCalibration();}
      if( (char)c == 'r' ) { step++; }

        //if( (char)c == 'f' ) { fullScreenCalibration(); }
          //imwrite("frame.png",frame); }

=======
        else if(pathShow && !endVid){
            paintGazePath(faceImage);
        }
        else{
            if(!vidTime && !endVid){
                imshow(main_window_name, *faceImage);
            }
        
        };
        
        
        int c = cv::waitKey(10);
        
        if( (char)c == 'c' ) { break; }
        if( (char)c == ' ' ) { toggleSetup();}
        if( (char)c == 'p' ) { pathShow = !pathShow; pathConfig = false;}
        if( (char)c == 'j' ) { showEyeGazePoints();}
        if( (char)c == 't' ) { trackingMode = !trackingMode; }
        if( (char)c == 'g' ) { resetCalibration();}
        if( (char)c == 'v' ) { vidTime = !vidTime;}
        if( (char)c == 'r' ) { resetCalibration(); }
        if( (char)c == 'b' ) { breaker();}

        
>>>>>>> Stashed changes
    }
  }

  releaseCornerKernels();

  return 0;
}
    


<<<<<<< Updated upstream

void toggleSetup(){
    
    
    if(!calibrationStep && !trackingMode){
        calibrationStep = true;
    }
    
    int calibWidth = ((s.width / 2) + (s.width / 8)) - ((s.width / 2) - (s.width / 8));
    int calibHeight = ((s.height / 2) + (s.height / 5)) - ((s.height / 2) - (s.height / 5));
    
    if (calibrationStep){
    
        face_cascade.detectMultiScale( debugImage, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
        
        if (faces.size() != 0) {
            
        
            if(((faces[0].height * faces[0].width) <  (calibWidth * calibHeight)) ){
    
    
                printf("SUITABLE SIZE \n");
                calibStatus = 10;
            
                trackingMode = true;
                calibrationStep = false;
            
                detectedFaceHeight = faces[0].height;
                detectedFaceWidth = faces[0].width;
          
            
                //save initial face rectangle size
                //set depth level to be initSize / thisSize
    
                //set tracking to enabled
    
                //
    
            }
            else if(((faces[0].height * faces[0].width) <  (calibWidth/2 * calibHeight/2)) ){
    
                printf("Too far \n");
                calibStatus = 2;
    
            }
            else{
            
                printf("Too close \n");
                calibStatus = 1;
            
            }
            
        }
    
    }
    
    
}


=======
>>>>>>> Stashed changes
void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;
  
  cv::Point leftPupilMain, rightPupilMain;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
    
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

    
  //-- Find Eye Centers
  leftPupil = findEyeCenter(faceROI,leftEyeRegion);
  rightPupil = findEyeCenter(faceROI,rightEyeRegion);
  
<<<<<<< Updated upstream


    
  // get corner regions
    
//  cv::Rect leftRightCornerRegion(leftEyeRegion);
//  leftRightCornerRegion.width -= leftPupil.x;
//  leftRightCornerRegion.x += leftPupil.x;
//  leftRightCornerRegion.height /= 2;
//  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
//  cv::Rect leftLeftCornerRegion(leftEyeRegion);
//  leftLeftCornerRegion.width = leftPupil.x;
//  leftLeftCornerRegion.height /= 2;
//  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
//  cv::Rect rightLeftCornerRegion(rightEyeRegion);
//  rightLeftCornerRegion.width = rightPupil.x;
//  rightLeftCornerRegion.height /= 2;
//  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
//  cv::Rect rightRightCornerRegion(rightEyeRegion);
//  rightRightCornerRegion.width -= rightPupil.x;
//  rightRightCornerRegion.x += rightPupil.x;
//  rightRightCornerRegion.height /= 2;
//  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
//  rectangle(debugFace,leftRightCornerRegion,200);
//  rectangle(debugFace,leftLeftCornerRegion,200);
//  rectangle(debugFace,rightLeftCornerRegion,200);
//  rectangle(debugFace,rightRightCornerRegion,200);
    
=======
>>>>>>> Stashed changes
    
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
    
    
  //SETS LOCATION OF EYES ON MAIN
  rightPupil.x += faces[0].x;
  rightPupil.y += faces[0].y;
  leftPupil.x += faces[0].x;
  leftPupil.y += faces[0].y;


  //Calculate middle of eyes
  eyeMiddle.x = rightPupil.x - ((rightPupil.x - leftPupil.x) / 2);
<<<<<<< Updated upstream
  eyeMiddle.y = rightPupil.y - ((rightPupil.y - leftPupil.y) / 2);;
=======
  eyeMiddle.y = rightPupil.y - ((rightPupil.y - leftPupil.y) / 2);
>>>>>>> Stashed changes
    
    
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);
  
  
    
  //Draws on main window
<<<<<<< Updated upstream
  circle(debugImage, rightPupil, 3, 1234);
  circle(debugImage, leftPupil, 3, 1234);
  
  circle(debugImage, eyeMiddle, 3, 1234);

    
  //-- Find Eye Corners
//  if (kEnableEyeCorner) {
//      
//    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
//    leftRightCorner.x += leftRightCornerRegion.x;
//    leftRightCorner.y += leftRightCornerRegion.y;
//    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
//    leftLeftCorner.x += leftLeftCornerRegion.x;
//    leftLeftCorner.y += leftLeftCornerRegion.y;
//    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
//    rightLeftCorner.x += rightLeftCornerRegion.x;
//    rightLeftCorner.y += rightLeftCornerRegion.y;
//    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
//    rightRightCorner.x += rightRightCornerRegion.x;
//    rightRightCorner.y += rightRightCornerRegion.y;
//    circle(faceROI, leftRightCorner, 3, 200);
//    circle(faceROI, leftLeftCorner, 3, 200);
//    circle(faceROI, rightLeftCorner, 3, 200);
//    circle(faceROI, rightRightCorner, 3, 200);
//      
//  }

// imshow(face_window_name, faceROI);
    
//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
//  cv::Mat destinationROI = debugFace( roi );
//  faceROI.copyTo( destinationROI );

}

cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

=======
  circle(*faceImage, rightPupil, 3, 1234);
  circle(*faceImage, leftPupil, 3, 1234);
  
  //circle(debugImage, eyeMiddle, 3, 1234);

    

}


>>>>>>> Stashed changes
void detectAndDisplay( cv::Mat frame ) {

  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];
<<<<<<< Updated upstream

//  cvtColor( frame, frame_gray, CV_BGR2GRAY );
//  equalizeHist( frame_gray, frame_gray );
//  cv::pow(frame_gray, CV_64F, frame_gray);
=======
>>>>>>> Stashed changes
  
    
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    
<<<<<<< Updated upstream
    
//  findSkin(debugImage);
=======

>>>>>>> Stashed changes

  for( int i = 0; i < faces.size(); i++ )
  {
    
    //Draw rectangle on face.
<<<<<<< Updated upstream
    //rectangle(debugImage, faces[i], 1234);
=======
    //rectangle(*faceImage, faces[i], 1234);
>>>>>>> Stashed changes
      
      
  }
    
  //-- Show what you got
  if (faces.size() > 0) {
      
    findEyes(frame_gray, faces[0]);

  }
<<<<<<< Updated upstream
=======
    
>>>>>>> Stashed changes
}



<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
