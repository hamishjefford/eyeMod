#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

//#include <curses.h>



using namespace cv;

/** Constants **/


/** Function Headers */
void detectAndDisplay(cv::Mat frame);
void paintCalibration(cv::Mat framed);
void paintDepthDetails(cv::Mat frame);
void toggleSetup();
void fullScreenCalibration();
void drawCalibCircle();
void moveMouse();
void showEyeGazePoints();
void paintGazeLocation(cv::Mat framed);

#pragma mark - initVariables

/** Global variables */

cv::String face_cascade_name = "/Users/hamishjefford/Dropbox/eyeMod/eyeMod/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "eyeMod - Hamish Jefford";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat overlayImage;
std::vector<cv::Rect> faces;


cv::Point leftPupil, rightPupil, eyeMiddle;
cv::Point av1, av2, av3, av4, av5, av6, av7, av8, av9;

cv::Point topAv, bottomAv, rightAv, leftAv;

int step = 1;

time_t now;


int calibStatus = 0;
// 0 = Initial;
// 1 = Too Close;
// 2 = Too Far;
// 10 = Calibrated;

float detectedFaceWidth, detectedFaceHeight;

float scaleX, scaleY;

Vector<Point> calibP1, calibP2, calibP3, calibP4, calibP5, calibP6, calibP7, calibP8, calibP9;

float leftDiff = 0;
float rightDiff = 0;
float upDiff = 0;
float bottomDiff = 0;


//Screen Size
cv::Size s;

cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

bool trackingMode, calibrationStep, gazeTrackingStep, fullScreenMode, gazeTracking = false;

std::string calibrationInitText = "Centre your face within the area and press space";



#pragma mark - functions


void resetCalibration(){

    now = time(0);
    char* dt = ctime(&now);
    
    std::cout << "Number of sec since January 1,1970:" << now << std::endl;
    
    printf(dt);
    
    gazeTrackingStep = !gazeTrackingStep;
    
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
    
    for( auto iter = std::begin(calibP1) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av1.x = x1 / m;
    av1.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av1.x, av1.y);
    
    
    //printf("POSITION 2: \n");
    
    n = calibP2.size() ;
    m = calibP2.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP2) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av2.x = x1 / m;
    av2.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av2.x, av2.y);
    
    
    //printf("POSITION 3: \n");
    
    n = calibP3.size() ;
    m = calibP3.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP3) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av3.x = x1 / m;
    av3.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av3.x, av3.y);
    
    
    //printf("POSITION 4: \n");
    
    n = calibP4.size() ;
    m = calibP4.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP4) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av4.x = x1 / m;
    av4.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av4.x, av4.y);
    
    
    //printf("POSITION 5: \n");
    
    n = calibP5.size() ;
    m = calibP5.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP5) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av5.x = x1 / m;
    av5.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av5.x, av5.y);
    
    
    //printf("POSITION 6: \n");
    
    n = calibP6.size() ;
    m = calibP6.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP6) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av6.x = x1 / m;
    av6.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av6.x, av6.y);
    
    
    //printf("POSITION 7: \n");
    
    n = calibP7.size() ;
    m = calibP7.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP7) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av7.x = x1 / m;
    av7.y = y1 / m;
    
   // printf("X:%d, Y:%d \n", av7.x, av7.y);
    
    
    //printf("POSITION 8: \n");
    
    n = calibP8.size() ;
    m = calibP8.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP8) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av8.x = x1 / m;
    av8.y = y1 / m;
    
    //printf("X:%d, Y:%d \n", av8.x, av8.y);
    
    
    
    //printf("POSITION 9: \n");
    
    n = calibP9.size() ;
    m = calibP9.size() ;
    x1 = 0; y1 = 0;
    
    for( auto iter = std::begin(calibP9) ; n-- ; ++iter ){
        x1 = x1 + iter->x;
        y1 = y1 + iter->y;
    }
    
    av9.x = x1 / m;
    av9.y = y1 / m;
    printf("X:%d, Y:%d \n", av9.x, av9.y);
    

    topAv.y = ((av2.y + av3.y + av6.y) / 3);
    
    bottomAv.y = ((av4.y + av5.y + av9.y) / 3);
    
    rightAv.x = ((av3.x + av5.x + av8.x) / 3);
    
    leftAv.x = ((av2.x + av4.x + av7.x) / 3);
    
    
}

void drawCalibCircle(){
    
    s = debugImage.size();
    
    
    
    switch (step) {
        case 1:
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
            
        default:
            break;
    }
    
    
    
}

void paintGazeLocation(  cv::Mat framed   ){
    
    cv::Point gazePoint;
    
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

    }
    
    
    Point pt1 =  Point(gazePoint.x - 20, gazePoint.y);
    Point pt2 =  Point(gazePoint.x + 20, gazePoint.y);
    
    //Vertical
    Point pt3 =  Point(gazePoint.x, gazePoint.y - 20);
    Point pt4 =  Point(gazePoint.x, gazePoint.y + 20);
    
    line(debugImage, pt1, pt2, Scalar( 255, 255, 255 ), 2, 8);
    line(debugImage, pt3, pt4, Scalar( 255, 255, 255 ), 2, 8);
    
    circle(debugImage, gazePoint, 3, 1234);
    //circle(debugImage, eyeMiddle, 3, 1234);
    
}


void paintDepthDetails(cv::Mat frame){
    
    s = frame.size();
    float depthCalc = 0;
    
    cv::Mat calibration_Area = cv::Mat::zeros( s.height, s.width, CV_8UC3 );
    
    if(faces.size() != 0){
        
      depthCalc =  (faces[0].width * faces[0].height) / (detectedFaceHeight * detectedFaceWidth);
        
    }
    
    std::string depth_message = "Depth: " + std::to_string(depthCalc);
    
    putText(debugImage, depth_message, cvPoint(0 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
    
}


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
    
    
    //Text Displayed
    switch (calibStatus) {
        case 0:
            putText(debugImage, "Centre your face within the area and press space", cvPoint(s.width/2 - s.width /4 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
        
        case 1:
            putText(debugImage, "You're too close. Move back and try again.", cvPoint(s.width/2 - s.width /4 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
            
        case 2:
            putText(debugImage, "You're too far. Move closer and try again.", cvPoint(s.width/2 - s.width /4 ,30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(255,255,20), 1, CV_AA);
            break;
            
            
        default:
            break;
    
    }
    
    
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



  // Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture ) {
    while( true ) {
        frame = cvQueryFrame( capture );
        
        // mirror it
        cv::flip(frame, frame, 1);
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
          
        printf(" --(!) No captured frame -- Break!");
        break;
          
      }

      
        if(!gazeTrackingStep){
        
        if(calibrationStep){
            
            paintCalibration(debugImage);
        
        };
            
        if (gazeTracking){
            
            cv::resizeWindow(main_window_name, 1440, 900);
            cv::moveWindow(main_window_name, 0, 0);
            
            cv::resize(debugImage, debugImage, Size(1440, 900));
            paintGazeLocation(debugImage);
            
        }
        };
        
        if (gazeTrackingStep){
            
            time_t here = time(0);
            
            if ((now - here) < (-5 * step)){
                
                step++;
                
            }
            
            detectAndDisplay(debugImage);
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
                    calculateGazeScale();
                    gazeTrackingStep = false;
                    gazeTracking = true;
                    
                default:
                    break;
            }
        }
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

    }
  }

  releaseCornerKernels();

  return 0;
}
    



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
  eyeMiddle.y = rightPupil.y - ((rightPupil.y - leftPupil.y) / 2);;
    
    
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);
  
  
    
  //Draws on main window
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

void detectAndDisplay( cv::Mat frame ) {

  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

//  cvtColor( frame, frame_gray, CV_BGR2GRAY );
//  equalizeHist( frame_gray, frame_gray );
//  cv::pow(frame_gray, CV_64F, frame_gray);
  
    
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    
    
//  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ )
  {
    
    //Draw rectangle on face.
    //rectangle(debugImage, faces[i], 1234);
      
      
  }
    
  //-- Show what you got
  if (faces.size() > 0) {
      
    findEyes(frame_gray, faces[0]);

  }
}




