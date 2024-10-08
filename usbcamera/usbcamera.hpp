#ifndef TOE_USBCAMERA_H
#define TOE_USBCAMERA_H

#include "usbcamera.hpp"
#include <iostream>
#include <string>
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>

namespace toe
{

    class usb_camera  
    {

    public:
        double FPS = 0.0;

        int frame_width = 640;
        int frame_height = 480;

        usb_camera() = default;  // 构造函数
        ~usb_camera() = default; // 析构函数

        bool usb_camera_init(toe::usb_camera &usb_cam, cv::VideoCapture &capture, cv::VideoWriter &writer ,const nlohmann::json &input_json);
        void usb_camera_get_frame(cv::VideoCapture &capture, cv::Mat &frame);
        void usb_camera_detect(cv::Mat &frame, cv::Mat &result , const nlohmann::json &input_json);


    private:
        cv::VideoCapture cap;
        std::mutex usb_frame_mutex;
        cv::Mat frame;

    };

}
typedef struct
{
    double x;
    double y;
    double Deep;
    
}volleyball_ball_posion;


bool color_judge(cv::Mat &frame);
int rect_area_limit(int input , int limit_min, int limit_max) ;

extern std::array<cv::Mat, 2> usb_frame_array;    // 存放USB相机帧
extern std::array<std::mutex, 2> usb_mutex_array; // 存放USB相机帧互斥锁
#endif