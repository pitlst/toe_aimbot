#include "usbcamera.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

std::array<cv::Mat, 2> usb_frame_array;    // 存放USB相机帧
std::array<std::mutex, 2> usb_mutex_array; // 存放USB相机帧互斥锁

/// @brief   初始化USB相机参数
/// @param usb_cam 使用的相机类
/// @param capture 选定的视频捕获对象
/// @param writer 选定的视频写入对象（目前主要是用来更改视频编码格式）
void toe::usb_camera::usb_camera_init(toe::usb_camera &usb_cam, cv::VideoCapture &capture, cv::VideoWriter &writer)
{

    // 内参设定
    int fourcc = writer.fourcc('M', 'J', 'P', 'G');
    capture.set(cv::CAP_PROP_FOURCC, fourcc);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, usb_cam.frame_width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, usb_cam.frame_height);
    capture.set(cv::CAP_PROP_FPS, 120);
    double actualWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double actualHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    // 长宽比设定验证
    if (actualWidth != toe::usb_camera::frame_width || actualHeight != toe::usb_camera::frame_height)
    {
        std::cerr << "Error: 设定失败，实际分辨率与设定分辨率不一致" << std::endl;
        return;
    }
    // 相机打开验证
    if (!capture.isOpened())
    {
        std::cout << "Error: USB相机打开失败" << std::endl;
        return;
    }
    std::cout << "USB相机初始化成功" << std::endl;
}

/// @brief   获取USB相机帧
/// @param capture 选定的视频捕获对象
/// @param frame 相机帧
void toe::usb_camera::usb_camera_get_frame(cv::VideoCapture &capture, cv::Mat &frame)
{

    capture >> frame;

    usb_mutex_array[0].lock();
    usb_frame_array[0] = frame.clone();
    usb_mutex_array[0].unlock();
}

/// @brief   处理USB相机帧
/// @param frame 相机帧
/// @param result 处理后的帧
void toe::usb_camera::usb_camera_detect(cv::Mat &frame, cv::Mat &result)
{
    cv::Mat grad_x, grad_y, grey_img;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat temp1; // 临时变量
    cv::Mat frame_clone;

    frame_clone = frame.clone();

    int scale = 6;
    int delta = 0;
    int ddepth = CV_16S;

    // 求图像梯度部分
    cv::cvtColor(frame_clone, grey_img, cv::COLOR_BGR2GRAY);

    cv::Sobel(grey_img, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::Sobel(grey_img, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::Mat abs_grad_xy;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, abs_grad_xy);

    cv::Mat binary; //
    cv::GaussianBlur(abs_grad_xy, binary, cv::Size(9, 9), 2, 2);
    // 以上都是计算图像梯度//

    // 二值化，开操作部分//
    cv::Mat threshold_img; // 二值化结果
    cv::threshold(binary, threshold_img, 120, 255, cv::THRESH_BINARY);

    // 尝试开操作之类的处理一下
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));

    // 开操作
    cv::Mat opening;
    cv::morphologyEx(threshold_img, opening, cv::MORPH_OPEN, kernel);

    // 结束//

    // result = opening.clone();
    //判断圆形区域，绘制圆形
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(opening, circles, cv::HOUGH_GRADIENT, 1,
                     opening.rows / 4, // change this value to detect circles with different distances to each other
                     200, 30, 5, 300); // change the last two values to detect circles with different sizes

    //cv::Mat circles_img;
    //cv::cvtColor(frame_clone, circles_img, cv::COLOR_BGR2HSV);

    cv::Mat circles_img;
    circles_img = frame.clone();

    cv::Point possible_ball_center;
    int possible_ball_radius = 0;
    // 绘制圆形
    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // 绘制圆心
        cv::circle(circles_img, center, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
        // 绘制圆轮廓
        cv::circle(circles_img, center, radius, cv::Scalar(0, 0, 255), 3, 1, 0);
        if (i == circles.size() - 1)
        {
            possible_ball_center = center;
            possible_ball_radius = radius;
        }
    }
    result = circles_img.clone();
/*
    // 可能的球的矩形
    if (possible_ball_center.x >= 0 && possible_ball_center.y >= 0 && possible_ball_radius > 0 && possible_ball_radius > 0 &&
        possible_ball_center.x + possible_ball_radius <= circles_img.cols &&
        possible_ball_center.y + possible_ball_radius <= circles_img.rows)
    {
        // ROI是有效的，可以继续处理
        cv::Rect possible_ball(rect_area_limit(possible_ball_center.x - possible_ball_radius, 1, 640),
                               rect_area_limit(possible_ball_center.y - possible_ball_radius, 1, 480),
                               possible_ball_radius + possible_ball_radius,
                               possible_ball_radius + possible_ball_radius);

        cv::Mat croppedImage = circles_img(possible_ball);

        cv::Scalar averageColor = cv::mean(croppedImage);

        if (averageColor[0] < 30 && color_judge(croppedImage))
        {
            cv::circle(frame_clone, possible_ball_center, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
            // 绘制圆轮廓
            cv::circle(frame_clone, possible_ball_center, possible_ball_radius, cv::Scalar(0, 0, 255), 3, 4, 0);
        }
    }

    /// cv::imshow("circles", frame_clone);

    result = frame_clone.clone();
    */
}

/// @brief   判断颜色
/// @param frame 相机帧
/// @return 颜色是否符合要求
bool color_judge(cv::Mat &frame)
{
    if (frame.empty())
    {
        return false;
    }

    cv::Mat hsvImage;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lowerYellow(20, 100, 100);
    cv::Scalar upperYellow(30, 255, 255);
    cv::Scalar lowerBlue(100, 100, 100);
    cv::Scalar upperBlue(140, 255, 255);
    cv::Scalar lowerWhite(0, 0, 200); // 白色在HSV中通常有很高的Value
    cv::Scalar upperWhite(180, 50, 255);

    cv::Mat maskYellow, maskBlue, maskWhite;
    cv::inRange(hsvImage, lowerYellow, upperYellow, maskYellow);
    cv::inRange(hsvImage, lowerBlue, upperBlue, maskBlue);
    cv::inRange(hsvImage, lowerWhite, upperWhite, maskWhite);

    bool hasYellow = (cv::countNonZero(maskYellow) > 5);
    bool hasBlue = (cv::countNonZero(maskBlue) > 5);
    bool hasWhite = (cv::countNonZero(maskWhite) > 5);

    if (hasYellow && hasBlue && hasWhite)
    {
        return true;
    }
    else
    {
        return false;
    }
}
//@brief   限制矩形面积
/// @param input 输入矩形面积
/// @param limit_min 最小限制
/// @param limit_max 最大限制
/// @return 限制后的矩形面积
int rect_area_limit(int input, int limit_min, int limit_max)
{
    if (input < limit_min)
    {
        return limit_min;
    }
    else if (input > limit_max)
    {
        return limit_max;
    }
    else
    {
        return input;
    }
}