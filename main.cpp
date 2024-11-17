
/*

               _____                       _____                       ____
              /\    \                     /\    \                     /\    \
             /::\    \                   /::\    \                   /::\    \
             \:::\    \                 /::::\    \                 /::::\    \
              \:::\    \               /::::::\    \               /::::::\    \
               \:::\    \             /:::/\:::\    \             /:::/\:::\    \
                \:::\    \           /:::/  \:::\    \           /:::/__\:::\    \
                /::::\    \         /:::/    \:::\    \         /::::\   \:::\    \
               /::::::\    \       /:::/      \:::\    \       /::::::\   \:::\    \
              /:::/\:::\    \     /:::/        \:::\    \     /:::/\:::\   \:::\    \
             /:::/  \:::\____\   /:::/          \:::\____\   /:::/__\:::\   \:::\____\
            /:::/   /\::/    /   \:::\          /:::/    /   \:::\   \:::\   \::/    /
           /:::/   /  \/____/     \:::\        /:::/    /     \:::\   \:::\   \/____/
          /:::/   /                \:::\      /:::/    /       \:::\   \:::\    \
         /:::/   /                  \:::\    /:::/    /         \:::\   \:::\____\
        /:::/   /                    \:::\  /:::/    /           \:::\   \::/    /
       /:::/   /                      \:::\/:::/    /             \:::\   \/____/
      /:::/   /                        \::::::/    /               \:::\    \
     /:::/   /                          \::::/    /                 \:::\____\
     \::/   /                            \::/    /                   \::/    /
      \/___/                              \/____/                     \/____/

TOE创新实验室
*/

#include <iostream>
#include <string>
#include <thread>
#include <fstream>
#include <atomic>
#include <chrono>

// 捕获ctrl+c的中断
#include <signal.h>

#include "detect.hpp"
#include "camera.hpp"
#include "nlohmann/json.hpp"
#include "serial.hpp"
#include "usbcamera.hpp"
#include "openvino/openvino.hpp"

 #define DEBUG 1// 调试开关

// 原子变量，用于通知线程终止

std::atomic<bool> state;

// 全局变量

toe::hik_camera hik_cam; // 创建海康相机的对象

toe::usb_camera usb_cam; // 创建usb相机的对象

toe::serial serial; // 创建串口的对象

toe::ov_detect ov_detector; // 创建yolo的对象
                            // 创建父类////
// toe::ov_detect_base ov_detector_base;

nlohmann::json config;

cv::Mat usb_cam_frame; // 用于保存usb相机的图像

// 串口信息锁
std::mutex serial_nutex;
// 这里串口信息的结构体或者变量自行定义

extern volleyball_ball_posion ball_posion;

// 监控命令行ctrl+c,用于手动退出
void sigint_handler(int sig)
{
    if (sig == SIGINT)
    {
        state.store(false);
    }
}

// 串口线程
void serial_process()
{
    std::vector<double> msg;
    // serial.init_port(config);

    while (state.load())
    {
        // msg.push_back(ball_posion.x); // 这里可以根据实际情况修改串口信息
        // msg.push_back(ball_posion.y);
        // msg.push_back(ball_posion.Deep);
        //
        // serial.send_msg(msg);
    }
}
// 处理线程
void detect_process(void)
{
    cv::Mat MVS_frame;
    cv::Mat USB_frame;
    cv::Mat USB_detected_frame;

    cv::Mat drawing = cv::Mat();
    int k;
    int color = 0;
    ov_detector.detect_init(config, color);

    int frame_count = 0;
    std::chrono::steady_clock::time_point prev_time = std::chrono::steady_clock::now(); // 记录开始时间

    while (state.load())
    {
        std::vector<cv::Rect2f> bjects;

        // 括号不能删，这是锁的生命周期
        {
            // 获取图像
            std::lock_guard<std::mutex> MVS_lock(mutex_array[0]);

            MVS_frame = frame_array[0];

            std::lock_guard<std::mutex> USB_lock(usb_mutex_array[0]);

            USB_frame = usb_frame_array[0];
        }

            if (DEBUG)
            {   // 拷贝一次drawing

            USB_frame.copyTo(drawing);

            }
        if (MVS_frame.size[0] > 0)
        {
            // 这里可以编写图像处理与串口信息生成
            // 显示图像，正式运行上赛场时需要注释和下面的waitKey，这玩意严重影响性能
            if (DEBUG)
            {
                cv::imshow("1", MVS_frame);
                k = cv::waitKey(1);
            }
        }

        if (USB_frame.data)
        {

            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = current_time - prev_time;

            // 处理图像
            // usb_cam.usb_camera_detect(USB_frame, USB_detected_frame , config);
            ov_detector.detect(bjects , drawing);

            if (elapsed_seconds.count() >= 1)
            {
                usb_cam.FPS = frame_count / elapsed_seconds.count();
                std::cout << "USB相机处理后FPS:" << usb_cam.FPS << std::endl;

                prev_time = current_time;
                frame_count = 0; // 重置帧数
            }

            frame_count++;

            if (DEBUG) // 调试内容
            {

                for ( auto &object : ov_detector.objects)
                {
                    cv::rectangle(drawing, object, cv::Scalar(0, 255, 0), 2);
                }

                cv::imshow("USB相机", drawing);

                // cv::imshow("USB相机处理后", USB_detected_frame);
                cv::waitKey(1);
            }
        }

        if (k == 27)
        {
            state.store(false);
            break;
        }
    }
    cv::destroyAllWindows();
}
// 图像获取线程
// 这里我需要解释一下，因为海康的sdk直接有非用户触发，由相机内部的定时器触发来生成图像的api
// 所以对于使用该api的海康相机只要初始化就是异步的，不需要这个线程，初始化可以在主线程做，使用这个api是为了以后能够更方便的切换为硬触发，不需要修改代码
// 但是对于usb相机和realsence都需要手动由用户触发，不支持这种模式，所以需要保留这个线程以备手动触发获取图像

// 暂时注释掉，因为要简单使用，不用海康
void grab_img(void)
{
    // 初始化
    // 调用usb相机画面的实现函数
    cv::VideoCapture usb_camera_cap(0, cv::CAP_V4L2);
    cv::VideoWriter writer;
    usb_cam.usb_camera_init(usb_cam, usb_camera_cap, writer, config);

    int frame_count = 0;
    std::chrono::steady_clock::time_point prev_time = std::chrono::steady_clock::now(); // 记录开始时间

    while (state.load())
    {

        usb_cam.usb_camera_get_frame(usb_camera_cap, usb_cam_frame);
        ov_detector.push_img(usb_cam_frame);
        std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - prev_time;
        // ov_detector.get_results();

        //ov_detector.show_results(usb_cam_frame);
        // 处理图像
        if (elapsed_seconds.count() >= 1)
        {
            usb_cam.FPS = frame_count / elapsed_seconds.count();
            std::cout << "USB相机FPS:" << usb_cam.FPS << std::endl;

            prev_time = current_time;
            frame_count = 0; // 重置帧数
        }
        //cv::imshow("USB相机", usb_cam_frame);
        //cv::waitKey(1);
        frame_count++;
    }

    cv::destroyAllWindows(); // 销毁窗口

    // int device_num = 0;
    // hik_cam.hik_init(config, device_num);
    //// 重启一次防止不正常退出后启动异常
    // hik_cam.hik_end();
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // hik_cam.hik_init(config, device_num);
    // while (state.load())
    //{
    //     // 手动触发获取图像
    //     sleep(5);
    // }
    // hik_cam.hik_end();
}

int main()
{
    // 初始化全局变量
    state.store(true);
    // 为了提升cout的效率关掉缓存区同步，此时就不能使用c风格的输入输出了，例如printf
    // oi上常用的技巧，还有提升输出效率的就是减少std::endl和std::flush的使用
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    // 看一下项目的路径，防止执行错项目
    std::cout << PROJECT_PATH << std::endl;

    std::ifstream f(std::string(PROJECT_PATH) + std::string("/config.json"));
    config = nlohmann::json::parse(f);

    // 启动线程
    std::thread grab_thread = std::thread(grab_img);
    std::thread detect_thread = std::thread(detect_process);
    std::thread serial_thread = std::thread(serial_process);
    grab_thread.detach();
    detect_thread.detach();
    serial_thread.detach();
    // 简单线程看门狗实现，需配合循环挂起的bash脚本使用
    // 这里会检测线程是否不正常运行，如果不正常立刻退出
    while (state.load())
    {
        if (grab_thread.joinable() && detect_thread.joinable() && serial_thread.joinable())
        {
            state.store(false);
            break;
        }
    }
    return 0;
}
