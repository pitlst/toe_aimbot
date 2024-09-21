#include <iostream>
#include <string>
#include <thread>
#include <fstream>
#include <atomic>

// 捕获ctrl+c的中断
#include <signal.h>

#include "camera.hpp"
#include "nlohmann/json.hpp"
#include "serial.hpp"

// 原子变量，用于通知线程终止
std::atomic<bool> state;
// 全局变量
toe::hik_camera hik_cam;
nlohmann::json config;
// 任何跨线程的信息都需要使用锁来保护
// 串口信息锁
std::mutex serial_nutex;
// 这里串口信息的结构体或者变量自行定义




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
    while (state.load())
    {
        sleep(3);
    }
}
// 处理线程
void detect_process(void)
{
    cv::Mat frame;
    int k;
    while (state.load())
    {
        // 括号不能删，这是锁的生命周期
        {
            // 获取图像
            std::lock_guard<std::mutex> lock(mutex_array[0]);
            frame = frame_array[0];
        }
        if (frame.size[0] > 0)
        {
            // 这里可以编写图像处理与串口信息生成
            // 显示图像，正式运行上赛场时需要注释和下面的waitKey，这玩意严重影响性能
            cv::imshow("1", frame);
            k = cv::waitKey(1);
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
void grab_img(void)
{
    int device_num = 0;
    hik_cam.hik_init(config, device_num);
    // 重启一次防止不正常退出后启动异常
    hik_cam.hik_end();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    hik_cam.hik_init(config, device_num);
    while (state.load())
    {
        // 手动触发获取图像
        sleep(5);
    }
    hik_cam.hik_end();
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