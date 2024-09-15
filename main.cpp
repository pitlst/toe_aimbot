#include <iostream>
#include <string>
#include <thread>
#include <fstream>

#include "camera.hpp"
#include "nlohmann/json.hpp"
#include "serial.hpp"

toe::hik_camera hik_cam;
nlohmann::json config;

int mode = 0;
int color = 0;

void detect_process(void)
{
    // 编写推理和处理代码
}

void grab_img(void)
{
    int device_num = 0;
    hik_cam.hik_init(config, device_num);
    // 重启一次防止不正常退出后启动异常
    hik_cam.hik_end();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    hik_cam.hik_init(config, device_num);
    int k = 0;
    while (k != 27)
    {
        mutex_array[device_num].lock();
        cv::Mat frame = frame_array[device_num];
        mutex_array[device_num].unlock();
        if(frame.size[0] > 0)
        {
            // 这里记得将图像传递到其他线程
            cv::imshow("1", frame);
            k = cv::waitKey(1);
        }
    }
    hik_cam.hik_end();
    cv::destroyAllWindows();
}

int main()
{

    // 为了提升cout的效率关掉缓存区同步，此时就不能使用c风格的输入输出了，例如printf
    // oi上常用的技巧，还有提升输出效率的就是减少std::endl和std::flush的使用
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    std::cout << PROJECT_PATH << std::endl;
    std::ifstream f(std::string(PROJECT_PATH) + std::string("/config.json"));
    config = nlohmann::json::parse(f);

    std::thread grab_thread = std::thread(grab_img);
    grab_thread.detach();
    std::thread detect_thread = std::thread(detect_process);
    detect_thread.detach();
    while(1){;}
    return 0;
}