#include <iostream>
#include <fstream>

#include "camera.hpp"
#include "nlohmann/json.hpp"

int main()
{
    toe::hik_camera temp;
    std::ifstream f("../config.json");
    nlohmann::json temp_apra = nlohmann::json::parse(f);
    int device_num = 0;
    temp.hik_init(temp_apra, device_num);
    int k = 0;
    cv::Mat img;
    while (k != 27)
    {
        mutex_array[device_num].lock();
        img = frame_array[device_num];
        mutex_array[device_num].unlock();
        if (img.data)
        {
            cv::imshow("frame",img);
        }
        k = cv::waitKey(1);
    }
    temp.hik_end();
    return 0;
}