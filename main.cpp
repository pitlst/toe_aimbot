#include <iostream>
#include <fstream>

#include "camera.hpp"
#include "nlohmann/json.hpp"

int main()
{
    toe::hik_camera temp;
    std::ifstream f("../config.json");
    nlohmann::json temp_apra = nlohmann::json::parse(f);
    temp.hik_init(temp_apra,0);
    int k = 0;
    cv::Mat img;
    while (k != 27)
    {
        frame_mutex.lock();
        img = frame_rgb;
        frame_mutex.unlock();
        if (img.data)
        {
            cv::imshow("frame",img);
        }
        k = cv::waitKey(1);
    }
    temp.hik_end();
    return 0;
}