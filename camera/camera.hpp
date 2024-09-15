#ifndef TOE_CAMERA_H_
#define TOE_CAMERA_H_

#include <mutex>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "MvCameraControl.h"

#include "nlohmann/json.hpp"
#include "structs.hpp"

namespace toe
{
    class hik_camera final
    {
    public:
        hik_camera() = default;
        ~hik_camera() = default;

        bool hik_init(const nlohmann::json & input_json, int devive_num);
        bool hik_end();
        
    private:
        camera_data params_;
        cv::Mat frame;

        // 海康相机指针
        void *handle = nullptr;
    };
}

extern void __stdcall image_callback_EX(unsigned char *pData, MV_FRAME_OUT_INFO_EX* stImageInfo, void* pUser);
extern cv::Mat frame_rgb;
extern std::mutex frame_mutex;

#endif