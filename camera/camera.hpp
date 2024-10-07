#ifndef TOE_CAMERA_H_
#define TOE_CAMERA_H_

#include <mutex>
#include <array>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "MvCameraControl.h"

#include "nlohmann/json.hpp"
#include "structs.hpp"

namespace toe
{
    
    //#define DEBUE 1
    class hik_camera final
    {
    public:
        hik_camera() = default;
        ~hik_camera() = default;

        bool hik_init(const nlohmann::json & input_json, int devive_num);
        bool hik_end();
        
    private:
        camera_data params_;
        std::mutex frame_mutex;
        cv::Mat frame;

        // 海康相机指针
        void *handle = nullptr;
    };

}

// 暂时仅做了同时7个海康相机的支持
extern void __stdcall image_callback_0(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern void __stdcall image_callback_1(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern void __stdcall image_callback_2(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern void __stdcall image_callback_3(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern void __stdcall image_callback_4(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern void __stdcall image_callback_5(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern void __stdcall image_callback_6(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern std::array<cv::Mat, 7> frame_array;
extern std::array<std::mutex, 7> mutex_array;

#endif