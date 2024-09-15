#include "camera.hpp"

#include <string>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

using namespace toe;

cv::Mat frame_rgb;
std::mutex frame_mutex;

bool hik_camera::hik_init(const nlohmann::json & input_json, int devive_num)
{
    // 根据json解析参数
    auto temp_para = input_json["camera"][std::to_string(devive_num)];
    params_.device_id = devive_num;
    params_.width = temp_para["width"].get<int>();
    params_.height = temp_para["height"].get<int>();
    params_.offset_x = temp_para["offset_x"].get<int>();
    params_.offset_y = temp_para["offset_y"].get<int>();
    params_.ADC_bit_depth = temp_para["ADC_bit_depth"].get<int>();
    params_.exposure = temp_para["exposure"].get<int>();
    params_.gain = temp_para["gain"].get<int>();
    params_.balck_level = temp_para["balck_level"].get<int>();

    params_.Reverse_X = temp_para["Reverse_X"].get<bool>();
    params_.Reverse_Y = temp_para["Reverse_Y"].get<bool>();

    // 相机初始化
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    // 枚举设备
    // enum device
    int nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet)
    {
        printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
        exit(1);
    }
    if (stDeviceList.nDeviceNum > 0)
    {
        for (int i = 0; i < stDeviceList.nDeviceNum; i++)
        {
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo)
            {
                exit(1);
            }         
        }  
    } 
    else
    {
        printf("Find No Devices!\n");
        exit(1);
    }

    unsigned int nIndex = params_.device_id;
    // 选择设备并创建句柄
    // select device and create handle
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
        exit(1);
    }
    // 打开设备
    // open device
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
        exit(1);
    }
    // 设置触发模式为off
    // set trigger mode as off
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetTriggerMode fail! nRet [%x]\n", nRet);
        exit(1);
    }

    // ch：设置曝光时间，图像的长宽,和所取图像的偏移
    //注意，这里对offset的值应当提前归零，防止出现长度溢出问题
    nRet = MV_CC_SetIntValue(handle, "OffsetX", 0);
    if (MV_OK != nRet)
    {
        printf("设置OffsetX错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetIntValue(handle, "OffsetY", 0);
    if (MV_OK != nRet)
    {
        printf("设置OffsetX错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", params_.exposure);
    if (MV_OK != nRet)
    {
        printf("设置曝光错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetIntValue(handle, "Width", params_.width);
    if (MV_OK != nRet)
    {
        printf("设置Width错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetIntValue(handle, "Height", params_.height);
    if (MV_OK != nRet)
    {
        printf("设置Height错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetIntValue(handle, "OffsetX", params_.offset_x);
    if (MV_OK != nRet)
    {
        printf("设置Height错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetIntValue(handle, "OffsetY", params_.offset_y);
    if (MV_OK != nRet)
    {
        printf("设置Height错误,错误码:[%x]\n", nRet);
        exit(1);
    }

    // RGB格式0x02180014
    // bayerRG格式0x01080009
    nRet = MV_CC_SetEnumValue(handle, "PixelFormat", 0x01080009);
    if (MV_OK != nRet)
    {
        printf("设置传输图像格式错误,错误码:[%x]\n", nRet);
        exit(1);
    }
    nRet = MV_CC_SetFloatValue(handle, "Gain", params_.gain);
    if (MV_OK != nRet)
    {
        printf("设置增益错误,错误码:[%x]\n", nRet);
        exit(1);;
    }
    // 注册抓图回调
    // register image callback
    nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_EX, handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_RegisterImageCallBackEx fail! nRet [%x]\n", nRet);
        exit(1); 
    }
    // 开始取流
    // start grab image
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
        exit(1);
    }
    std::cout << "hik init" << std::endl;
    return true;
}

bool hik_camera::hik_end()
{
    // 停止取流
    // end grab image
    int nRet = MV_CC_StopGrabbing(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StopGrabbing fail! nRet [%x]\n", nRet);
        exit(1);
    }

    // 关闭设备
    // close device
    nRet = MV_CC_CloseDevice(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CloseDevice fail! nRet [%x]\n", nRet);
        exit(1);
    }

    // 销毁句柄
    // destroy handle
    nRet = MV_CC_DestroyHandle(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
        exit(1);
    }

    if (nRet != MV_OK)
    {
        if (handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
            handle = NULL;
        }
    }
    return true;
}

void __stdcall image_callback_EX(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        frame_mutex.lock();
        cv::cvtColor(img_bayerrg_, frame_rgb, cv::COLOR_BayerRG2RGB);
        frame_mutex.unlock();
    }
}