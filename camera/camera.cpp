#include "camera.hpp"

#include <string>
#include <sstream>
#include <stdexcept>

#include <stdlib.h>

using namespace toe;

std::array<cv::Mat, 7> frame_array;
std::array<std::mutex, 7> mutex_array;

bool hik_camera::hik_init(const nlohmann::json & input_json, int devive_num)
{
    // 检查devive_num是否超出要求
    if (devive_num > 6)
    {
        throw std::logic_error("devive_num的大小超出要求，暂时仅支持7个设备，devive_num不能大于6");
    }
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
        std::stringstream s;
        s << "V_CC_EnumDevices fail! nRet " << nRet;
        throw std::logic_error(s.str());
    }
    if (stDeviceList.nDeviceNum > 0)
    {
        for (int i = 0; i < stDeviceList.nDeviceNum; i++)
        {
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo)
            {
                std::stringstream s;
                s << "找到的设备报错，对应设备号为 " << i;
                throw std::logic_error(s.str());
            }         
        }  
    } 
    else
    {
        throw std::logic_error("Find No Devices!");
    }

    unsigned int nIndex = params_.device_id;
    // 选择设备并创建句柄
    // select device and create handle
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "MV_CC_CreateHandle fail! nRet " << nRet;
        throw std::logic_error(s.str());
    }
    // 打开设备
    // open device
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "MV_CC_OpenDevice fail! nRet " << nRet;
        throw std::logic_error(s.str());
    }
    // 设置触发模式为off
    // set trigger mode as off
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "MV_CC_SetTriggerMode fail! nRet " << nRet;
        throw std::logic_error(s.str());
    }

    // ch：设置曝光时间，图像的长宽,和所取图像的偏移
    //注意，这里对offset的值应当提前归零，防止出现长度溢出问题
    nRet = MV_CC_SetIntValue(handle, "OffsetX", 0);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置OffsetX错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    nRet = MV_CC_SetIntValue(handle, "OffsetY", 0);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置OffsetY错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", params_.exposure);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置曝光错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    nRet = MV_CC_SetIntValue(handle, "Width", params_.width);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置Width错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    nRet = MV_CC_SetIntValue(handle, "Height", params_.height);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置Height错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    // 这里设置相机偏移两遍是因为有的时候上次窗长宽与偏移相冲突
    nRet = MV_CC_SetIntValue(handle, "OffsetX", params_.offset_x);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置OffsetX错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    nRet = MV_CC_SetIntValue(handle, "OffsetY", params_.offset_y);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置OffsetY错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }

    // RGB格式0x02180014
    // bayerRG格式0x01080009
    nRet = MV_CC_SetEnumValue(handle, "PixelFormat", 0x01080009);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置传输图像格式错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    nRet = MV_CC_SetFloatValue(handle, "Gain", params_.gain);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "设置增益错误,错误码:" << nRet;
        throw std::logic_error(s.str());
    }
    // 注册抓图回调
    // register image callback
    switch (devive_num)
    {
    case 0:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_0, handle);
        break;
    case 1:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_1, handle);
        break;
    case 2:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_2, handle);
        break;
    case 3:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_3, handle);
        break;
    case 4:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_4, handle);
        break;
    case 5:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_5, handle);
        break;
    case 6:
        nRet = MV_CC_RegisterImageCallBackEx(handle, image_callback_6, handle);
        break;
    default:
        std::stringstream s;
        s << "意外的设备号:" << devive_num;
        throw std::logic_error(s.str());
    }
    
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "MV_CC_RegisterImageCallBackEx fail! nRet " << nRet;
        throw std::logic_error(s.str());
    }
    // 开始取流
    // start grab image
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet)
    {
        std::stringstream s;
        s << "MV_CC_StartGrabbing fail! nRet " << nRet;
        throw std::logic_error(s.str());
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

void __stdcall image_callback_0(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[0].lock();
        cv::cvtColor(img_bayerrg_, frame_array[0], cv::COLOR_BayerRG2RGB);
        mutex_array[0].unlock();
    }
}

void __stdcall image_callback_1(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[1].lock();
        cv::cvtColor(img_bayerrg_, frame_array[1], cv::COLOR_BayerRG2RGB);
        mutex_array[1].unlock();
    }
}

void __stdcall image_callback_2(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[2].lock();
        cv::cvtColor(img_bayerrg_, frame_array[2], cv::COLOR_BayerRG2RGB);
        mutex_array[2].unlock();
    }
}

void __stdcall image_callback_3(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[3].lock();
        cv::cvtColor(img_bayerrg_, frame_array[3], cv::COLOR_BayerRG2RGB);
        mutex_array[3].unlock();
    }
}

void __stdcall image_callback_4(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[4].lock();
        cv::cvtColor(img_bayerrg_, frame_array[4], cv::COLOR_BayerRG2RGB);
        mutex_array[4].unlock();
    }
}

void __stdcall image_callback_5(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[5].lock();
        cv::cvtColor(img_bayerrg_, frame_array[5], cv::COLOR_BayerRG2RGB);
        mutex_array[5].unlock();
    }
}

void __stdcall image_callback_6(unsigned char *pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {
        cv::Mat img_bayerrg_ = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        mutex_array[6].lock();
        cv::cvtColor(img_bayerrg_, frame_array[6], cv::COLOR_BayerRG2RGB);
        mutex_array[6].unlock();
    }
}