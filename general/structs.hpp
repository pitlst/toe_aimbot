#ifndef TOE_STRUCTS_H_
#define TOE_STRUCTS_H_

#include <string>
#include "opencv2/opencv.hpp"

// 数据转换用联合体
union acm_data{
    uint8_t     bit[4];
    float       data;
};
// 相机参数
struct camera_data
{
    int device_id;
    int width;
    int height;
    int offset_x;
    int offset_y;
    int ADC_bit_depth;
    int exposure;
    int gain;
    int balck_level;
    bool Reverse_X;
    bool Reverse_Y;
};


#endif