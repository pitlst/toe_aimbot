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

struct detect_data
{
    int camp;

    std::string engine_file_path;
    std::string bin_file_path;
    std::string xml_file_path;

    int batch_size;
    int h;
    int w;
    int c;

    int width;
    int height;

    float nms_thresh;
    float bbox_conf_thresh;
    float merge_thresh;

    int classes;
    int colors;
    int sizes;
    int kpts;
    
    // anchors
    std::vector<float> a1;
    std::vector<float> a2;
    std::vector<float> a3;
    std::vector<float> a4;
};

/// @brief 球员数据结构体
/// @details 包含球的位置信息
struct volleyball_data
{
    float ball_x;
    float ball_y;
    float ball_z;

    float ball_vx;
    float ball_vy;
    float ball_vz;

    float conf;
    int type;
    cv::Rect rect;
    cv::Point2f pts[5];
};


struct pick_merge_store{
    int id;
    std::vector<cv::Point2f> merge_pts;
    std::vector<float> merge_confs;
};

#endif