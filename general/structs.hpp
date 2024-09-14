#ifndef TOE_STRUCTS_H_
#define TOE_STRUCTS_H_

#include <string>
#include <opencv2/opencv.hpp>

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
// 推理模块参数
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
// 装甲板参数
struct armor_data
{
    float x_c;
    float y_c;
    float z;

    float conf;
    int t_size;
    int type;
    int color;
    
    cv::Rect rect;
    cv::Point2f pts[5];
};

// 决策器配置参数
struct determine_data
{
    size_t max_history_armor_len;
    size_t max_loss_armor_fps;
    int z_scale;
    int width;
    int height;
};


struct pick_merge_store{
    int id;
    std::vector<cv::Point2f> merge_pts;
    std::vector<float> merge_confs;
};

typedef armor_data s_armor;
#endif