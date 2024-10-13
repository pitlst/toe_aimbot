#ifndef DETECT_HPP
#define DETECT_HPP

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "structs.hpp"
#include "nlohmann/json.hpp"

namespace toe
{
    class ov_detect
    {
    public:
        ov_detect() = default;
        ~ov_detect() = default;

        std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                          cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
        const std::vector<std::string> class_names = {"volleyball"};
        bool detect_init(const nlohmann::json & input_json);
        //void detect_frame( cv::Mat &frame, cv::Mat &output);
        void preprocess(void);
        void inference(void);
        void postprocess(void);

        void push_img(const cv::Mat& img);
        bool show_results(cv::Mat& img);
        std::vector<volleyball_data> get_results();
        bool detect();
    
    protected:
        // 输入的图像缓存
        const int max_size_ = 10;
        std::vector<cv::Mat> input_imgs;

    private:    
    //创建OpenVINO核、编译模型、创建推理请求
        ov::Core core;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;

        detect_data param_;

        //创建排球对象的输出数据结构
        std::vector<volleyball_data> output_data_;
        std::vector<volleyball_data> output_data;
        
        std::vector<float> stride_;
        std::vector<std::vector<float>> anchors; 

        // 输出tensor层数
        size_t out_tensor_size;

        std::vector<float> blob;
        cv::Mat input_image;

        //锁
        std::mutex img_mutex_;
        std::mutex input_image_mutex;
        std::mutex output_data_mutex;
    };

}

namespace toe
{
    inline void sigmoid(const float *src, float *dst, int length)
    {
        for (int i = 0; i < length; ++i)
        {
            dst[i] = (1.0 / (1.0 + std::exp(-src[i])));
        }
    }

    inline float sigmoid(float x)
    {
        return (1.0 / (1.0 +  std::exp(-x)));
    }



        inline float calc_iou(const volleyball_data &a, const volleyball_data &b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        float inter_area = inter.area();
        float union_area = a.rect.area() + b.rect.area() - inter_area;
        double iou = inter_area / union_area;
        if (std::isnan(iou))
        {
            iou = -1;
        }
        return iou;
    }
}

#endif