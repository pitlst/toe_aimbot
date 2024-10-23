#ifndef DETECT_HPP
#define DETECT_HPP

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "structs.hpp"
#include "nlohmann/json.hpp"

#define PROJECT_PATH "/home/toe-volleyball/toe_aimbot/"

namespace toe
{
    class ov_detect_base
    {
    public:
        ov_detect_base() = default;
        ~ov_detect_base() = default;

        virtual void preprocess() = 0;
        virtual void inference() = 0;
        virtual void postprocess() = 0;

        // 根据配置文件初始化
        void Init(const nlohmann::json &input_json, int color);
        
        /// @brief 将图像推送到处理队列中
        /// @param img 输入图像
        void push_img(const cv::Mat &img);

        /// @brief 可视化推理结果，调试使用
        /// @param img 输出目标图像
        /// @return 
        bool show_results(cv::Mat &img);

        /// @brief 推理全套流程执行函数
        /// @return 
        bool detect();


        std::vector<volleyball_data> get_results();

    protected:
        // 输入的图像缓存
        const int max_size_ = 10;
        std::vector<cv::Mat> input_imgs;

        // 配置参数
        detect_data param_;
        // 检测到的排球容器
        std::vector<volleyball_data> output_data;
        // 最后输出的排球结构体
        volleyball_data final_armor;
        // 输入图像的线程锁
        std::mutex img_mutex_;
        // 输出排球的线程锁
        std::mutex outputs_mutex_;

    public:
        cv::Mat input_img;
    };

    class ov_detect : public ov_detect_base
    {
    public:
        ov_detect() = default;
        ~ov_detect() = default;

        std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                          cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
        const std::vector<std::string> class_names = {"volleyball"};
        bool detect_init(const nlohmann::json &input_json,int color);
        // void detect_frame( cv::Mat &frame, cv::Mat &output);
        void preprocess(void);
        void inference(void);
        void postprocess(void);

        //std::vector<volleyball_data> get_results();

    private:
        // 创建OpenVINO核、编译模型、创建推理请求
        ov::Core core;
        //ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;

        // 创建排球对象的输出数据结构
        std::vector<volleyball_data> output_nms_;
        //std::vector<volleyball_data> output_data;

        std::vector<float> stride_;
        std::vector<std::vector<float>> anchors;
        // 输出tensor层数
        size_t out_tensor_size;
        // 准备输入网络的图像数据
        std::vector<float> blob;
        cv::Mat input_image_temp;

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
        return (1.0 / (1.0 + std::exp(-x)));
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