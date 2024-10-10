#ifndef DETECT_HPP
#define DETECT_HPP

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
namespace toe
{
    class yolo
    {
    public:
        yolo() = default;
        ~yolo() = default;
        ov::Core core;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;

        std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                          cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
        const std::vector<std::string> class_names = {"volleyball"};

    private:
    };

}

bool detect_init(toe::yolo &yolo_class);
void detect_frame(toe::yolo &yolo_class, cv::Mat &frame, cv::Mat &output);

#endif