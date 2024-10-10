#include "detect.hpp"
#include "opencv2/objdetect.hpp"
#include "openvino/openvino.hpp"

#include <iostream>
#include <string>
#include <vector>
/*
bool detect_init(toe::yolo &yolo_class)
{
    yolo_class.compiled_model = yolo_class.core.compile_model
    ("/home/toe-volleyball/toe_aimbot/data/best_openvino_model/best.xml", "CPU");
    
    return true;
}

void detect_frame(toe::yolo &yolo_class, cv::Mat &frame, cv::Mat &output)
{
    
    // 这里是创建推理请求
    yolo_class.infer_request = yolo_class.compiled_model.create_infer_request();
    
    cv::Mat img = frame.clone();
    // Mat letterbox_img = letterbox(img);
    // Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(224, 224), Scalar(), true);

    // 传入推理请求
    auto input_port = yolo_class.compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img.ptr(0));
    yolo_class.infer_request.set_input_tensor(input_tensor);

    yolo_class.infer_request.infer();
    //yolo_class.infer_request.start_async();
    //yolo_class.infer_request.wait();

    auto yolo_output = yolo_class.infer_request.get_output_tensor(0);
    auto output_shape = yolo_output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;
    int rows = output_shape[2];       // 8400
    int dimensions = output_shape[1]; // 84: box[cx, cy, w, h]+80 classes scores

    std::cout << "111111111111" << std::endl;
    // -------- Step 8. Postprocess the result --------
    float *data = yolo_output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,84]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;

    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;

    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++)
    {
        cv::Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold)
        {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w));
            int top = int((cy - 0.5 * h));
            int width = int(w);
            int height = int(h);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

}
*/