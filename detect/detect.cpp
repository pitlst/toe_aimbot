#include "detect.hpp"
#include "opencv2/objdetect.hpp"
#include "openvino/openvino.hpp"

#include <iostream>
#include <string>
#include <vector>

#define DEBUG 1

std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                  cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
const std::vector<std::string> class_names = {"volleyball"};

void toe::ov_detect_base::Init(const nlohmann::json &input_json, int color)
{
    // 利用json文件初始化参数
    nlohmann::json temp_json = input_json;

    param_.engine_file_path = temp_json["path"]["engine_file_path"].get<std::string>();
    param_.bin_file_path = temp_json["path"]["bin_file_path"].get<std::string>();
    param_.xml_file_path = temp_json["path"]["xml_file_path"].get<std::string>();

    param_.camp = color;

    param_.batch_size = temp_json["NCHW"]["openvino"]["batch_size"].get<int>();
    param_.c = temp_json["NCHW"]["openvino"]["C"].get<int>();
    param_.w = temp_json["NCHW"]["openvino"]["W"].get<int>();
    param_.h = temp_json["NCHW"]["openvino"]["H"].get<int>();

    param_.width = temp_json["camera"]["0"]["width"].get<int>();
    param_.height = temp_json["camera"]["0"]["height"].get<int>();

    param_.nms_thresh = temp_json["thresh"]["nms_thresh"].get<double>();
    param_.bbox_conf_thresh = temp_json["thresh"]["bbox_conf_thresh"].get<double>();
    param_.merge_thresh = temp_json["thresh"]["merge_thresh"].get<double>();

    param_.classes = temp_json["nums"]["classes"].get<int>();

    param_.debug = temp_json["DEBUG"].get<bool>();

    std::vector<float> temp_vector;
    for (nlohmann::json ch : temp_json["anchors"]["1"])
    {
        temp_vector.emplace_back(ch.get<int>());
    }
    param_.a1 = temp_vector;
    temp_vector.clear();
    for (nlohmann::json ch : temp_json["anchors"]["2"])
    {
        temp_vector.emplace_back(ch.get<int>());
    }
    param_.a2 = temp_vector;
    temp_vector.clear();
    for (nlohmann::json ch : temp_json["anchors"]["3"])
    {
        temp_vector.emplace_back(ch.get<int>());
    }
    param_.a3 = temp_vector;
    temp_vector.clear();
    for (nlohmann::json ch : temp_json["anchors"]["4"])
    {
        temp_vector.emplace_back(ch.get<int>());
    }
    param_.a4 = temp_vector;
    temp_vector.clear();
}

bool toe::ov_detect::detect_init(const nlohmann::json &input_json, int color)
{
    nlohmann::json temp_json = input_json;
    toe::ov_detect_base::Init(temp_json, color);
    // 子类初始化
    std::cout << "load network" << std::endl;
    // 加载模型
    // std::cout <<param_.xml_file_path<<std::endl;
    std::shared_ptr<ov::Model> model = core.read_model(std::string(PROJECT_PATH) + param_.xml_file_path, std::string(PROJECT_PATH) + param_.bin_file_path);
    if(model->is_dynamic())
    {
        model->reshape({1,3,static_cast<long int>(640),static_cast<long int>(640)});
    }
   
    ov_detect::compiled_model = core.compile_model(model, "CPU"); 
    // 创建推理请求
    infer_request = compiled_model.create_infer_request();

    short W ,H;

    const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    input_shape = inputs[0].get_shape();

    H = input_shape[1];
    W = input_shape[2];
    model_input_shape_ = cv::Size2f(W,H);


    std::cout << W << "    " << H <<std::endl;

    const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
    output_shape = outputs[0].get_shape();

    H = output_shape[1];
    W = output_shape[2];
    model_output_shape_ = cv::Size(W,H);

    std::cout << W << "    " << H <<std::endl;


    return true;
    

    /////////////////////////////////
}

// 推理预处理部分
void toe::ov_detect::preprocess(void)
{

    // 获取原始图像尺寸
    int img_h = input_img.rows;
    int img_w = input_img.cols;

    // 获取目标尺寸
    int target_w = 640;
    int target_h = 640;

    cv::Scalar color = cv::Scalar(114, 114, 114);

    // 计算缩放比例
    scale = std::min(static_cast<float>(target_w) / img_w,
                     static_cast<float>(target_h) / img_h);
    

    // 计算缩放后的尺寸
    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);
    

    // 调整图像大小
    cv::Mat resized_img;
    cv::resize(input_img, resized_img, cv::Size(new_w, new_h));

    // 计算填充位置
    int dw = target_w - new_w;
    int dh = target_h - new_h;

    // 将填充分为两部分 (上下或左右)
    dw /= 2;
    dh /= 2;
    // 创建目标图像并填充
    cv::Mat padded_img(target_h, target_w, input_img.type(), color);

    // 将调整大小后的图像复制到填充图像的中心
    cv::Rect roi(dw, dh, new_w, new_h);
    resized_img.copyTo(padded_img(roi));

    // 返回填充信息 (top, bottom, left, right)
    padding_info = {
        static_cast<float>(dh),
        static_cast<float>(target_h - new_h - dh),
        static_cast<float>(dw),
        static_cast<float>(target_w - new_w - dw)};
    
    // 归一化
    int img_h_ = padded_img.rows;
    int img_w_ = padded_img.cols;
    float *blob_data = blob.data();

    // scale_factor_.x = static_cast<float>(input_img.cols / model_input_shape_.width);
    // scale_factor_.y = static_cast<float>(input_img.rows / model_input_shape_.height);
    input_tensor = infer_request.get_input_tensor();
    auto data1 = input_tensor.data<float>();

    auto input_port = compiled_model.input();

    std::cout << scale << std::endl;

    size_t i = 0;
    for (size_t row = 0; row < img_h_; ++row)
    {
        uchar *uc_pixel = input_img.data + row * input_img.step;
        for (size_t col = 0; col < img_w_; ++col)
        {
            // 三通道
            blob_data[i] = (float)uc_pixel[2] / 255.0;
            blob_data[i + img_h_ * img_w_] = (float)uc_pixel[1] / 255.0;
            blob_data[i + 2 * img_h_ * img_w_] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    std::memcpy(data1, blob.data(), sizeof(float) * blob.size());
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape() , blob_data);
    
    infer_request.set_input_tensor(input_tensor);

}

void toe::ov_detect::inference(void)
{
    // 推理
    infer_request.infer();
}

void toe::ov_detect::postprocess(void)
{
    output_tensor = infer_request.get_output_tensor(0);
    float *outdata = output_tensor.data<float>();

    output_shape = output_tensor.get_shape();
    auto num_box = output_shape[2];         // 8400
    auto num_classes = output_shape[1] - 4; // 1: box[cx, cy, w, h]+1 classes scores

    float top = padding_info[0];
    float left = padding_info[2];

    for (size_t i = 0; i < num_box; i++)
    {
        if (outdata[4 * num_box + i * num_classes] > param_.bbox_conf_thresh)
        {

            float cx = outdata[i];
            float cy = outdata[i + num_box];
            float w = outdata[i + 2 * num_box];
            float h = outdata[i + 3 * num_box];

            float x = cx - w / 2;
            float y = cy - h / 2;

            x = (x - left) / scale;
            y = (y - top) / scale;
            w /= scale;
            h /= scale;

            Detection det;
            det.box = cv::Rect(static_cast<int>(x),
                               static_cast<int>(y),
                               static_cast<int>(w),
                               static_cast<int>(h));
            det.conf = outdata[4 * num_box + i * num_classes];
            Detections.push_back(det);
        }
    }

    result = nms(Detections , 0.60);
}

std::vector<Detection> nms(const std::vector<Detection> &Detections, float iou_threshold)
{
    std::vector<Detection> result; // nmms后的结果
    std::vector<bool> suppressed(Detections.size(), false);

    std::vector<size_t> index(Detections.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [Detections](size_t i1, size_t i2)
              { return Detections[i1].conf > Detections[i1].conf; });

    for (size_t i = 0; i < index.size(); ++i)
    {
        if (suppressed[index[i]])
            continue;
        result.push_back(Detections[index[i]]);
        for (size_t j = i + 1; j < index.size(); ++i)
        {
            if (suppressed[index[j]])
                continue;
            const cv::Rect &a = Detections[index[i]].box;
            const cv::Rect &b = Detections[index[j]].box;

            float area_a = a.area();
            float area_b = b.area();

            cv::Rect intersection = a & b;
            float area_intersection = intersection.area();

            float iou = area_intersection / (area_a + area_b - area_intersection);
            if (iou > iou_threshold)
            {
                suppressed[index[j]] = true;
            }
        }
    }

    return result;

}

bool toe::ov_detect_base::show_results(cv::Mat &img , const std::vector<Detection> &detections)
{
     // 定义颜色
    static const cv::Scalar colors = cv::Scalar(128, 128, 0);
    
    for (const auto& det : detections) {
        // 绘制边界框
        cv::rectangle(img, det.box, colors, 2);
        
        // 创建标签文本
        std::string label;
            label = cv::format("%.2f", det.conf);
        
        // 计算文本位置
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // 绘制文本
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    return true;
}

/// @brief
/// @param src
/// @return
cv::Rect2f toe::ov_detect::GetBoundingBox(const cv::Rect2f &src) const
{
    cv::Rect2f box = src;

    box.width = 1.0 * (box.width - box.x) * scale_factor_.x;
    box.height = 1.0 * (box.height - box.y) * scale_factor_.y;

    box.x *= scale_factor_.x;
    box.y *= scale_factor_.y;

    return box;
}

// 推送图像到队列中
void toe::ov_detect_base::push_img(const cv::Mat &img)
{
    img_mutex_.lock();
    if (input_imgs.size() >= max_size_)
    {
        input_imgs.clear();
    }
    input_imgs.emplace_back(img.clone());
    img_mutex_.unlock();
}



// 获取输出结果进到olleyball_data中
std::vector<volleyball_data> toe::ov_detect_base::get_results()
{
    std::vector<volleyball_data> temp_return;
    outputs_mutex_.lock();
    temp_return = output_data;
    outputs_mutex_.unlock();
    return temp_return;
}

bool toe::ov_detect::detect(void)
{
    if (!input_imgs.empty())
    {
        // std::cout << "read" << std::endl;
        img_mutex_.lock();
        input_img = input_imgs.back();
        input_imgs.clear();
        img_mutex_.unlock();

        std::cout << "trans" << std::endl;
        preprocess();
        std::cout << "infer" << std::endl;
        inference();
        std::cout << "post" << std::endl;
        postprocess();
    }
    return true;
}

/// @brief 用于将图片填充为640x640
/// @param source 输入图像
/// @return 返回变化后的输出图像
cv::Mat toe::ov_detect::letterbox(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect2f(0, 0, col, row)));
    return result;
}

/// @brief 取得关注区域
/// @param img 输入图像
/// @param result 返回关注区域的坐标
/// @return 返回关注区域的坐标
cv::Rect2f toe::ov_detect::getROI(cv::Mat img, bbox result)
{

    float x1 = result.x1;
    float y1 = result.y1;
    float x2 = result.x2;
    float y2 = result.y2;
    float width = x2 - x1;
    float height = y2 - y1;

    return cv::Rect2f(x1, y1, width, height);
}

// /// @brief 运行推理
// /// @param img 输入要推理的图像
// /// @param rois roi的坐标容器
// /// @param debugImg 可视化的输出图像
// void toe::ov_detect::StartInference(const cv::Mat img, std::vector<cv::Rect2f> &rois, cv::Mat &debugImg)
// {
//     cv::Mat letterbox_img = letterbox(img);
//     float scale = letterbox_img.size[0] / 640;
//     cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
//     // -------- Step 5. Feed the blob into the input node of the Model -------
//     // Get input port for model with one input
//     auto input_port = compiled_model.input();
//     // std::cout << "DEBUGG" << std::endl;
//     //  Create tensor from external memory
//     ov::Tensor input_tensor1(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
//     // Set input tensor for model with one input
//     infer_request.set_input_tensor(input_tensor1);

//     // -------- Step 6. Start inference --------
//     infer_request.infer();

//     // -------- Step 7. Get the inference result --------
//     output_tensor = infer_request.get_output_tensor(0);
//     output_shape = output_tensor.get_shape();
//     auto rows = output_shape[2];       // 8400
//     auto dimensions = output_shape[1]; // 84: box[cx, cy, w, h]+80 classes scores

//     // -------- Step 8. Postprocess the result --------
//     auto *data = output_tensor.data<float>();
//     cv::Mat output_buffer(dimensions, rows, CV_32F, data);
//     cv::transpose(output_buffer, output_buffer); //[8400,84]

//     std::vector<int> class_ids;
//     std::vector<float> class_scores;
//     std::vector<cv::Rect> boxes;
//     std::vector<cv::Rect2f> boxes2f;
//     // Figure out the bbox, class_id and class_score
//     int outputBufferRows = output_buffer.rows;
//     for (int i = 0; i < outputBufferRows; i++)
//     {
//         cv::Mat classes_scores = output_buffer.row(i).colRange(4, dimensions);
//         cv::Point class_id;
//         double maxClassScore;
//         cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

//         if (maxClassScore > param_.bbox_conf_thresh)
//         {
//             class_scores.push_back(maxClassScore);
//             class_ids.push_back(class_id.x);
//             float cx = output_buffer.at<float>(i, 0);
//             float cy = output_buffer.at<float>(i, 1);
//             float w = output_buffer.at<float>(i, 2);
//             float h = output_buffer.at<float>(i, 3);
//             float left = float((cx - 0.5 * w) * scale);
//             float top = float((cy - 0.5 * h) * scale);
//             float width = float(w * scale);
//             float height = float(h * scale);
//             boxes.push_back(cv::Rect(left, top, width, height));
//             boxes2f.push_back(cv::Rect2f(left, top, width, height));
//         }
//     }
//     // NMS
//     std::vector<int> indices;
//     cv::dnn::NMSBoxes(boxes, class_scores, param_.bbox_conf_thresh, param_.nms_thresh, indices);

//     cv::Mat draw_img = img.clone();
//     bbox result;
//     for (size_t i = 0; i < indices.size(); i++)
//     {
//         int index = indices[i];
//         result.x1 = boxes2f[index].tl().x;
//         result.y1 = boxes2f[index].tl().y; // top left
//         result.x2 = boxes2f[index].br().x;
//         result.y2 = boxes2f[index].br().y; // bottom right
//         result.class_id = class_ids[index];
//         result.score = class_scores[index];
//         //  visualizeResult(draw_img, result);
//         cv::Rect2f item;
//         item = getROI(img, result);

//         // std::cout<<"color id: "<<result.class_id<< std::endl;
//         rois.emplace_back(item);
//     }
//     // 画出roi
//     for (auto roi : rois)
//     {
//         rectangle(debugImg, roi, cv::Scalar(255, 255, 0), 2);
//     }
//     // if (DEBUG == 1)
//     //{
//     //     imshow("result", debugImg);
//     //     cv::waitKey(0);
//     // }
// }

/*

// ... existing code ...
#include <deque>

namespace toe
{
    // ... existing code ...
    
    // 添加卡尔曼滤波器类
    class KalmanBoxTracker {
    public:
        KalmanBoxTracker() {
            // 状态向量：[x, y, w, h, vx, vy, vw, vh]
            kalman = cv::KalmanFilter(8, 4, 0);
            kalman.transitionMatrix = (cv::Mat_<float>(8, 8) << 
                1, 0, 0, 0, 1, 0, 0, 0,
                0, 1, 0, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 1);
            
            // 测量矩阵：只测量位置和大小
            kalman.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
            kalman.measurementMatrix.at<float>(0, 0) = 1;
            kalman.measurementMatrix.at<float>(1, 1) = 1;
            kalman.measurementMatrix.at<float>(2, 2) = 1;
            kalman.measurementMatrix.at<float>(3, 3) = 1;
            
            // 过程噪声协方差
            setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-2));
            
            // 测量噪声协方差
            setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
            
            // 后验错误协方差
            setIdentity(kalman.errorCovPost, cv::Scalar::all(1));
            
            initialized = false;
            age = 0;
            hits = 0;
            hit_streak = 0;
            time_since_update = 0;
        }
        
        cv::Rect update(const cv::Rect& bbox) {
            time_since_update = 0;
            hits += 1;
            hit_streak += 1;
            
            // 测量值：[x, y, w, h]
            cv::Mat measurement = (cv::Mat_<float>(4, 1) << 
                                 bbox.x, bbox.y, bbox.width, bbox.height);
            
            if (!initialized) {
                // 初始化状态
                kalman.statePost.at<float>(0) = bbox.x;
                kalman.statePost.at<float>(1) = bbox.y;
                kalman.statePost.at<float>(2) = bbox.width;
                kalman.statePost.at<float>(3) = bbox.height;
                kalman.statePost.at<float>(4) = 0;  // vx
                kalman.statePost.at<float>(5) = 0;  // vy
                kalman.statePost.at<float>(6) = 0;  // vw
                kalman.statePost.at<float>(7) = 0;  // vh
                initialized = true;
            } else {
                // 预测
                kalman.predict();
                // 更新
                kalman.correct(measurement);
            }
            
            age += 1;
            
            // 返回滤波后的边界框
            return cv::Rect(kalman.statePost.at<float>(0),
                           kalman.statePost.at<float>(1),
                           kalman.statePost.at<float>(2),
                           kalman.statePost.at<float>(3));
        }
        
        cv::Rect predict() {
            if (!initialized) {
                return cv::Rect();
            }
            
            cv::Mat prediction = kalman.predict();
            time_since_update += 1;
            
            if (time_since_update > 0) {
                hit_streak = 0;
            }
            
            return cv::Rect(prediction.at<float>(0),
                           prediction.at<float>(1),
                           prediction.at<float>(2),
                           prediction.at<float>(3));
        }
        
    private:
        cv::KalmanFilter kalman;
        bool initialized;
        int age;
        int hits;
        int hit_streak;
        int time_since_update;
    };
    
    class ov_detect : public ov_detect_base
    {
    public:
        // ... existing code ...
        
    private:
        // ... existing code ...
        
        // 添加卡尔曼滤波器
        KalmanBoxTracker tracker;
        bool use_kalman_filter = true;
        
        // ... existing code ...
    };
}
// ... existing code ...

*/
