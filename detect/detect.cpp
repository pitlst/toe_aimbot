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
    ov_detect::compiled_model = core.compile_model(model, "CPU" );//,ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    // 创建推理请求

    infer_request = compiled_model.create_infer_request();
    // 因为网络是8，32，16的排列，所以anchor对应的排列需要更改

    auto out_node = compiled_model.outputs();

    //out_tensor_size = out_node.size();
    //std::cout << "out_tensor_size is " << out_tensor_size << std::endl;
    // 设定输入网络为FP16
    anchors.emplace_back(param_.a3);
    stride_.emplace_back(16);

////////////////////////////////////////
    short width, height;

    const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    const ov::Shape input_shape = inputs[0].get_shape();

    height = input_shape[1];
    width = input_shape[2];
    model_input_shape_ = cv::Size2f(width, height);

    const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
    const ov::Shape output_shape = outputs[0].get_shape();

    height = output_shape[1];
    width = output_shape[2];
    model_output_shape_ = cv::Size(width, height);

    std::cout << "network_init_done. " << std::endl;
    blob.resize(param_.w * param_.h * 3);
    return true;
    /////////////////////////////////
}

// 推理预处理部分
void toe::ov_detect::preprocess(void)
{

    ov::Tensor input_tensor = infer_request.get_input_tensor();
    auto data1 = input_tensor.data<float>();
    // 输入图像预处理
    // 干成640*640
    cv::resize(input_img, input_img, cv::Size(640, 640));
    // 归一化
    int img_h = input_img.rows;
    int img_w = input_img.cols;
    float *blob_data = blob.data();

    scale_factor_.x = static_cast<float>(input_img.cols / model_input_shape_.width);
    scale_factor_.y = static_cast<float>(input_img.rows / model_input_shape_.height);

    size_t i = 0;
    for (size_t row = 0; row < img_h; ++row)
    {
        uchar *uc_pixel = input_img.data + row * input_img.step;
        for (size_t col = 0; col < img_w; ++col)
        {
            // 三通道
            blob_data[i] = (float)uc_pixel[2] / 255.0;
            blob_data[i + img_h * img_w] = (float)uc_pixel[1] / 255.0;
            blob_data[i + 2 * img_h * img_w] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    // 复制数据
    std::memcpy(data1, blob.data(), sizeof(float) * blob.size());
}

void toe::ov_detect::inference(void)
{
    // 推理
    infer_request.infer();
}

void toe::ov_detect::postprocess(void)
{
   
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

bool toe::ov_detect_base::show_results(cv::Mat &img)
{
    return true;
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



bool toe::ov_detect::detect(std::vector<cv::Rect2f> &rois, cv::Mat &debugImg)
{
    if (!input_imgs.empty())
    {
        // std::cout << "read" << std::endl;
        img_mutex_.lock();
        input_img = input_imgs.back();
        input_imgs.clear();
        img_mutex_.unlock();

        StartInference(input_img , rois , debugImg);
        // std::cout << "trans" << std::endl;
        //preprocess();
        // std::cout << "infer" << std::endl;
        //inference();
        // std::cout << "post" << std::endl;
        //postprocess();
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



/// @brief 运行推理
/// @param img 输入要推理的图像
/// @param rois roi的坐标容器
/// @param debugImg 可视化的输出图像
void toe::ov_detect::StartInference(const cv::Mat img, std::vector<cv::Rect2f> &rois, cv::Mat &debugImg)
{
    cv::Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / 640;
    cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model.input();
    //std::cout << "DEBUGG" << std::endl;
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request.infer();

    // -------- Step 7. Get the inference result --------
    output_tensor = infer_request.get_output_tensor(0);
    output_shape = output_tensor.get_shape();
    auto rows = output_shape[2];       // 8400
    auto dimensions = output_shape[1]; // 84: box[cx, cy, w, h]+80 classes scores

    // -------- Step 8. Postprocess the result --------
    auto *data = output_tensor.data<float>();
    cv::Mat output_buffer(dimensions, rows, CV_32F, data);
    cv::transpose(output_buffer, output_buffer); //[8400,84]

    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect2f> boxes2f;
    // Figure out the bbox, class_id and class_score
    int outputBufferRows = output_buffer.rows;
    for (int i = 0; i < outputBufferRows; i++)
    {
        cv::Mat classes_scores = output_buffer.row(i).colRange(4, dimensions);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > param_.bbox_conf_thresh)
        {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            float left = float((cx - 0.5 * w) * scale);
            float top = float((cy - 0.5 * h) * scale);
            float width = float(w * scale);
            float height = float(h * scale);
            boxes.push_back(cv::Rect(left, top, width, height));
            boxes2f.push_back(cv::Rect2f(left, top, width, height));
        }
    }
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, param_.bbox_conf_thresh, param_.nms_thresh, indices);

    cv::Mat draw_img = img.clone();
    bbox result;
    for (size_t i = 0; i < indices.size(); i++)
    {
        int index = indices[i];
        result.x1 = boxes2f[index].tl().x;
        result.y1 = boxes2f[index].tl().y; // top left
        result.x2 = boxes2f[index].br().x;
        result.y2 = boxes2f[index].br().y; // bottom right
        result.class_id = class_ids[index];
        result.score = class_scores[index];
        //  visualizeResult(draw_img, result);
        cv::Rect2f item;
        item = getROI(img, result);

        //std::cout<<"color id: "<<result.class_id<< std::endl;
        rois.emplace_back(item);
    }
    // 画出roi
    for (auto roi : rois)
    {
        rectangle(debugImg, roi, cv::Scalar(255, 255, 0), 2);
    }
    //if (DEBUG == 1)
    //{
    //    imshow("result", debugImg);
    //    cv::waitKey(0);
    //}
}