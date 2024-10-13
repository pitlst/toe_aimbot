#include "detect.hpp"
#include "opencv2/objdetect.hpp"
#include "openvino/openvino.hpp"

#include <iostream>
#include <string>
#include <vector>

bool toe::ov_detect::detect_init(const nlohmann::json &input_json)
{
    nlohmann::json temp_json = input_json;

    /*
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
    */
    compiled_model = core.compile_model("/home/toe-volleyball/toe_aimbot/data/best_openvino_model/best.xml", "CPU");
    infer_request = compiled_model.create_infer_request();

    auto out_node = compiled_model.outputs();
    out_tensor_size = out_node.size();
    /*
        for (auto out_n : out_node)
        {
            auto out_name = out_n.get_any_name();
            if (out_name == "stride_8")
            {
                anchors.emplace_back(param_.a2);
                stride_.emplace_back(8);
            }
            else if (out_name == "stride_16")
            {
                anchors.emplace_back(param_.a3);
                stride_.emplace_back(16);
            }
            else if (out_name == "stride_32")
            {
                anchors.emplace_back(param_.a4);
                stride_.emplace_back(32);
            }
        }
    */
    stride_.emplace_back(8);
    std::cout << "network_init_done. " << std::endl;
    blob.resize(640 * 640 * 3);
    return true;
}

void toe::ov_detect::preprocess(void)
{

    ov::Tensor input_tensor = infer_request.get_input_tensor();
    auto data1 = input_tensor.data<float>();
    // 输入图像预处理

    cv::resize(input_image, input_image, cv::Size(640, 640));
    // 归一化
    int img_h = input_image.rows;
    int img_w = input_image.cols;
    float *blob_data = blob.data();
    size_t i = 0;
    for (size_t row = 0; row < img_h; ++row)
    {
        uchar *uc_pixel = input_image.data + row * input_image.step;
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

void toe::ov_detect::postprocess(void)
{
    output_data_.clear();
    // 解码网络输出
    for (size_t i = 0; i < out_tensor_size; i++)
    {
        // 获取输出tensor的指针
        ov::Tensor output_tensor = infer_request.get_output_tensor(i);
        const float *out_data = output_tensor.data<float>();

        int nums = 0;
        int now_stride = stride_[i];
        // std::cout<<"1111111111111"<<std::endl;
        std::vector<float> *l_anchor = &(anchors[i]);
        int out_h = 640 / now_stride;
        int out_w = 640 / now_stride;
        int num_out = 5 + 10 + 1;
        float pred_data[num_out] = {0};
        // 图像三通道
        for (int na = 0; na < 3; ++na)
        {
            for (int h_id = 0; h_id < out_h; ++h_id)
            {
                for (int w_id = 0; w_id < out_w; ++w_id)
                {
                    pred_data[num_out] = {0};
                    int data_idx = (na * out_h * out_w + h_id * out_w + w_id) * num_out;
                    // 计算当前框的目标存在置信度
                    double obj_conf = toe::sigmoid(out_data[data_idx + 4]);
                    if (obj_conf > 0.8)
                    {
                        toe::sigmoid(out_data + data_idx, pred_data, 5);
                        toe::sigmoid(out_data + data_idx + 15, pred_data + 15, 1);
                        std::memcpy(pred_data + 5, out_data + data_idx + 5, sizeof(float) * 10);

                        // 计算当前框的颜色
                        int col_id = std::max_element(pred_data + 15 + 1,
                                                      pred_data + 15 + 1) -
                                     (pred_data + 15 + 1);

                        // 计算当前框的类别

                        int cls_id = std::max_element(pred_data + 15, pred_data + 15 + 1) - (pred_data + 15);

                        std::cout << cls_id << std::endl;
                        // 计算当前框的最终置信度
                        double final_conf = obj_conf * std::pow(pred_data[15 + cls_id] *
                                                                    pred_data[15 + 1 + col_id] *
                                                                    pred_data[15 + 1],
                                                                1 / 3.);

                        if (final_conf > 0.8)
                        {
                            nums++;
                            volleyball_data now;
                            float x = (pred_data[0] * 2.0 - 0.5 + w_id) * now_stride;
                            float y = (pred_data[1] * 2.0 - 0.5 + h_id) * now_stride;
                            float w = std::pow(pred_data[2] * 2, 2) * l_anchor->at(na * 2);
                            float h = std::pow(pred_data[3] * 2, 2) * l_anchor->at(na * 2 + 1);
                            for (int p = 0; p < 5; ++p)
                            {
                                float px = (pred_data[5 + p * 2] * l_anchor->at(na * 2) + w_id * now_stride);
                                float py = (pred_data[5 + p * 2 + 1] * l_anchor->at(na * 2 + 1) + h_id * now_stride);
                                px = std::max(std::min(px, (float)(640)), 0.f);
                                py = std::max(std::min(py, (float)(640)), 0.f);
                                now.pts[p] = cv::Point2f(px, py);
                            }
                            float x0 = x - w * 0.5;
                            float y0 = y - h * 0.5;
                            float x1 = x + w * 0.5;
                            float y1 = y + h * 0.5;

                            // 检查输出目标的上下界，应当在640和0之间
                            x0 = std::max(std::min(x0, (float)(640.0f)), 0.f);
                            y0 = std::max(std::min(y0, (float)(640.0f)), 0.f);
                            x1 = std::max(std::min(x1, (float)(640.0f)), 0.f);
                            y1 = std::max(std::min(y1, (float)(640.0f)), 0.f);

                            now.ball_x = (x0 + x1) / 2;
                            now.ball_y = (y0 + y1) / 2;
                            now.rect = cv::Rect(x0, y0, x1 - x0, y1 - y0);
                            // now.conf = final_conf;
                            // now.color = col_id;
                            now.type = cls_id;
                            // now.t_size = t_size;
                            output_data_.emplace_back(now);
                        }
                    }
                }
            }
        }
    }

    std::cout << "output_data_ is " << output_data_.size() << std::endl;

    // nms去除重叠装甲板
    output_data.clear();
    std::vector<pick_merge_store> picked;
    std::sort(output_data_.begin(), output_data_.end(), [](const volleyball_data &a, const volleyball_data &b)
              { return a.conf > b.conf; });
    for (int i = 0; i < output_data_.size(); ++i)
    {
        volleyball_data &now = output_data_[i];
        bool keep = true;
        for (int j = 0; j < picked.size(); ++j)
        {
            volleyball_data &pre = output_data_[picked[j].id];
            float iou = calc_iou(now, pre);
            if (iou > 0.8)
            {
                keep = false;
                if (now.type == pre.type && iou > 0.8)
                {
                    picked[j].merge_confs.push_back(now.conf);
                    for (int k = 0; k < 5; ++k)
                    {
                        picked[j].merge_pts.push_back(now.pts[k]);
                    }
                }
                break;
            }
        }
        if (keep)
        {
            picked.push_back({i, {}, {}});
        }
    }

    // 根据置信度对关键点做加权平均，矫正关键点位置
    for (int i = 0; i < picked.size(); ++i)
    {
        int merge_num = picked[i].merge_confs.size();
        volleyball_data now = output_data_[picked[i].id];
        double conf_sum = now.conf;
        for (int j = 0; j < 5; ++j)
        {
            now.pts[j] *= now.conf;
        }
        for (int j = 0; j < merge_num; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                now.pts[k] += picked[i].merge_pts[j * 5 + k] * picked[i].merge_confs[j];
            }
            conf_sum += picked[i].merge_confs[j];
        }
        for (int j = 0; j < 5; ++j)
        {
            now.pts[j] /= conf_sum;
        }
        output_data.emplace_back(now);
    }
}

void toe::ov_detect::inference(void)
{
    // 推理
    infer_request.infer();
}

// 推送图像到队列中
void toe::ov_detect::push_img(const cv::Mat &img)
{
    img_mutex_.lock();
    if (input_imgs.size() >= max_size_)
    {
        input_imgs.clear();
    }
    input_imgs.emplace_back(img.clone());
    img_mutex_.unlock();
}

bool toe::ov_detect::show_results(cv::Mat &img)
{
    cv::resize(img, img, cv::Size(640, 640));
    cv::Point ct0, ct1, ct2, ct3;
    for (auto i = 0; i < output_data.size(); i++)
    {
        cv::rectangle(img, output_data.at(i).rect, cv::Scalar(0, 255, 0), 2);
    }
    return true;
}

// 获取输出结果进到olleyball_data中
std::vector<volleyball_data> toe::ov_detect::get_results()
{
    std::vector<volleyball_data> temp_return;
    output_data_mutex.lock();
    temp_return = output_data_;
    output_data_mutex.unlock();
    return temp_return;
}

bool toe::ov_detect::detect()
{
    if (!input_imgs.empty())
    {
        // std::cout << "read" << std::endl;
        input_image_mutex.lock();
        input_image = input_imgs.back();
        input_imgs.clear();
        input_image_mutex.unlock();

        // std::cout << "trans" << std::endl;
        preprocess();
        // std::cout << "infer" << std::endl;
        inference();
        // std::cout << "post" << std::endl;
        postprocess();
    }
    return true;
}
