#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

class YOLO_KalmanTracker {
public:
    YOLO_KalmanTracker();
    void init(float x, float y, float width, float height);
    void update(float x, float y, float width, float height);
    void predict();
    Point2f getPredictedCenter();
    Rect getPredictedRect();

private:
    KalmanFilter kalmanFilter;
    Mat state;
    bool isInitialized;
};

YOLO_KalmanTracker::YOLO_KalmanTracker() : isInitialized(false) {
    // 初始化卡尔曼滤波器
    kalmanFilter = KalmanFilter(4, 2, 0);
    // 状态转移矩阵
    kalmanFilter.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    // 测量矩阵
    setIdentity(kalmanFilter.measurementMatrix);
    // 过程噪声协方差矩阵
    setIdentity(kalmanFilter.processNoiseCov, Scalar::all(1e-4));
    // 测量噪声协方差矩阵
    setIdentity(kalmanFilter.measurementNoiseCov, Scalar::all(1e-1));
    // 误差协方差后验矩阵
    setIdentity(kalmanFilter.errorCovPost, Scalar::all(1));
    // 初始状态
    state = Mat::zeros(4, 1, CV_32F);
}

void YOLO_KalmanTracker::init(float x, float y, float width, float height) {
    // 初始化状态向量
    state.at<float>(0) = x;
    state.at<float>(1) = y;
    state.at<float>(2) = width;
    state.at<float>(3) = height;
    kalmanFilter.statePost = state.clone();
    isInitialized = true;
}

void YOLO_KalmanTracker::update(float x, float y, float width, float height) {
    if (!isInitialized) {
        init(x, y, width, height);
        return;
    }
    // 更新测量值
    Mat measurement = (Mat_<float>(2, 1) << x, y);
    // 更新卡尔曼滤波器
    kalmanFilter.correct(measurement);
    // 更新状态
    state = kalmanFilter.statePost;
}

void YOLO_KalmanTracker::predict() {
    if (!isInitialized) return;
    // 预测
    kalmanFilter.predict();
    // 获取预测状态
    state = kalmanFilter.statePre;
}

Point2f YOLO_KalmanTracker::getPredictedCenter() {
    if (!isInitialized) return Point2f(-1, -1);
    return Point2f(state.at<float>(0), state.at<float>(1));
}

Rect YOLO_KalmanTracker::getPredictedRect() {
    if (!isInitialized) return Rect();
    float x = state.at<float>(0) - state.at<float>(2) / 2;
    float y = state.at<float>(1) - state.at<float>(3) / 2;
    return Rect(x, y, state.at<float>(2), state.at<float>(3));
}

// int main() {
//     YOLO_KalmanTracker tracker;
//     VideoCapture cap("input.mp4"); // 替换为你的视频文件路径
//     if (!cap.isOpened()) {
//         cout << "Error opening video file" << endl;
//         return -1;
//     }

//     Mat frame;
//     while (cap.read(frame)) {
//         // 假设此处已通过 YOLO 检测到排球的边界框
//         vector<Rect> detections;
//         // 模拟检测到的排球位置，实际应用中应替换为 YOLO 检测结果
//         detections.push_back(Rect(100, 100, 50, 50));

//         if (!detections.empty()) {
//             Rect ball = detections[0];
//             Point2f center = Point2f(ball.x + ball.width / 2, ball.y + ball.height / 2);
//             tracker.update(center.x, center.y, ball.width, ball.height);
//         } else {
//             tracker.predict();
//         }

//         // 绘制预测结果
//         Point2f predictedCenter = tracker.getPredictedCenter();
//         Rect predictedRect = tracker.getPredictedRect();
//         circle(frame, predictedCenter, 5, Scalar(0, 0, 255), -1);
//         rectangle(frame, predictedRect, Scalar(0, 255, 0), 2);

//         imshow("Frame", frame);
//         if (waitKey(30) >= 0) break;
//     }

//     return 0;
// }