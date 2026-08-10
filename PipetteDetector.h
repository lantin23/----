#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class PipetteDetector
{
public:
    PipetteDetector();

    // 设置参数
    void setMinArea(double area) { m_minArea = area; }
    void setMaxArea(double area) { m_maxArea = area; }
    void setAngleThreshold(double thresh) { m_angleThresh = thresh; }
    void setSpaceRatio(double ratio) { m_spaceRatio = ratio; }
    void setMinTips(int count) { m_minTips = count; }

    // 获取参数（添加这些缺失的方法）
    double getMinArea() const { return m_minArea; }
    double getMaxArea() const { return m_maxArea; }
    double getAngleThreshold() const { return m_angleThresh; }
    double getSpaceRatio() const { return m_spaceRatio; }
    int getMinTips() const { return m_minTips; }

    // 检测主函数
    bool detect(const cv::Mat& frame);
    void draw(cv::Mat& frame);

    // 结果获取
    bool isNeat() const { return m_isNeat; }
    std::string getReason() const { return m_reason; }
    std::vector<cv::Point2f> getCenters() const { return m_centers; }
    std::vector<double> getAngles() const { return m_angles; }

private:
    bool checkNeatness();
    void clearData();
    bool preprocessImage(const cv::Mat& frame, cv::Mat& binary, cv::Mat& edges);
    bool isGoodContour(const std::vector<cv::Point>& contour);
    double calculateAngle(const std::vector<cv::Point>& contour);
    cv::Point2f calculateCenter(const std::vector<cv::Point>& contour);
    bool checkAngleConsistency();
    bool checkSpacingUniformity();

    // 检测数据
    std::vector<cv::Point2f> m_centers;
    std::vector<double> m_angles;
    bool m_isNeat;
    std::string m_reason;

    // 可调参数
    double m_minArea;         // 最小面积（像素）
    double m_maxArea;         // 最大面积（像素）
    double m_angleThresh;     // 角度偏差阈值（度）
    double m_spaceRatio;      // 间距允许偏差比例
    int m_minTips;            // 最少枪头数量

    // 图像处理参数
    const int m_gaussianKernel = 7;
    const int m_cannyThreshold1 = 30;
    const int m_cannyThreshold2 = 90;
    const float m_yTolerance = 15.0f;  // 同一行的Y坐标容差（像素）
    const float m_minSpace = 15.0f;    // 最小间距（像素）
};