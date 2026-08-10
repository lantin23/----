#include "PipetteDetector.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

PipetteDetector::PipetteDetector()
    : m_isNeat(false),
    m_reason(""),
    m_minArea(80.0),
    m_maxArea(5000.0),
    m_angleThresh(6.0),
    m_spaceRatio(0.18),
    m_minTips(4)
{
}

void PipetteDetector::clearData()
{
    m_centers.clear();
    m_angles.clear();
    m_isNeat = false;
    m_reason.clear();
}

bool PipetteDetector::preprocessImage(const Mat& frame, Mat& binary, Mat& edges)
{
    if (frame.empty()) return false;

    Mat gray, blur;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(m_gaussianKernel, m_gaussianKernel), 0);

    // 使用 Otsu 二值化
    threshold(blur, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 形态学操作：去除小噪点，连接断裂区域
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    // 边缘检测
    Canny(binary, edges, m_cannyThreshold1, m_cannyThreshold2);

    return true;
}

bool PipetteDetector::isGoodContour(const vector<Point>& contour)
{
    double area = contourArea(contour);
    if (area < m_minArea || area > m_maxArea) return false;

    // 检查轮廓形状：矩形度（面积 / 外接矩形面积）
    Rect boundingRect = cv::boundingRect(contour);
    double rectArea = boundingRect.width * boundingRect.height;
    double rectangularity = area / rectArea;
    if (rectangularity < 0.4) return false;  // 太细长或不规则

    // 检查轮廓凸度
    double perimeter = arcLength(contour, true);
    if (perimeter <= 0) return false;

    // 可选：检查周长与面积的比例（圆形度）
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    if (circularity < 0.3) return false;  // 不是近似圆形/椭圆形

    return true;
}

Point2f PipetteDetector::calculateCenter(const vector<Point>& contour)
{
    Moments m = moments(contour);
    if (m.m00 == 0) return Point2f(0, 0);
    return Point2f((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
}

double PipetteDetector::calculateAngle(const vector<Point>& contour)
{
    RotatedRect rect = minAreaRect(contour);
    double angle = rect.angle;

    // 调整角度范围到 [-45, 45] 度
    if (angle > 45) angle -= 90;
    if (angle < -45) angle += 90;

    return angle;
}

bool PipetteDetector::detect(const Mat& frame)
{
    clearData();
    if (frame.empty()) return false;

    Mat binary, edges;
    if (!preprocessImage(frame, binary, edges)) return false;

    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 提取有效枪头信息
    for (const auto& contour : contours)
    {
        if (!isGoodContour(contour)) continue;

        Point2f center = calculateCenter(contour);
        if (center.x == 0 && center.y == 0) continue;

        double angle = calculateAngle(contour);

        m_centers.push_back(center);
        m_angles.push_back(angle);
    }

    // 检查检测结果
    m_isNeat = checkNeatness();
    return m_isNeat;
}

bool PipetteDetector::checkAngleConsistency()
{
    if (m_angles.empty()) return false;

    // 计算平均角度
    double sumAngle = 0.0;
    for (double angle : m_angles) sumAngle += angle;
    double avgAngle = sumAngle / m_angles.size();

    // 检查每个角度的偏差
    for (double angle : m_angles)
    {
        double deviation = fabs(angle - avgAngle);
        if (deviation > m_angleThresh)
        {
            m_reason = "角度歪斜 (偏差: " + to_string(deviation) + "°)";
            return false;
        }
    }

    return true;
}

bool PipetteDetector::checkSpacingUniformity()
{
    if (m_centers.empty()) return false;

    // 按 Y 坐标分桶（分成不同的行）
    vector<vector<Point2f>> rows;

    for (const auto& point : m_centers)
    {
        bool placed = false;
        for (auto& row : rows)
        {
            // 如果当前行的第一个点的 Y 坐标与当前点接近
            if (!row.empty() && fabs(row[0].y - point.y) < m_yTolerance)
            {
                row.push_back(point);
                placed = true;
                break;
            }
        }
        if (!placed)
        {
            rows.push_back({ point });
        }
    }

    // 对每一行进行检查
    for (auto& row : rows)
    {
        // 一行至少需要2个点才能计算间距
        if (row.size() < 2) continue;

        // 按 X 坐标排序
        sort(row.begin(), row.end(), [](const Point2f& a, const Point2f& b) {
            return a.x < b.x;
            });

        // 计算相邻点之间的间距
        vector<double> spaces;
        for (size_t i = 1; i < row.size(); i++)
        {
            double dx = row[i].x - row[i - 1].x;
            if (dx > m_minSpace)
            {
                spaces.push_back(dx);
            }
        }

        if (spaces.empty()) continue;

        // 计算平均间距
        double sumSpace = 0.0;
        for (double space : spaces) sumSpace += space;
        double avgSpace = sumSpace / spaces.size();

        // 检查间距均匀性
        for (double space : spaces)
        {
            double error = fabs(space - avgSpace) / avgSpace;
            if (error > m_spaceRatio)
            {
                m_reason = "间距不均匀 (误差: " + to_string(error * 100) + "%)";
                return false;
            }
        }
    }

    return true;
}

bool PipetteDetector::checkNeatness()
{
    // 检查数量
    if (m_centers.size() < m_minTips)
    {
        m_reason = "枪头数量不足 (检测到: " + to_string(m_centers.size()) + ", 需要: " + to_string(m_minTips) + ")";
        return false;
    }

    // 检查角度一致性
    if (!checkAngleConsistency()) return false;

    // 检查间距均匀性
    if (!checkSpacingUniformity()) return false;

    m_reason = "整齐 ✓";
    return true;
}

void PipetteDetector::draw(Mat& frame)
{
    if (frame.empty()) return;

    // 绘制每个枪头的中心点和方向
    for (size_t i = 0; i < m_centers.size(); i++)
    {
        const auto& center = m_centers[i];

        // 绘制中心点
        circle(frame, center, 5, Scalar(0, 255, 0), -1);
        circle(frame, center, 7, Scalar(255, 255, 255), 2);

        // 绘制角度方向线（用于调试）
        if (i < m_angles.size())
        {
            double angleRad = m_angles[i] * CV_PI / 180.0;
            Point2f endPoint(
                center.x + 20 * cos(angleRad),
                center.y + 20 * sin(angleRad)
            );
            line(frame, center, endPoint, Scalar(0, 255, 255), 2);
        }

        // 显示编号
        putText(frame, to_string(i + 1),
            Point(center.x - 8, center.y - 8),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
    }

    // 显示检测结果
    string resultText = m_isNeat ? "✓ TIDY" : "✗ MESSY: " + m_reason;
    Scalar textColor = m_isNeat ? Scalar(0, 255, 0) : Scalar(0, 0, 255);

    // 背景框
    rectangle(frame, Point(10, 10), Point(450, 80), Scalar(0, 0, 0), -1);
    rectangle(frame, Point(10, 10), Point(450, 80), Scalar(255, 255, 255), 2);

    putText(frame, resultText, Point(20, 50),
        FONT_HERSHEY_SIMPLEX, 0.9, textColor, 2);

    // 显示统计信息
    string info = "Tips: " + to_string(m_centers.size());
    putText(frame, info, Point(20, 75),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
}