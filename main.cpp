#include <iostream>
#include <opencv2/opencv.hpp>
#include "PipetteDetector.h"

using namespace cv;
using namespace std;

static void printHelp()
{
    cout << "\n========================================" << endl;
    cout << "       枪头整齐度检测系统 v2.0" << endl;
    cout << "========================================" << endl;
    cout << "功能：检测移液枪头是否排列整齐" << endl;
    cout << "检测项：数量、角度一致性、间距均匀性" << endl;
    cout << "========================================\n" << endl;
}

static void showControls()
{
    cout << "\n操作说明：" << endl;
    cout << "  ESC    - 退出程序" << endl;
    cout << "  SPACE  - 保存当前截图" << endl;
    cout << "  c      - 清除统计信息" << endl;
    cout << "  h      - 显示帮助" << endl;
}

static bool saveScreenshot(const Mat& frame, int count)
{
    string filename = "screenshot_" + to_string(count) + ".jpg";
    bool success = imwrite(filename, frame);
    if (success)
    {
        cout << "✓ 截图已保存: " << filename << endl;
    }
    else
    {
        cout << "✗ 截图保存失败" << endl;
    }
    return success;
}

static void processImageMode(PipetteDetector& detector)
{
    cout << "\n--- 单张图片检测模式 ---" << endl;
    cout << "支持的格式: jpg, png, bmp" << endl;
    cout << "请输入图片路径: ";

    string imgPath;
    cin.ignore();
    getline(cin, imgPath);

    // 去除路径两端的空格和引号
    imgPath.erase(0, imgPath.find_first_not_of(" \t\n\r\f\v\"'"));
    imgPath.erase(imgPath.find_last_not_of(" \t\n\r\f\v\"'") + 1);

    Mat img = imread(imgPath);
    if (img.empty())
    {
        cout << "✗ 图片读取失败！请检查路径: " << imgPath << endl;
        return;
    }

    cout << "✓ 图片加载成功 (尺寸: " << img.cols << "x" << img.rows << ")" << endl;

    detector.detect(img);
    detector.draw(img);

    // 显示结果
    imshow("检测结果", img);
    cout << "\n检测结果: " << (detector.isNeat() ? "整齐 ✓" : "不整齐 ✗") << endl;
    cout << "原因: " << detector.getReason() << endl;
    cout << "检测到枪头数量: " << detector.getCenters().size() << endl;

    cout << "\n按任意键继续..." << endl;
    waitKey(0);
    destroyWindow("检测结果");
}

static void processCameraMode(PipetteDetector& detector)
{
    cout << "\n--- 实时摄像头检测模式 ---" << endl;
    cout << "正在打开摄像头..." << endl;

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "✗ 摄像头打开失败！请检查：" << endl;
        cout << "  1. 摄像头是否已连接" << endl;
        cout << "  2. 摄像头驱动是否正常" << endl;
        cout << "  3. 是否被其他程序占用" << endl;
        return;
    }

    // 设置摄像头参数
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);

    cout << "✓ 摄像头已启动" << endl;
    showControls();

    Mat frame;
    int screenshotCount = 0;
    bool showHelp = false;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // 镜像翻转（更自然）
        flip(frame, frame, 1);

        // 执行检测
        detector.detect(frame);
        detector.draw(frame);

        // 显示帮助信息（如果需要）
        if (showHelp)
        {
            rectangle(frame, Point(10, 100), Point(400, 280), Scalar(0, 0, 0), -1);
            rectangle(frame, Point(10, 100), Point(400, 280), Scalar(255, 255, 255), 2);

            putText(frame, "Controls:", Point(20, 130), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
            putText(frame, "ESC - Exit", Point(20, 160), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
            putText(frame, "SPACE - Screenshot", Point(20, 185), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
            putText(frame, "c - Clear info", Point(20, 210), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
            putText(frame, "h - Hide help", Point(20, 235), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        }

        // 显示帧率信息（简单计算）
        static int frameCount = 0;
        static double lastTime = getTickCount();
        frameCount++;
        double currentTime = getTickCount();
        double timeDiff = (currentTime - lastTime) / getTickFrequency();
        if (timeDiff >= 1.0)
        {
            double fps = frameCount / timeDiff;
            lastTime = currentTime;
            frameCount = 0;

            string fpsText = "FPS: " + to_string((int)fps);
            putText(frame, fpsText, Point(frame.cols - 100, 30),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        }

        imshow("实时检测", frame);

        // 按键处理
        int key = waitKey(1);
        if (key == 27) // ESC
        {
            cout << "\n退出程序..." << endl;
            break;
        }
        else if (key == 32) // SPACE
        {
            saveScreenshot(frame, ++screenshotCount);
        }
        else if (key == 'c' || key == 'C')
        {
            cout << "清除统计信息" << endl;
        }
        else if (key == 'h' || key == 'H')
        {
            showHelp = !showHelp;
        }
    }

    cap.release();
    destroyAllWindows();
    cout << "摄像头已关闭" << endl;
}

static void configureParameters(PipetteDetector& detector)
{
    cout << "\n--- 参数配置 ---" << endl;
    cout << "当前参数:" << endl;
    cout << "  最小面积: " << detector.getMinArea() << endl;
    cout << "  最大面积: " << detector.getMaxArea() << endl;
    cout << "  角度阈值: " << detector.getAngleThreshold() << "°" << endl;
    cout << "  间距偏差: " << detector.getSpaceRatio() * 100 << "%" << endl;
    cout << "  最少枪头: " << detector.getMinTips() << endl;

    cout << "\n是否需要修改参数？(y/n): ";
    char choice;
    cin >> choice;

    if (choice == 'y' || choice == 'Y')
    {
        double val;
        cout << "请输入最小面积 (当前 " << detector.getMinArea() << "): ";
        cin >> val;
        if (val > 0) detector.setMinArea(val);

        cout << "请输入最大面积 (当前 " << detector.getMaxArea() << "): ";
        cin >> val;
        if (val > 0) detector.setMaxArea(val);

        cout << "请输入角度阈值(度) (当前 " << detector.getAngleThreshold() << "): ";
        cin >> val;
        if (val > 0) detector.setAngleThreshold(val);

        cout << "请输入间距偏差比例 (0-1, 当前 " << detector.getSpaceRatio() << "): ";
        cin >> val;
        if (val > 0 && val < 1) detector.setSpaceRatio(val);

        int num;
        cout << "请输入最少枪头数量 (当前 " << detector.getMinTips() << "): ";
        cin >> num;
        if (num > 0) detector.setMinTips(num);

        cout << "✓ 参数已更新" << endl;
    }
}

int main()
{
    printHelp();

    PipetteDetector detector;

    // 可选：配置参数
    cout << "是否调整检测参数？(y/n): ";
    char configChoice;
    cin >> configChoice;
    if (configChoice == 'y' || configChoice == 'Y')
    {
        configureParameters(detector);
    }

    while (true)
    {
        cout << "\n请选择模式:" << endl;
        cout << "  1. 单张图片检测" << endl;
        cout << "  2. 实时摄像头检测" << endl;
        cout << "  3. 退出程序" << endl;
        cout << "请输入选择(1/2/3): ";

        int mode;
        cin >> mode;

        switch (mode)
        {
        case 1:
            processImageMode(detector);
            break;
        case 2:
            processCameraMode(detector);
            break;
        case 3:
            cout << "感谢使用，再见！" << endl;
            return 0;
        default:
            cout << "✗ 输入错误，请重新选择！" << endl;
            break;
        }
    }

    return 0;
}