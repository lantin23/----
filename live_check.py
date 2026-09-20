# -*- coding: utf-8 -*-
"""抓取实时摄像头一帧并运行检测，输出结果"""
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipette_detector import PipetteDetector


def main():
    index = 1
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        print("相机打开失败")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame = None
    for _ in range(40):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
        time.sleep(0.05)
    cap.release()

    if frame is None:
        print("抓不到帧（相机被占用？）")
        return

    det = PipetteDetector()
    det.detect(frame)

    print("帧尺寸: %dx%d" % (frame.shape[1], frame.shape[0]))
    print("圆形: %d" % len(det.result.centers))
    print("散落: %d" % len(det.result.fallen_tips))
    print("判定: %s" % ("整齐" if det.is_neat() else "不整齐"))
    print("原因: %s" % det.result.reason_text)
    for i, tip in enumerate(det.result.fallen_tips):
        cx, cy, bw, bh = tip[0], tip[1], tip[2], tip[3]
        ang = tip[4] if len(tip) >= 5 else 0.0
        print("  散落%d: (%d,%d) %dx%d ang=%.0f" % (i + 1, int(cx), int(cy), int(bw), int(bh), ang))

    # 保存结果图
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_result.png")
    cv2.imwrite(out, det.draw(frame))
    print("结果图: %s" % out)


if __name__ == "__main__":
    main()
