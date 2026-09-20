"""
批量测试脚本：对exam目录下所有图片运行检测，输出对比表
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipette_detector import PipetteDetector


def safe_imread(path):
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    img_dir = r"C:\Users\23870\Pictures\exam"
    out_dir = r"C:\Users\23870\Pictures\result"

    # 分类整理
    tidy_files = ["y1.jpg", "y2.jpg", "y3.jpg", "y4.jpg", "y5.jpg", "y6.jpg", "y7.jpg", "y8.jpg", "y9.jpg", "y10.jpg", "y11.jpg", "y12.jpg","y14.jpg", "y15.jpg", "y16.jpg", "y17.jpg", "y18.jpg", "y19.jpg", "y20.jpg", "y21.jpg"]
    messy_files = ["n1.jpg", "n2.jpg", "n3.jpg", "n4.jpg", "n5.jpg", "n6.jpg", "n7.jpg", "n8.jpg", "n9.jpg", "n10.jpg", "n11.jpg", "n12.jpg","n14.jpg", "n15.jpg", "n16.jpg", "n17.jpg", "n18.jpg", "n19.jpg", "n20.jpg", "n21.jpg", "n22.jpg", "n23.jpg", "n24.jpg", "n25.jpg", "n26.jpg", "n27.jpg", "n28.jpg", "n29.jpg", "n30.jpg", "n31.jpg", "n32.jpg", "n33.jpg", "n34.jpg"]

    detector = PipetteDetector()

    print("=" * 80)
    print("批量检测")
    print("=" * 80)
    print(f"{'文件':<12} {'期望':<8} {'实际':<8} {'圆形':<6} {'散落':<6} {'说明'}")
    print("-" * 80)

    all_files = tidy_files + messy_files
    for fname in all_files:
        path = os.path.join(img_dir, fname)
        if not os.path.exists(path):
            print(f"{fname:<12} 缺失")
            continue

        img = safe_imread(path)
        if img is None:
            print(f"{fname:<12} 读取失败")
            continue

        expected = "整齐" if fname.startswith("y") else "不整齐"
        detector.detect(img)
        actual = "整齐" if detector.is_neat() else "不整齐"

        ok = "✓" if expected == actual else "✗"
        reasons = detector.result.reason_text
        if len(reasons) > 30:
            reasons = reasons[:27] + "..."

        print(f"{fname:<12} {expected:<8} {actual:<6} {ok} 圆形={len(detector.result.centers):<3} 散落={len(detector.result.fallen_tips):<3} {reasons}")

        # 保存结果图
        result_img = detector.draw(img)
        result_path = os.path.join(out_dir, f"verify_{fname.replace('.jpg', '.png')}")
        cv2.imwrite(result_path, result_img)

    print("=" * 80)
    print(f"\n结果图保存在: {out_dir}/verify_*.png")


if __name__ == "__main__":
    main()
