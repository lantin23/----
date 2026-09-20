# -*- coding: utf-8 -*-
"""
数据分析脚本：对 exam 目录下所有测试图片运行检测，
收集每张图的期望/实际判定、圆形数、散落枪头数、失败原因，
汇总为 CSV 并输出整体统计（准确率、混淆矩阵、逐类指标）。
"""
import os
import sys
import csv
import cv2
import numpy as np

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/符号打印报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipette_detector import PipetteDetector
from config import load_config, apply_detector_config

IMG_DIR = r"C:\Users\23870\Pictures\exam"
OUT_DIR = r"C:\Users\23870\Pictures\result"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_data.csv")


def safe_imread(path):
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    detector = PipetteDetector()
    cfg = load_config("config.json")
    apply_detector_config(detector, cfg)

    # 收集 exam 目录下所有图片，按文件名自然排序
    files = sorted(
        [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda s: (s[0], int("".join(ch for ch in s[1:].split(".")[0] if ch.isdigit()) or 0))
    )

    rows = []
    for fname in files:
        path = os.path.join(IMG_DIR, fname)
        img = safe_imread(path)
        if img is None:
            continue
        expected = "整齐" if fname.lower().startswith("y") else "不整齐"
        detector.detect(img)
        actual = "整齐" if detector.is_neat() else "不整齐"
        rows.append({
            "file": fname,
            "expected": expected,
            "actual": actual,
            "correct": expected == actual,
            "circles": len(detector.result.centers),
            "fallen": len(detector.result.fallen_tips),
            "reason": detector.result.reason_text,
        })

    # 写 CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "expected", "actual", "correct", "circles", "fallen", "reason"])
        w.writeheader()
        w.writerows(rows)

    # 统计
    n_total = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    # 混淆矩阵: 行=期望, 列=实际 ; TP=整齐判整齐, TN=不整齐判不整齐
    tp = sum(1 for r in rows if r["expected"] == "整齐" and r["actual"] == "整齐")
    fp = sum(1 for r in rows if r["expected"] == "不整齐" and r["actual"] == "整齐")
    fn = sum(1 for r in rows if r["expected"] == "整齐" and r["actual"] == "不整齐")
    tn = sum(1 for r in rows if r["expected"] == "不整齐" and r["actual"] == "不整齐")

    n_tidy = tp + fn
    n_messy = tn + fp

    acc = n_correct / n_total if n_total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    tidy_acc = tp / n_tidy if n_tidy else 0   # 整齐类召回
    messy_acc = tn / n_messy if n_messy else 0  # 不整齐类召回

    print("=" * 90)
    print("检测结果汇总")
    print("=" * 90)
    print(f"{'文件':<10} {'期望':<7} {'实际':<7} {'圆形':<5} {'散落':<5} {'说明'}")
    print("-" * 90)
    for r in rows:
        mark = "OK" if r["correct"] else "XX"
        print(f"{r['file']:<10} {r['expected']:<7} {r['actual']:<6} {r['circles']:<5} {r['fallen']:<5} {mark} {r['reason']}")

    print("=" * 90)
    print(f"样本总数: {n_total}   (整齐 {n_tidy} / 不整齐 {n_messy})")
    print(f"判定正确: {n_correct}   判定错误: {n_total - n_correct}")
    print(f"总体准确率: {acc*100:.1f}%")
    print(f"精确率(整齐): {precision*100:.1f}%   召回率(整齐): {recall*100:.1f}%   F1: {f1:.3f}")
    print(f"整齐类准确率(召回): {tidy_acc*100:.1f}%   不整齐类准确率: {messy_acc*100:.1f}%")
    print()
    print("混淆矩阵 (行=期望, 列=实际):")
    print(f"                实际整齐  实际不整齐")
    print(f"  期望整齐       {tp:<9} {fn}")
    print(f"  期望不整齐     {fp:<9} {tn}")
    print()
    print(f"数据已保存: {CSV_PATH}")

    # 错误样本
    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print(f"\n误判样本 ({len(wrong)} 个):")
        for r in wrong:
            print(f"  {r['file']}: 期望 {r['expected']}, 实际 {r['actual']}, 圆形={r['circles']}, 散落={r['fallen']}, 原因={r['reason']}")
    else:
        print("\n无误判样本")


if __name__ == "__main__":
    main()
