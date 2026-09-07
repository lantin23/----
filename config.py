# -*- coding: utf-8 -*-
"""
config.py - 配置加载与日志初始化

从 config.json 读取运行参数（串口、自动控制、检测阈值、指令协议），
并对 stdlib logging 做统一初始化（控制台 + 日志文件）。

用法:
    from config import load_config, setup_logging, apply_detector_config
    cfg = load_config("config.json")
    logger = setup_logging(cfg)
    detector = PipetteDetector()
    applied = apply_detector_config(detector, cfg)
"""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- 串口 ----
    "serial": {
        "port": None,        # 串口名，如 "COM3"；null 表示自动检测，检测不到则模拟
        "baudrate": 9600,
        "timeout": 1.0,
    },
    # ---- 指令协议（发给下位机的 ASCII 文本） ----
    "commands": {
        "shake_on": "SHAKE_ON",
        "shake_off": "SHAKE_OFF",
        "next": "NEXT",
        "led_on": "LED_ON",
        "led_off": "LED_OFF",
    },
    # ---- ACK 回传 ----
    "ack": {
        "enabled": True,     # 是否等待下位机 ACK
        "ok": "OK",          # 期望的 ACK 内容（忽略大小写）
        "timeout": 1.5,      # 每次等 ACK 的秒数
        "retries": 2,        # 额外重发次数（总尝试 = 1 + retries）
    },
    # ---- 自动震荡控制 ----
    "auto_control": {
        "camera_index": 0,
        "shake_seconds": 60.0,    # 单轮震荡时长
        "settle_seconds": 2.0,    # 停止后等待枪头静止的时长
        "max_rounds": 3,          # 最大震荡轮数
        "capture_frames": 3,      # 每次抓帧数（取最后一帧）
        "frame_width": 1280,
        "frame_height": 720,
        "show_preview": True,     # 是否显示检测预览窗口
    },
    # ---- 检测阈值（对应 PipetteDetector 的属性名） ----
    "detector": {
        "detection_mode": "circle",        # circle / contour
        "slot_direction": "auto",          # auto / horizontal / vertical
        "min_tips": 4,
        "slot_alignment_tolerance": 12.0,
        "outlier_ratio_threshold": 0.15,
        "min_tips_per_slot": 2,
        "check_uniform_spacing": False,
        # 霍夫圆检测
        "hough_min_dist": 25.0,
        "hough_param1": 80.0,
        "hough_param2": 20.0,
        "hough_min_radius": 10,
        "hough_max_radius": 40,
        "circle_merge_ratio": 0.5,
        "intensity_diff_threshold": 15.0,
        # 散落枪头（细长形）
        "fallen_tip_min_aspect": 1.6,
        "fallen_tip_max_aspect": 4.5,
        "fallen_tip_min_area": 300.0,
        "fallen_tip_max_area": 20000.0,
        "fallen_tip_max_count": 0,        # 允许的散落枪头数，0=不容忍任何散落
        "fallen_tip_min_fill": 0.20,
        "fallen_tip_min_std": 25.0,
        "fallen_tip_max_dist_from_array": 250.0,
        # 完整散落枪头（可见长锥形）
        "full_tip_min_area": 5000.0,
        "full_tip_min_dim": 160.0,
        "full_tip_max_dim": 260.0,
        "full_tip_min_aspect": 3.0,
        "full_tip_max_aspect": 5.0,
        # 对角线散落枪头（斜跨在槽上）
        "diag_tip_min_len": 100.0,
        "diag_tip_min_angle": 25.0,
        "diag_tip_max_angle": 75.0,
        "diag_tip_cluster_dist": 50.0,
    },
    # ---- 日志 ----
    "logging": {
        "level": "INFO",
        "dir": "logs",
        "file": "app.log",
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并：override 覆盖 base 的同名键，缺项保留 base 默认值。"""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """读取配置文件；不存在或损坏时使用默认配置。"""
    cfg = deepcopy(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            cfg = _deep_merge(cfg, user_cfg)
            print(f"[配置] 已加载 {path}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[配置] 读取 {path} 失败({e})，使用默认配置")
    else:
        print(f"[配置] 未找到 {path}，使用默认配置")
    return cfg


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """初始化日志：输出到控制台 + logs/app.log。返回全局 logger。"""
    log_cfg = cfg.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = log_cfg.get("dir", "logs")
    log_file = os.path.join(log_dir, log_cfg.get("file", "app.log"))
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("pipette")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def apply_detector_config(detector, cfg: Dict[str, Any]) -> List[str]:
    """把 detector 配置应用到检测器实例。返回成功应用的键列表。

    只应用检测器上真实存在的 property（避免误建实例属性），
    单个键设置失败会跳过并打印警告，不影响其余键。
    """
    detector_cfg = cfg.get("detector", {})
    applied: List[str] = []
    for key, value in detector_cfg.items():
        if not hasattr(detector, key):
            print(f"[配置] 跳过未知检测参数: {key}")
            continue
        try:
            setattr(detector, key, value)
            applied.append(key)
        except (ValueError, TypeError) as e:
            print(f"[配置] 设置 {key}={value!r} 失败({e})，已跳过")
    return applied
