import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pipette_detector import PipetteDetector
from hardware_control import SerialController, list_ports
from auto_control import AutoControlLoop
from config import load_config, setup_logging, apply_detector_config


# ==================== 工具函数 ====================

def print_banner():
    print("\n" + "=" * 55)
    print("       枪头整齐度检测系统 v4.0 (Python)")
    print("=" * 55)
    print("功能: 检测移液枪头是否排列整齐")
    print("模式: circle(默认) - 检测枪头尾部圆形截面")
    print("      contour       - 检测枪身轮廓(旧版场景)")
    print("检测项: 数量、槽中心线对齐")
    print("=" * 55 + "\n")


def print_controls():
    print("\n操作说明:")
    print("  ESC    - 退出程序")
    print("  SPACE  - 保存当前截图")
    print("  c      - 清除统计信息")
    print("  h      - 显示/隐藏帮助")


def safe_input_int(prompt: str, default: int = 0) -> int:
    """安全读取整数输入，输入错误返回 default"""
    try:
        raw = input(prompt).strip()
        return int(raw) if raw else default
    except (ValueError, KeyboardInterrupt, EOFError):
        print(f"  输入无效，使用默认值 {default}")
        return default


def safe_input_float(prompt: str, default: float = 0.0) -> float:
    """安全读取浮点输入，输入错误返回 default"""
    try:
        raw = input(prompt).strip()
        return float(raw) if raw else default
    except (ValueError, KeyboardInterrupt, EOFError):
        print(f"  输入无效，使用默认值 {default}")
        return default


def safe_input_yes_no(prompt: str, default: str = "y") -> bool:
    """安全读取 yes/no 输入"""
    try:
        raw = input(prompt).strip().lower()
        if raw in ('y', 'yes', '是', 'Y'):
            return True
        elif raw in ('n', 'no', '否', 'N'):
            return False
        else:
            return default == "y"
    except (KeyboardInterrupt, EOFError):
        return default == "y"


def safe_destroy_window(window_name: str):
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def open_camera(index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
    """打开摄像头。优先使用 index，失败则依次尝试 0~5 中可用的设备。

    返回 (cap, used_index)；全部失败时返回 (None, -1)。
    """
    order = [index] + [i for i in range(6) if i != index]
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            return cap, idx
        cap.release()
    return None, -1


def save_screenshot(frame: np.ndarray, output_dir: str = ".") -> bool:
    """保存截图，文件名带时间戳"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")
    try:
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"  截图已保存: {filename}")
        else:
            print("  截图保存失败")
        return success
    except cv2.error as e:
        print(f"  截图保存异常: {e}")
        return False


def get_image_files(directory: str) -> List[str]:
    """获取目录下所有图片文件"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []

    for ext in extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
        image_files.extend(Path(directory).glob(f"*{ext.upper()}"))

    return sorted([str(f) for f in image_files])


# ==================== 模式处理 ====================

def process_image_mode(detector: PipetteDetector):
    """单张图片检测模式 - 支持连续检测"""
    print("\n--- 单张图片检测模式 ---")
    print("支持的格式: jpg, png, bmp")
    print("输入 'q' 或 'quit' 返回主菜单\n")

    window_name = "检测结果"

    while True:
        img_path = input("请输入图片路径: ").strip()
        img_path = img_path.strip("\"' ")

        if img_path.lower() in ('q', 'quit', 'exit', '退出'):
            print("  返回主菜单")
            safe_destroy_window(window_name)
            break

        if not img_path:
            print("  路径为空，请重新输入")
            continue

        if not os.path.exists(img_path):
            print(f"  文件不存在: {img_path}")
            if not safe_input_yes_no("是否继续检测其他图片? (y/n, 默认y): ", "y"):
                print("  返回主菜单")
                safe_destroy_window(window_name)
                break
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  图片读取失败: {img_path}")
            if not safe_input_yes_no("是否继续检测其他图片? (y/n, 默认y): ", "y"):
                print("  返回主菜单")
                safe_destroy_window(window_name)
                break
            continue

        print(f"  图片加载成功 (尺寸: {img.shape[1]}x{img.shape[0]})")

        detector.detect(img)
        result_img = detector.draw(img)

        try:
            cv2.imshow(window_name, result_img)
        except cv2.error:
            safe_destroy_window(window_name)
            cv2.imshow(window_name, result_img)

        print(f"\n  📊 检测结果: {'✓ 整齐' if detector.is_neat() else '✗ 不整齐'}")
        print(f"  原因: {detector.result.reason_text}")
        print(f"  检测到枪头数量: {len(detector.get_centers())}")

        print("\n  按任意键继续检测下一张...")
        key = cv2.waitKey(0)

        safe_destroy_window(window_name)

        if key == 27:
            print("  返回主菜单")
            break

        if not safe_input_yes_no("\n是否继续检测其他图片? (y/n, 默认y): ", "y"):
            print("  返回主菜单")
            break


def process_camera_mode(detector: PipetteDetector, cfg=None):
    """实时摄像头检测模式"""
    cfg = cfg or {}
    cam_cfg = cfg.get("camera", {})
    index = int(cam_cfg.get("index", 0))
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    fps = int(cam_cfg.get("fps", 30))

    print("\n--- 实时摄像头检测模式 ---")
    print(f"  正在打开摄像头 (索引 {index})...")

    cap, used_index = open_camera(index, width, height, fps)
    if cap is None:
        print("  摄像头打开失败! 请检查:")
        print("    1. 摄像头是否已连接")
        print("    2. 摄像头驱动是否正常")
        print("    3. 是否被其他程序占用")
        print(f"    4. config.json 中 camera.index 是否正确 (当前 {index})")
        return

    if used_index != index:
        print(f"  提示: 索引 {index} 不可用，已改用索引 {used_index}")

    print("  摄像头已启动")
    print_controls()

    frame_count = 0
    last_time = time.time()
    fps = 0.0
    show_help = False
    window_name = "实时检测"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("  读取帧失败，退出摄像头模式")
                break

            frame = cv2.flip(frame, 1)
            detector.detect(frame)
            display = detector.draw(frame)

            frame_count += 1
            current_time = time.time()
            time_diff = current_time - last_time
            if time_diff >= 1.0:
                fps = frame_count / time_diff
                last_time = current_time
                frame_count = 0

            if fps > 0:
                fps_text = f"FPS: {int(fps)}"
                cv2.putText(display, fps_text, (display.shape[1] - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if show_help:
                _draw_help_panel(display)

            try:
                cv2.imshow(window_name, display)
            except cv2.error:
                safe_destroy_window(window_name)
                cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\n  退出摄像头模式...")
                break
            elif key == 32:
                save_screenshot(display)
            elif key == ord('c') or key == ord('C'):
                print("  清除统计信息")
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help

    except KeyboardInterrupt:
        print("\n  用户中断，退出摄像头模式")
    finally:
        cap.release()
        safe_destroy_window(window_name)
        print("  摄像头已关闭")


def process_batch_mode(detector: PipetteDetector):
    """批量图片检测模式"""
    print("\n" + "=" * 50)
    print("批量图片检测模式")
    print("=" * 50)

    window_name = "批量检测进度"

    while True:
        dir_path = input("\n请输入图片目录路径 (输入 q 返回主菜单): ").strip()
        dir_path = dir_path.strip("\"' ")

        if dir_path.lower() in ('q', 'quit', 'exit', '退出'):
            print("  返回主菜单")
            safe_destroy_window(window_name)
            break

        if not os.path.isdir(dir_path):
            print(f"  目录不存在: {dir_path}")
            if not safe_input_yes_no("是否重新输入目录? (y/n, 默认y): ", "y"):
                print("  返回主菜单")
                safe_destroy_window(window_name)
                break
            continue

        image_paths = get_image_files(dir_path)
        if not image_paths:
            print(f"  目录中没有找到图片文件: {dir_path}")
            if not safe_input_yes_no("是否重新输入目录? (y/n, 默认y): ", "y"):
                print("  返回主菜单")
                safe_destroy_window(window_name)
                break
            continue

        print(f"  找到 {len(image_paths)} 张图片")

        save_choice = input("是否保存检测结果图像? (y/n, 默认y): ").strip().lower()
        save_images = save_choice != 'n'

        output_dir = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
            print(f"  结果将保存到: {output_dir}")

        print("\n" + "=" * 50)
        print("开始批量检测... (按 ESC 中断)")
        print("=" * 50)

        results = []
        success_count = 0
        fail_count = 0
        total = len(image_paths)

        for idx, img_path in enumerate(image_paths, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{idx}/{total}] ✗ {os.path.basename(img_path)} - 读取失败")
                results.append({"file": img_path, "success": False, "error": "读取失败"})
                continue

            start_time = time.time()
            detector.detect(img)
            elapsed = time.time() - start_time

            is_neat = detector.is_neat()
            tip_count = len(detector.get_centers())
            reasons = detector.get_reasons()

            status = "✓" if is_neat else "✗"
            reason_text = "整齐" if is_neat else ("; ".join(reasons) if reasons else "不整齐")
            print(f"[{idx}/{total}] {status} {os.path.basename(img_path)} - {reason_text} (枪头:{tip_count}, {elapsed*1000:.1f}ms)")

            if is_neat:
                success_count += 1
            else:
                fail_count += 1

            result_img = None
            if save_images:
                result_img = detector.draw(img)
                save_path = os.path.join(output_dir, f"result_{idx:04d}_{os.path.basename(img_path)}")
                cv2.imwrite(save_path, result_img)

            results.append({
                "file": os.path.basename(img_path),
                "is_neat": is_neat,
                "tip_count": tip_count,
                "reasons": reasons,
                "elapsed_ms": elapsed * 1000
            })

            if idx % 10 == 0 and save_images and result_img is not None:
                try:
                    cv2.imshow(window_name, result_img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("\n  用户中断批量检测")
                        break
                except cv2.error:
                    safe_destroy_window(window_name)
                    cv2.imshow(window_name, result_img)

        safe_destroy_window(window_name)

        print("\n" + "=" * 50)
        print("检测完成!")
        print("=" * 50)
        print(f"  总图片数: {total}")
        print(f"  整齐: {success_count}")
        print(f"  不整齐: {fail_count}")
        print(f"  整齐率: {success_count/total*100:.1f}%" if total > 0 else "  整齐率: 0%")

        if save_images:
            report_path = os.path.join(output_dir, "report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("批量检测报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总图片数: {total}\n")
                f.write(f"整齐: {success_count}\n")
                f.write(f"不整齐: {fail_count}\n")
                f.write(f"整齐率: {success_count/total*100:.1f}%\n" if total > 0 else "整齐率: 0%\n")
                f.write("\n详细结果:\n")
                f.write("-" * 50 + "\n")
                for r in results:
                    f.write(f"{r['file']}: {'整齐' if r['is_neat'] else '不整齐'} (枪头:{r['tip_count']})\n")

            print(f"\n  报告已保存: {report_path}")
            print(f"  结果图像保存在: {os.path.abspath(output_dir)}")

        if not safe_input_yes_no("\n是否继续检测其他目录? (y/n, 默认y): ", "y"):
            print("  返回主菜单")
            break


def process_auto_mode(detector: PipetteDetector, cfg=None):
    """自动震荡控制模式 - 串口控制硬件 + 摄像头实时检测"""
    print("\n--- 自动震荡控制模式 (串口+硬件) ---")
    print("流程: 震荡60s → 停止 → 检测 → 整齐则NEXT，否则再震荡(最多3轮)")
    print("      连续3轮不整齐 → 提示手动整理 → 确认后NEXT\n")

    cfg = cfg or {}
    serial_cfg = cfg.get("serial", {})
    ack_cfg = cfg.get("ack", {})
    cmd_cfg = cfg.get("commands", {})
    auto_cfg = cfg.get("auto_control", {})

    # 串口配置
    ports = list_ports()
    simulate = False
    port = serial_cfg.get("port")
    if ports:
        print(f"  检测到串口: {', '.join(ports)}")
        default_port = port or ports[0]
        port = input(f"  请输入串口名 (直接回车用 {default_port}): ").strip()
        if not port:
            port = default_port
    else:
        print("  未检测到串口 → 使用模拟模式（指令只打印，不发硬件）")
        simulate = True

    baud = safe_input_int(
        "  波特率 (直接回车用 %d): " % serial_cfg.get("baudrate", 9600),
        serial_cfg.get("baudrate", 9600))
    shake_secs = safe_input_float(
        "  单轮震荡时长/秒 (直接回车用 %.0f): " % auto_cfg.get("shake_seconds", 60.0),
        auto_cfg.get("shake_seconds", 60.0))
    rounds = safe_input_int(
        "  最大轮数 (直接回车用 %d): " % auto_cfg.get("max_rounds", 3),
        auto_cfg.get("max_rounds", 3))

    ctrl = SerialController(
        port=port,
        baudrate=baud,
        timeout=serial_cfg.get("timeout", 1.0),
        commands=cmd_cfg or None,
        simulate=simulate,
        ack_enabled=ack_cfg.get("enabled", True),
        ack_ok=ack_cfg.get("ok", "OK"),
        ack_timeout=ack_cfg.get("timeout", 1.5),
        ack_retries=ack_cfg.get("retries", 2),
    )
    ctrl.open()

    loop = AutoControlLoop(
        detector, ctrl,
        camera_index=auto_cfg.get("camera_index", 0),
        shake_seconds=shake_secs,
        settle_seconds=auto_cfg.get("settle_seconds", 2.0),
        max_rounds=rounds,
        capture_frames=auto_cfg.get("capture_frames", 3),
        frame_width=auto_cfg.get("frame_width", 1280),
        frame_height=auto_cfg.get("frame_height", 720),
        show_preview=auto_cfg.get("show_preview", True),
    )
    neat = loop.run()

    ctrl.close()
    print("\n流程结束。")
    if loop.hw_error:
        print("  最终结果: 硬件通信异常（ACK 超时），流程已终止，请检查串口/下位机")
    else:
        print("  最终结果:", "整齐（自动判定）" if neat else "不整齐（已手动确认后继续）")


def _draw_help_panel(frame: np.ndarray):
    """在帧上绘制帮助面板"""
    cv2.rectangle(frame, (10, 100), (300, 260), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 100), (300, 260), (255, 255, 255), 2)

    lines = [
        ("Controls:", 0.6, (255, 255, 255)),
        ("ESC - Exit", 0.5, (200, 200, 200)),
        ("SPACE - Screenshot", 0.5, (200, 200, 200)),
        ("c - Clear info", 0.5, (200, 200, 200)),
        ("h - Hide help", 0.5, (200, 200, 200)),
    ]
    y = 130
    for text, scale, color in lines:
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y += 28


def configure_parameters(detector: PipetteDetector):
    """参数配置"""
    print("\n--- 参数配置 ---")
    print("当前参数:")
    print(f"  检测模式: {detector.detection_mode}")
    print(f"  槽方向: {detector.slot_direction}")
    print(f"  最少枪头: {detector.min_tips}")
    print(f"  槽对齐容差: {detector.slot_alignment_tolerance} 像素")
    print(f"  异常比例阈值: {detector.outlier_ratio_threshold * 100:.0f}%")
    print(f"  每槽最少枪头: {detector.min_tips_per_slot}")
    print(f"  严格间距检查: {'开启' if detector.check_uniform_spacing else '关闭'}")

    if detector.detection_mode == "circle":
        print("\n  [圆形检测参数]")
        print(f"    霍夫 minDist: {detector.hough_min_dist}")
        print(f"    霍夫 param1: {detector.hough_param1}")
        print(f"    霍夫 param2: {detector.hough_param2}")
        print(f"    半径范围: [{detector.hough_min_radius}, {detector.hough_max_radius}]")
        print(f"    合并比例: {detector.circle_merge_ratio}")
        print(f"    强度差阈值: {detector.intensity_diff_threshold}")
    else:
        print("\n  [轮廓检测参数]")
        print(f"    最小面积: {detector.min_area}")
        print(f"    最大面积: {detector.max_area}")
        print(f"    角度阈值: {detector.angle_threshold}°")

    choice = input("\n是否需要修改参数? (y/n): ").strip().lower()
    if choice != 'y':
        return

    # 检测模式
    mode = input("  检测模式 (circle/contour, 默认不变): ").strip().lower()
    if mode in ('circle', 'contour'):
        detector.detection_mode = mode
        print(f"    检测模式已设为: {mode}")

    # 槽方向
    direction = input("  槽方向 (auto/horizontal/vertical, 默认不变): ").strip().lower()
    if direction in ('auto', 'horizontal', 'vertical'):
        detector.slot_direction = direction
        print(f"    槽方向已设为: {direction}")

    _try_set_param(detector, "min_tips", "最少枪头数量",
                   lambda v: setattr(detector, "min_tips", v),
                   is_float=False)

    _try_set_param(detector, "slot_alignment_tolerance", "槽对齐容差(像素)",
                   lambda v: setattr(detector, "slot_alignment_tolerance", v),
                   is_float=True)

    _try_set_param(detector, "outlier_ratio_threshold", "异常比例阈值(0-1)",
                   lambda v: setattr(detector, "outlier_ratio_threshold", v),
                   is_float=True)

    _try_set_param(detector, "min_tips_per_slot", "每槽最少枪头数(>=1)",
                   lambda v: setattr(detector, "min_tips_per_slot", v),
                   is_float=False)

    enable_spacing = safe_input_yes_no(
        f"  是否启用严格相邻间距均匀性检查? (当前 {'开启' if detector.check_uniform_spacing else '关闭'} y/n, 默认n): ",
        "n"
    )
    detector.check_uniform_spacing = enable_spacing

    if enable_spacing:
        _try_set_param(detector, "space_ratio", "间距偏差比例(0-1)",
                       lambda v: setattr(detector, "space_ratio", v),
                       is_float=True)

    # 模式专属参数
    if detector.detection_mode == "circle":
        print("\n  [圆形检测参数]")
        _try_set_param(detector, "hough_min_dist", "霍夫 minDist(圆心最小距离)",
                       lambda v: setattr(detector, "hough_min_dist", v),
                       is_float=True)
        _try_set_param(detector, "hough_param1", "霍夫 param1(Canny高阈值)",
                       lambda v: setattr(detector, "hough_param1", v),
                       is_float=True)
        _try_set_param(detector, "hough_param2", "霍夫 param2(累加器阈值，越小越多)",
                       lambda v: setattr(detector, "hough_param2", v),
                       is_float=True)
        _try_set_param(detector, "hough_min_radius", "最小半径",
                       lambda v: setattr(detector, "hough_min_radius", v),
                       is_float=False)
        _try_set_param(detector, "hough_max_radius", "最大半径",
                       lambda v: setattr(detector, "hough_max_radius", v),
                       is_float=False)
        _try_set_param(detector, "circle_merge_ratio", "合并比例(0-1]",
                       lambda v: setattr(detector, "circle_merge_ratio", v),
                       is_float=True)
        _try_set_param(detector, "intensity_diff_threshold", "强度差阈值(>=0, 0关闭验证)",
                       lambda v: setattr(detector, "intensity_diff_threshold", v),
                       is_float=True)
    else:
        print("\n  [轮廓检测参数]")
        _try_set_param(detector, "min_area", "最小面积",
                       lambda v: setattr(detector, "min_area", v),
                       is_float=True)
        _try_set_param(detector, "max_area", "最大面积",
                       lambda v: setattr(detector, "max_area", v),
                       is_float=True)
        _try_set_param(detector, "angle_threshold", "角度阈值(度)",
                       lambda v: setattr(detector, "angle_threshold", v),
                       is_float=True)

    print("  参数已更新")


def _try_set_param(detector: PipetteDetector, attr: str, label: str,
                   setter, is_float: bool = True):
    """安全设置单个参数，捕获 ValueError"""
    current = getattr(detector, attr)
    prompt = f"  请输入{label} (当前 {current}): "

    if is_float:
        val = safe_input_float(prompt, current)
    else:
        val = safe_input_int(prompt, int(current))

    if val <= 0:
        print("    值必须大于 0，跳过")
        return

    try:
        setter(val)
    except ValueError as e:
        print(f"    设置失败: {e}")


# ==================== 主函数 ====================

def main():
    cfg = load_config("config.json")
    logger = setup_logging(cfg)

    print_banner()

    detector = PipetteDetector()
    applied = apply_detector_config(detector, cfg)
    if applied:
        print(f"  已从配置应用 {len(applied)} 项检测参数")
        logger.info("检测参数已应用 %d 项: %s", len(applied), ", ".join(applied))

    choice = input("是否调整检测参数? (y/n): ").strip().lower()
    if choice == 'y':
        configure_parameters(detector)

    while True:
        print("\n请选择模式:")
        print("  1. 单张图片检测 (可连续检测)")
        print("  2. 实时摄像头检测")
        print("  3. 批量图片检测")
        print("  4. 参数配置")
        print("  5. 自动震荡控制 (串口+硬件)")
        print("  6. 退出程序")
        print("请输入选择(1/2/3/4/5/6): ", end="")

        mode = safe_input_int("", 0)

        if mode == 1:
            process_image_mode(detector)
        elif mode == 2:
            process_camera_mode(detector, cfg)
        elif mode == 3:
            process_batch_mode(detector)
        elif mode == 4:
            configure_parameters(detector)
        elif mode == 5:
            process_auto_mode(detector, cfg)
        elif mode == 6:
            print("\n感谢使用，再见!")
            break
        else:
            print("  输入错误，请重新选择 (1/2/3/4/5/6)")

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n程序被用户中断，再见!")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"\n程序发生未预期异常: {e}")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        sys.exit(1)
