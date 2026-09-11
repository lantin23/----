"""
枪头整齐度检测系统 v5.0 (Python 版)
自动震荡控制: 串口控制硬件 + 摄像头实时检测
"""

import sys
import time

import cv2

from pipette_detector import PipetteDetector
from hardware_control import SerialController, list_ports
from auto_control import AutoControlLoop, list_cameras, resolve_backend
from config import load_config, setup_logging, apply_detector_config, save_config


# ==================== 工具函数 ====================

def print_banner():
    print("\n" + "=" * 55)
    print("       枪头整齐度检测系统 v5.0 (Python)")
    print("=" * 55)
    print("功能: 震荡后实时检测枪头是否排列整齐，并通过串口控制硬件")
    print("模式: circle(默认) - 检测枪头尾部圆形截面")
    print("      contour       - 检测枪身轮廓")
    print("检测项: 数量、槽中心线对齐、散落枪头")
    print("=" * 55 + "\n")


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


# ==================== 自动控制模式 ====================

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
    port = serial_cfg.get("port")
    if ports:
        print(f"  检测到串口: {', '.join(ports)}")
        default_port = port or ports[0]
        port = input(f"  请输入串口名 (直接回车用 {default_port}): ").strip()
        if not port:
            port = default_port
    else:
        print("  [错误] 未检测到可用串口，无法进行硬件控制")
        return

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
        ack_enabled=ack_cfg.get("enabled", True),
        ack_ok=ack_cfg.get("ok", "OK"),
        ack_timeout=ack_cfg.get("timeout", 1.5),
        ack_retries=ack_cfg.get("retries", 2),
    )
    if not ctrl.open():
        print("  [错误] 串口打开失败，无法执行自动控制")
        return

    loop = AutoControlLoop(
        detector, ctrl,
        camera_index=auto_cfg.get("camera_index", 0),
        camera_backend=auto_cfg.get("camera_backend", "msmf"),
        shake_seconds=shake_secs,
        settle_seconds=auto_cfg.get("settle_seconds", 5.0),
        max_rounds=rounds,
        capture_frames=auto_cfg.get("capture_frames", 3),
        frame_width=auto_cfg.get("frame_width", 1280),
        frame_height=auto_cfg.get("frame_height", 720),
        show_preview=auto_cfg.get("show_preview", True),
        led_enabled=auto_cfg.get("led_enabled", False),
    )
    neat = loop.run()

    ctrl.close()
    print("\n流程结束。")
    if loop.hw_error:
        print("  最终结果: 硬件通信异常（ACK 超时），流程已终止，请检查串口/下位机")
    else:
        print("  最终结果:", "整齐（自动判定）" if neat else "不整齐（已手动确认后继续）")


# ==================== 相机预览 / 切换 ====================

def process_camera_preview(detector: PipetteDetector, cfg=None):
    """相机预览 / 切换：列出可用相机，预览并可选设为默认设备。"""
    print("\n--- 相机预览 / 切换 ---")
    cfg = cfg or {}
    backend_name = cfg.get("auto_control", {}).get("camera_backend", "msmf")
    cams = list_cameras(backend_name=backend_name)
    if not cams:
        print("  [相机] 未检测到可用相机")
        return

    print("  检测到以下摄像头设备号:")
    for idx in cams:
        print(f"    [{idx}]  设备号 {idx}")

    default_idx = cfg.get("auto_control", {}).get("camera_index", 0)
    idx = safe_input_int(
        f"  请输入要预览的设备号 (直接回车用 {default_idx}): ", default_idx)
    if idx not in cams:
        print(f"  设备号 {idx} 不可用，可用设备: {cams}")
        return

    cap = cv2.VideoCapture(idx, resolve_backend(backend_name))
    if not cap.isOpened():
        print("  [相机] 打开失败")
        return
    print(f"  正在预览设备号 {idx}，按 ESC 退出...")
    try:
        # MSMF 部分相机需预热/偶发读帧失败，允许连续失败若干次才判定失败
        fail_count = 0
        while True:
            ret, frame = cap.read()
            if ret and frame is not None:
                fail_count = 0
                detector.detect(frame)
                display = detector.draw(frame)
                cv2.imshow("相机预览", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                fail_count += 1
                time.sleep(0.2)
                if fail_count >= 5:
                    print("  [相机] 连续读取画面失败")
                    break
    finally:
        cap.release()
        try:
            cv2.destroyWindow("相机预览")
        except cv2.error:
            pass

    if safe_input_yes_no(f"  是否将设备号 {idx} 设为默认相机? (y/n, 默认y): ", "y"):
        cfg.setdefault("auto_control", {})["camera_index"] = idx
        if save_config(cfg):
            print(f"  已保存 camera_index={idx} 到 config.json，下次自动控制将使用该相机")


# ==================== 参数配置 ====================

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
        print("\n请选择功能:")
        print("  1. 参数配置")
        print("  2. 自动震荡控制 (串口+硬件)")
        print("  3. 相机预览 / 切换")
        print("  4. 退出程序")
        print("请输入选择(1/2/3/4): ", end="")

        mode = safe_input_int("", 0)

        if mode == 1:
            configure_parameters(detector)
        elif mode == 2:
            process_auto_mode(detector, cfg)
        elif mode == 3:
            process_camera_preview(detector, cfg)
        elif mode == 4:
            print("\n感谢使用，再见!")
            break
        else:
            print("  输入错误，请重新选择 (1/2/3/4)")

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
