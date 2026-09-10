# -*- coding: utf-8 -*-
"""
auto_control.py - 自动震荡-检测-决策控制循环

程序控制时序:

  1. 打开 LED 照明（可选，led_enabled）
  2. 开始震荡 -> 计时(默认 60 秒) -> 停止震荡
  3. 等待枪头静止，从摄像头抓帧并检测整齐度
  4. 若整齐        -> 发送 NEXT，结束
  5. 若不整齐      -> 再次震荡（最多重复 max_rounds 轮）
  6. 连续多轮不整齐 -> 提示用户手动整理，确认后发送 NEXT
  7. 关闭 LED（可选）

"""

import time
from typing import Callable, List, Optional

import cv2
import numpy as np

from hardware_control import SerialController


class AutoControlLoop:
    """自动震荡控制循环（状态机）。"""

    def __init__(
        self,
        detector,
        serial_ctrl: SerialController,
        camera_index: int = 0,
        shake_seconds: float = 60.0,  #震荡时间
        settle_seconds: float = 5.0,  #等待枪头静止时间
        max_rounds: int = 3,          #最大震荡轮数
        capture_frames: int = 3,
        frame_width: int = 1280,
        frame_height: int = 720,
        show_preview: bool = True,
        led_enabled: bool = False,   # 是否启用 LED 照明控制
        confirm_callback: Optional[Callable[[], None]] = None,
        on_round_result: Optional[Callable[[int, bool, object], None]] = None,
    ):
        self.detector = detector
        self.serial = serial_ctrl
        self.camera_index = camera_index
        self.shake_seconds = shake_seconds
        self.settle_seconds = settle_seconds
        self.max_rounds = max_rounds
        self.capture_frames = capture_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.show_preview = show_preview
        self.led_enabled = led_enabled

        # 手动整理后的确认回调；默认用 input() 等待回车
        self.confirm_callback = confirm_callback
        # 每轮结果回调 (round_no, is_neat, result)，用于日志/界面
        self.on_round_result = on_round_result
        self._cap = None
        # 硬件通信是否异常（ACK 超时），由 run() 置位
        self.hw_error = False

    # ==================== 摄像头 ====================

    def _open_camera(self) -> bool:
        """打开摄像头并实测读帧，确认真实可用（避免 isOpened 假成功）。

        Windows 上默认 MSMF 后端在无相机时 isOpened() 可能返回 True 但读不到帧，
        优先用 DSHOW 后端，并实际 read() 一帧来验证。
        """
        dshow = getattr(cv2, "CAP_DSHOW", None)
        backends = [dshow, cv2.CAP_ANY] if dshow is not None else [cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            # 部分相机需预热，重试读取若干次，直到拿到真实画面
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._cap = cap
                    print(f"  [相机] 已启动 (设备号 {self.camera_index})")
                    return True
                time.sleep(0.2)
            cap.release()

        self._cap = None
        print("  [相机] 打开失败：未检测到可用相机")
        print("        请检查: 1) USB 相机是否连接  2) 是否被其他程序占用")
        print("                3) Windows 相机隐私设置是否允许访问")
        return False

    def _grab_frame(self) -> Optional[np.ndarray]:
        """抓取多帧并返回最后一帧（等待画面稳定）。"""
        frame = None
        for _ in range(self.capture_frames):
            ret, f = self._cap.read()
            if ret and f is not None:
                frame = f
        return frame

    def _close_camera(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ==================== 等待 ====================

    def _wait_with_countdown(self, seconds: float, label: str):
        """倒计时等待，Ctrl+C 可中断整个流程。"""
        end = time.time() + seconds
        while True:
            remaining = end - time.time()
            if remaining <= 0:
                break
            print(f"  {label} ... 剩余 {remaining:5.1f}s", end="\r", flush=True)
            time.sleep(min(0.5, remaining))
        print(f"  {label} ... 完成" + " " * 20)

    # ==================== 主流程 ====================

    def run(self) -> bool:
        """
        执行完整控制流程，返回最终是否自动判定为整齐。

        True  = 某轮检测整齐，已发送 NEXT
        False = 连续 max_rounds 轮不整齐，已提示手动整理并确认后发送 NEXT
        """
        print("\n" + "=" * 55)
        print("  自动震荡-检测控制")
        print("=" * 55)

        if not self.serial.is_open:
            print("  [错误] 串口未连接，无法执行自动控制")
            self.hw_error = True
            return False

        if not self._open_camera():
            return False

        # 打开照明（可选）
        if self.led_enabled:
            self.serial.set_led(True)

        final_neat = False
        self.hw_error = False
        try:
            for round_no in range(1, self.max_rounds + 1):
                print(f"\n--- 第 {round_no}/{self.max_rounds} 轮 ---")

                # 1. 开始震荡
                if not self.serial.shake_start():
                    print("  [错误] 震荡启动未被硬件确认（ACK 超时），终止本轮控制")
                    self.hw_error = True
                    break

                # 2. 计时震荡
                self._wait_with_countdown(self.shake_seconds, "震荡中")

                # 3. 停止震荡
                if not self.serial.shake_stop():
                    print("  [错误] 震荡停止未被硬件确认（ACK 超时），终止本轮控制")
                    self.hw_error = True
                    break

                # 4. 等待枪头静止
                self._wait_with_countdown(self.settle_seconds, "等待枪头静止")

                # 5. 抓帧检测
                frame = self._grab_frame()
                if frame is None:
                    print("  [相机] 抓帧失败，跳过本轮")
                    continue

                self.detector.detect(frame)
                is_neat = self.detector.is_neat()

                # 6. 结果输出
                tip_count = len(self.detector.get_centers())
                fallen = len(self.detector.result.fallen_tips)
                reasons = "" if is_neat else "; ".join(self.detector.get_reasons())
                print(f"  检测结果: {'✓ 整齐' if is_neat else '✗ 不整齐'}"
                      f"  (枪头:{tip_count}, 散落:{fallen})")
                if reasons:
                    print(f"  原因: {reasons}")

                if self.show_preview and frame is not None:
                    try:
                        display = self.detector.draw(frame)
                        cv2.imshow("自动检测", display)
                        cv2.waitKey(1)
                    except cv2.error:
                        pass

                if self.on_round_result:
                    self.on_round_result(round_no, is_neat, self.detector.result)

                # 7. 决策
                if is_neat:
                    final_neat = True
                    print("  → 整齐，发送 NEXT（进行下一步）")
                    if not self.serial.proceed():
                        print("  [错误] NEXT 未被硬件确认（ACK 超时）")
                        self.hw_error = True
                    break
                if round_no < self.max_rounds:
                    print(f"  → 不整齐，进入第 {round_no + 1} 轮震荡")
                else:
                    print(f"  → 连续 {self.max_rounds} 轮不整齐，需手动处理")
        finally:
            # 关闭照明（可选）+ 摄像头
            if self.led_enabled:
                self.serial.set_led(False)
            self._close_camera()
            try:
                cv2.destroyWindow("自动检测")
            except cv2.error:
                pass

        # 连续不整齐 → 手动整理 + 确认（硬件通信失败则跳过）
        if not final_neat and not self.hw_error:
            self._manual_confirm_and_proceed()

        return final_neat

    def _manual_confirm_and_proceed(self):
        """连续多轮不整齐时，提示用户手动整理，确认后发送 NEXT。"""
        print("\n" + "=" * 55)
        print("  连续多轮检测不整齐，请手动整理枪头")
        print("=" * 55)
        if self.confirm_callback:
            self.confirm_callback()
        else:
            try:
                input("  整理完成后按回车确认进行下一步... ")
            except (KeyboardInterrupt, EOFError):
                pass
        if self.serial.proceed():
            print("  → 已确认，发送 NEXT（进行下一步）")
        else:
            print("  [错误] NEXT 未被硬件确认（ACK 超时）")
            self.hw_error = True


def list_cameras(max_index: int = 8) -> List[int]:
    """枚举当前可用摄像头设备号（实测读帧验证，避免 isOpened 假成功）。"""
    dshow = getattr(cv2, "CAP_DSHOW", None)
    backends = [dshow, cv2.CAP_ANY] if dshow is not None else [cv2.CAP_ANY]
    found: List[int] = []
    # 探测过程中 OpenCV 后端会打印无害告警，临时静默日志
    prev_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    try:
        for idx in range(max_index + 1):
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    found.append(idx)
                    break
    finally:
        if prev_level is not None and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(prev_level)
    return found
