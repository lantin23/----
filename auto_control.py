# -*- coding: utf-8 -*-
"""
auto_control.py - 自动震荡-检测-决策控制循环

  1. 打开 LED 照明（可选）
  2. 开始震荡 -> 计时(默认 60 秒) -> 停止震荡
  3. 等待枪头静止，从摄像头抓帧并检测整齐度
  4. 若整齐        -> 发送 NEXT，结束
  5. 若不整齐      -> 再次震荡（最多重复 max_rounds 轮）
  6. 连续多轮不整齐 -> 提示用户手动整理，确认后发送 NEXT
  7. 关闭 LED*

通过依赖注入 camera_index / confirm_callback，进行自测试
"""

import time
from typing import Callable, Optional

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
        shake_seconds: float = 60.0,
        settle_seconds: float = 2.0,
        max_rounds: int = 3,
        capture_frames: int = 3,
        frame_width: int = 1280,
        frame_height: int = 720,
        show_preview: bool = True,
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
        # 手动整理后的确认回调；默认用 input() 等待回车
        self.confirm_callback = confirm_callback
        # 每轮结果回调 (round_no, is_neat, result)，用于日志/界面
        self.on_round_result = on_round_result
        self._cap = None
        # 硬件通信是否异常（ACK 超时），由 run() 置位
        self.hw_error = False

    # ==================== 摄像头 ====================

    def _open_camera(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print("  [相机] 打开失败，请检查 USB 摄像头连接")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        print("  [相机] 已启动")
        return True

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

        if not self._open_camera():
            return False

        # 打开照明
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
            # 关闭照明 + 摄像头
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
