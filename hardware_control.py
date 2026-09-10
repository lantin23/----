# -*- coding: utf-8 -*-
"""
hardware_control.py - 串口硬件控制模块

通过串口（pyserial）控制下位机（Arduino / STM32 等）执行震荡、LED 照明、
以及"进行下一步"动作。指令采用可配置的 ASCII 协议，每条指令以换行符结尾。

默认协议（可自定义）:
    SHAKE_ON   开始震荡
    SHAKE_OFF  停止震荡
    NEXT       进行下一步动作
    LED_ON     打开 LED 照明
    LED_OFF    关闭 LED 照明

ACK 回传（抗干扰）:
    下位机收到并执行完指令后，应回传一行 "OK\\n" 表示确认。
    软件发送后会等待 ACK；超时未收到则自动重发，重试仍失败则返回 False。

用法:
    ctrl = SerialController(port="COM3", baudrate=9600)
    if ctrl.open():
        ctrl.shake_start()      # 发送 SHAKE_ON 并等待 OK
        time.sleep(60)
        ctrl.shake_stop()
        ctrl.set_led(True)
        ctrl.proceed()
        ctrl.close()
"""

import time
from typing import Dict, List, Optional

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False


class SerialController:
    """串口控制器：负责与下位机通信，发送震荡 / LED / 下一步指令。"""

    DEFAULT_COMMANDS: Dict[str, str] = {
        "shake_on": "SHAKE_ON",
        "shake_off": "SHAKE_OFF",
        "next": "NEXT",
        "led_on": "LED_ON",
        "led_off": "LED_OFF",
    }

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 9600,
        timeout: float = 1.0,
        commands: Optional[Dict[str, str]] = None,
        # ---- ACK 回传配置 ----
        ack_enabled: bool = True,   # 是否等待下位机 ACK
        ack_ok: str = "OK",         # 期望的 ACK 内容（忽略大小写）
        ack_timeout: float = 1.5,   # 每次等待 ACK 的秒数
        ack_retries: int = 2,       # 额外重发次数（总尝试 = 1 + retries）
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.commands = dict(self.DEFAULT_COMMANDS)
        if commands:
            self.commands.update(commands)

        self.ack_enabled = ack_enabled
        self.ack_ok = ack_ok.strip().upper()
        self.ack_timeout = ack_timeout
        self.ack_retries = ack_retries

        self._ser = None

    # ==================== 连接管理 ====================

    def open(self) -> bool:
        """打开串口。成功返回 True，失败返回 False（不降级、不模拟）。"""
        if not _SERIAL_AVAILABLE:
            print("  [串口] 未安装 pyserial，无法连接硬件")
            return False
        if not self.port:
            print("  [串口] 未指定串口名，无法连接硬件")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"  [串口] 已连接 {self.port} @ {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"  [串口] 打开失败({e})，请检查端口号 / 占用情况")
            return False

    def close(self):
        """关闭串口。"""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ==================== 底层发送 ====================

    def _write_line(self, text: str):
        """发送一行指令（自动追加换行）。"""
        if self._ser is None or not self._ser.is_open:
            print(f"  [串口] 未连接，跳过发送: {text}")
            return
        try:
            payload = (text + "\n").encode("utf-8")
            self._ser.write(payload)
            self._ser.flush()
            print(f"  [串口→] {text}")
        except serial.SerialException as e:
            print(f"  [串口] 发送失败({e}): {text}")

    def send_raw(self, text: str):
        """发送自定义指令（不自动追加换行、不等 ACK）。"""
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.write(text.encode("utf-8"))
                self._ser.flush()
                print(f"  [串口→] {text!r}")
            except serial.SerialException as e:
                print(f"  [串口] 发送失败({e}): {text!r}")

    # ==================== 回传（ACK） ====================

    def read_line(self, timeout: Optional[float] = None) -> Optional[str]:
        """读取一行下位机回传。可指定本次读取的超时秒数。"""
        if self._ser is None or not self._ser.is_open:
            return None
        old_timeout = self._ser.timeout
        try:
            if timeout is not None:
                self._ser.timeout = timeout
            raw = self._ser.readline()
            if not raw:
                return None
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
        finally:
            self._ser.timeout = old_timeout

    def _wait_ack(self) -> Optional[str]:
        """等待下位机回传一行，最多 ack_timeout 秒。"""
        deadline = time.time() + self.ack_timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            line = self.read_line(timeout=remaining)
            if line:
                return line

    def _write_line_with_ack(self, text: str) -> bool:
        """发送一行指令并等待 ACK，超时自动重发。返回是否被确认。"""
        if not self.is_open:
            print(f"  [串口] 未连接，无法发送 {text!r}")
            return False
        attempts = self.ack_retries + 1
        for attempt in range(1, attempts + 1):
            self._write_line(text)
            # 关闭 ACK 时，视为直接成功
            if not self.ack_enabled:
                return True

            ack = self._wait_ack()
            if ack is not None and ack.upper() == self.ack_ok:
                print(f"  [串口←] ACK {ack}")
                return True

            if attempt < attempts:
                print(f"  [串口] 未收到 ACK（期望 {self.ack_ok!r}，收到 {ack!r}），"
                      f"第 {attempt}/{self.ack_retries} 次重发...")
                time.sleep(0.3)
            else:
                print(f"  [串口] 发送 {text!r} 连续 {attempts} 次均未收到 ACK")
        return False

    def _send_cmd(self, key: str) -> bool:
        cmd = self.commands.get(key)
        if cmd is None:
            raise KeyError(f"未定义的串口指令键: {key}")
        return self._write_line_with_ack(cmd)

    # ==================== 高层动作（返回是否被 ACK 确认） ====================

    def shake_start(self) -> bool:
        """开始震荡。"""
        return self._send_cmd("shake_on")

    def shake_stop(self) -> bool:
        """停止震荡。"""
        return self._send_cmd("shake_off")

    def proceed(self) -> bool:
        """进行下一步动作。"""
        return self._send_cmd("next")

    def set_led(self, on: bool) -> bool:
        """打开 / 关闭 LED 照明。"""
        return self._send_cmd("led_on" if on else "led_off")


def list_ports() -> List[str]:
    """列出当前系统可用串口（如 COM3）。"""
    if not _SERIAL_AVAILABLE:
        return []
    try:
        return [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        return []
