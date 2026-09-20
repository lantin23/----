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
    SHAKE_ON 被接受后返回 OK，100 圈到位后另行返回 STOPPED。
    NEXT 被接受后返回 OK，电机2往返到位、电机3往返脉冲发完后返回 NEXT_DONE。
    SHAKE_ON / NEXT 不自动重发，避免 ACK 丢失导致重复运动。

用法:
    ctrl = SerialController(port="COM3", baudrate=115200)
    if ctrl.open():
        if ctrl.shake_start() and ctrl.wait_stopped():
            # 此时才可开始相机检测
            pass
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
        baudrate: int = 115200,
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
        self._shake_pending = False

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
        deadline = time.monotonic() + self.ack_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            line = self.read_line(timeout=remaining)
            if line and line.startswith("DBG "):
                print(f"  [诊断] {line}")
                continue
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
        """启动一次100圈动作，必须收到 OK；不自动重发运动指令。"""
        self._shake_pending = False
        self._shake_pending = self._start_motion("shake_on")
        return self._shake_pending

    def _start_motion(self, key: str) -> bool:
        """只发送一次运动命令并等待接受确认，失败时不盲目重发。"""
        if not self.is_open:
            return False
        command = self.commands[key]
        try:
            # 仅在新动作开始前清理旧响应，OK 与完成通知之间不能清空。
            self._ser.reset_input_buffer()
            payload = (command + "\n").encode("utf-8")
            if self._ser.write(payload) != len(payload):
                print(f"  [串口] {command} 未完整发送，终止流程")
                return False
            self._ser.flush()
        except OSError as exc:
            print(f"  [串口] {command} 发送失败: {exc}")
            return False
        print(f"  [串口→] {command}")
        ack = self._wait_ack()
        if ack is not None and ack.upper() == self.ack_ok:
            print(f"  [串口←] ACK {ack}")
            return True
        print(f"  [串口] 启动未确认（收到 {ack!r}），不自动重发；请检查电机状态")
        return False

    def wait_stopped(self, timeout: float = 65.0) -> bool:
        """等待本轮 STOPPED；错误或超时均不允许开始检测。"""
        if not self.is_open or not getattr(self, "_shake_pending", False):
            return False
        self._shake_pending = False
        return self._wait_completion("STOPPED", timeout)

    def _wait_completion(self, expected: str, timeout: float) -> bool:
        """接受确认不是完成确认；只认对应的完成信号。"""
        deadline = time.monotonic() + timeout
        print(f"  等待电机完成通知 {expected}（最多 {timeout:g} 秒）...")
        while self.is_open:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            line = self.read_line(timeout=remaining)
            if not line:
                continue
            print(f"  [串口←] {line}")
            status = line.strip().upper()
            if status == expected:
                return True
            if status.startswith("ERROR") or status == "BUSY":
                return False
        print(f"  [串口] 未收到 {expected}，不能确认动作已完成，终止流程")
        return False

    def shake_stop(self) -> bool:
        """停止震荡。"""
        return self._send_cmd("shake_off")

    def proceed(self, timeout: float = 40.0) -> bool:
        """NEXT只发一次；收到OK后等待电机2、3动作链完成NEXT_DONE。"""
        if not self._start_motion("next"):
            return False
        # 固件每段8秒超时，共四段，另留处理和串口传输余量。
        return self._wait_completion("NEXT_DONE", timeout)

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
