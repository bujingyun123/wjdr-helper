"""
Device：封装 Scrcpy 连接与 ADB 点击控制。
- 后台线程持续接收视频帧并写入 SharedContext.current_frame
- 提供 click/swipe 方法供 Task 调用
"""
from __future__ import annotations

import threading
import time
import subprocess
from typing import Optional, Callable
import logging

from core.shared_state import SharedContext

logger = logging.getLogger(__name__)

try:
    import av as _av
    if not hasattr(_av, 'AVError') and hasattr(_av, 'FFmpegError'):
        _av.AVError = _av.FFmpegError
except ImportError:
    pass

try:
    import scrcpy
    _SCRCPY_AVAILABLE = True
except ImportError:
    _SCRCPY_AVAILABLE = False

try:
    import adbutils
    _ADB_AVAILABLE = True
except ImportError:
    _ADB_AVAILABLE = False


class Device:
    """
    管理唯一的 Scrcpy 连接，将帧写入 SharedContext，
    并提供 touch 控制接口。
    """

    def __init__(self, ctx: SharedContext, config: dict) -> None:
        self._ctx = ctx
        self._config = config
        self._client: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self.is_connected: bool = False
        self._control_lock = threading.Lock()

        # 连接状态变更回调（供 UI 监听）
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None

    # ── 连接管理 ──────────────────────────────────────────

    def connect(self) -> None:
        """非阻塞方式启动连接（在后台线程运行）"""
        if not _SCRCPY_AVAILABLE:
            raise RuntimeError("未安装 scrcpy-client，请执行: pip install scrcpy-client")

        self._ctx.status_text = "正在连接..."
        self._thread = threading.Thread(target=self._connect_worker, daemon=True)
        self._thread.start()

    def _connect_worker(self) -> None:
        emulator_cfg = self._config.get("emulator", {})
        device_id = emulator_cfg.get("device_id", "")
        max_width = emulator_cfg.get("max_width", 800)
        max_fps = emulator_cfg.get("max_fps", 30)

        try:
            client = scrcpy.Client(
                device=device_id,
                max_width=max_width,
                max_fps=max_fps,
            )
            client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)
            client.add_listener(scrcpy.EVENT_INIT, self._on_init)

            self._client = client
            client.start(threaded=False)

            # start() 返回说明流已结束
            if self.is_connected:
                self.is_connected = False
                self._ctx.status_text = "连接已断开"
                if self.on_disconnect:
                    self.on_disconnect()
        except Exception as e:
            self._ctx.status_text = f"连接失败: {e}"
            self.is_connected = False
            if self.on_disconnect:
                self.on_disconnect()

    def _on_init(self) -> None:
        self.is_connected = True
        self._ctx.status_text = f"已连接: {self._config.get('emulator', {}).get('device_id', '')}"
        if self.on_connect:
            self.on_connect()

    def _on_frame(self, frame) -> None:
        if frame is not None:
            self._ctx.set_frame(frame.copy())

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
        self.is_connected = False
        self._ctx.status_text = "已断开连接"

    # ── 触摸控制 ──────────────────────────────────────────

    def click(self, x: int, y: int, down_duration: float = 0.05) -> bool:
        """
        发送点击事件。若未连接则静默失败返回 False。
        注意：此方法可在主线程中调用（down_duration 极短，不影响 Tick）。
        """
        if not self.is_connected or not self._client:
            return False
        try:
            with self._control_lock:
                ctrl = self._client.control
                ctrl.touch(x, y, scrcpy.ACTION_DOWN)
                time.sleep(down_duration)
                ctrl.touch(x, y, scrcpy.ACTION_UP)
            return True
        except Exception:
            return False

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.3,
        *,
        blocking: bool = True,
    ) -> bool:
        """发送滑动事件，duration 单位为秒。"""
        if not self.is_connected or not self._client:
            return False
        if not blocking:
            threading.Thread(
                target=self.swipe,
                args=(x1, y1, x2, y2, duration),
                kwargs={"blocking": True},
                daemon=True,
            ).start()
            return True
        try:
            total = max(0.02, float(duration))
            step_dt = 0.016
            steps = max(1, int(total / step_dt))
            with self._control_lock:
                ctrl = self._client.control
                ctrl.touch(int(x1), int(y1), scrcpy.ACTION_DOWN)
                for i in range(1, steps + 1):
                    t = i / steps
                    x = int(x1 + (x2 - x1) * t)
                    y = int(y1 + (y2 - y1) * t)
                    ctrl.touch(x, y, scrcpy.ACTION_MOVE)
                    time.sleep(total / steps)
                ctrl.touch(int(x2), int(y2), scrcpy.ACTION_UP)
            return True
        except Exception:
            return False

    def press_back(self) -> bool:
        """
        通过 ADB 发送返回键（KEYCODE_BACK）。
        """
        device_id = self._config.get("emulator", {}).get("device_id", "")
        adb_exe = self._config.get("emulator", {}).get("adb_path", "adb")
        cmd = [adb_exe]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["shell", "input", "keyevent", "4"])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3, check=False)
            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="ignore").strip()
                logger.warning(
                    "ADB 返回键执行失败 (code=%s, adb=%s, device=%s, stderr=%s)",
                    result.returncode,
                    adb_exe,
                    device_id or "<default>",
                    err or "<empty>",
                )
                return False
            return True
        except FileNotFoundError:
            logger.warning("找不到 adb 可执行文件: %s", adb_exe)
            return False
        except subprocess.TimeoutExpired:
            logger.warning("ADB 返回键执行超时 (adb=%s, device=%s)", adb_exe, device_id or "<default>")
            return False
        except Exception:
            logger.exception("ADB 返回键执行异常")
            return False

    def adb_swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> bool:
        """
        通过 ADB 发送滑动命令：input swipe x1 y1 x2 y2 duration_ms
        """
        emulator_cfg = self._config.get("emulator", {})
        device_id = emulator_cfg.get("device_id", "")
        adb_exe = emulator_cfg.get("adb_path", "adb")
        duration_ms = max(1, int(duration_ms))

        cmd = [adb_exe]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(
            [
                "shell",
                "input",
                "swipe",
                str(int(x1)),
                str(int(y1)),
                str(int(x2)),
                str(int(y2)),
                str(duration_ms),
            ]
        )
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5, check=False)
            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="ignore").strip()
                logger.warning(
                    "ADB 滑动执行失败 (code=%s, adb=%s, device=%s, stderr=%s, from=(%s,%s), to=(%s,%s), duration_ms=%s)",
                    result.returncode,
                    adb_exe,
                    device_id or "<default>",
                    err or "<empty>",
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    duration_ms,
                )
                return False
            return True
        except FileNotFoundError:
            logger.warning("找不到 adb 可执行文件: %s", adb_exe)
            return False
        except subprocess.TimeoutExpired:
            logger.warning(
                "ADB 滑动执行超时 (adb=%s, device=%s, from=(%s,%s), to=(%s,%s), duration_ms=%s)",
                adb_exe,
                device_id or "<default>",
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                duration_ms,
            )
            return False
        except Exception:
            logger.exception("ADB 滑动执行异常")
            return False

    # ── 设备列表（供 UI 刷新用）─────────────────────────

    @staticmethod
    def list_devices() -> list[str]:
        try:
            if _ADB_AVAILABLE:
                adb = adbutils.AdbClient()
                return [d.serial for d in adb.device_list()]
            r = subprocess.run(["adb", "devices"], capture_output=True, timeout=8)
            lines = r.stdout.decode("utf-8", errors="ignore").splitlines()
            return [ln.split("\t")[0] for ln in lines[1:] if "\tdevice" in ln]
        except Exception:
            return []
