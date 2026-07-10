#!/usr/bin/env python3
"""
实时屏幕 OCR 识别工具
使用 py-scrcpy-client 获取模拟器实时视频流，RapidOCR 识别画面文字并标注。

依赖安装:
    pip install scrcpy-client adbutils opencv-python pillow rapidocr-onnxruntime

使用说明:
    1. 连接 ADB 设备后，点击「刷新」选择设备，再点击「连接设备」
    2. 点击「开始识别」，画面上会实时框出识别到的文字，并显示每帧 OCR 耗时
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import os
import threading
import time
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

try:
    import av as _av
    if not hasattr(_av, "AVError") and hasattr(_av, "FFmpegError"):
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

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    _RAPIDOCR_AVAILABLE = False


def _load_project_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists() or not _YAML_AVAILABLE:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_chinese_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


OCR_PRESETS: Dict[str, Dict[str, Any]] = {
    "快速": {
        "use_cls": False,
        "det_limit_side_len": 320,
        "ocr_max_width": 480,
        "ocr_interval": 8,
        "skip_unchanged": True,
        "show_text_labels": False,
    },
    "平衡": {
        "use_cls": False,
        "det_limit_side_len": 480,
        "ocr_max_width": 640,
        "ocr_interval": 5,
        "skip_unchanged": True,
        "show_text_labels": True,
    },
    "精准": {
        "use_cls": True,
        "det_limit_side_len": 736,
        "ocr_max_width": 0,
        "ocr_interval": 3,
        "skip_unchanged": False,
        "show_text_labels": True,
    },
}


@dataclass
class OcrItem:
    text: str
    score: float
    box: List[List[float]]  # 4 点多边形 [[x,y], ...]
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

    def __post_init__(self):
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        self.x1 = int(min(xs))
        self.y1 = int(min(ys))
        self.x2 = int(max(xs))
        self.y2 = int(max(ys))


def _scale_boxes(box: List[List[float]], sx: float, sy: float) -> List[List[float]]:
    return [[p[0] * sx, p[1] * sy] for p in box]


def _prepare_ocr_frame(frame: np.ndarray, max_width: int) -> Tuple[np.ndarray, float, float]:
    """缩小 OCR 输入，返回 (ocr_frame, scale_x, scale_y) 用于坐标还原。"""
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / w
    new_w = max_width
    new_h = max(1, int(h * scale))
    ocr_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    inv = 1.0 / scale
    return ocr_frame, inv, inv


def _frame_change_score(gray: np.ndarray, prev: Optional[np.ndarray]) -> float:
    """64x64 灰度差分，用于跳过静止画面。"""
    if prev is None:
        return 1.0
    return float(np.mean(cv2.absdiff(gray, prev))) / 255.0


class ScrcpyOcrApp:
    BOX_COLORS = [
        (0, 200, 80),
        (80, 180, 255),
        (255, 180, 40),
        (220, 80, 200),
        (40, 220, 220),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("实时屏幕 OCR  —  py-scrcpy-client + RapidOCR")
        self.root.geometry("1280x860")
        self.root.minsize(960, 640)

        proj_cfg = _load_project_config()
        self._default_device = proj_cfg.get("emulator", {}).get("device_id", "")

        self.current_frame: Optional[np.ndarray] = None
        self.display_scale: float = 1.0
        self._screenshot_photo = None

        self.scrcpy_client = None
        self.is_connected = False
        self.is_recognizing = False

        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)
        self._ocr_lock = threading.Lock()
        self._ocr_engine: Optional[RapidOCR] = None
        self._ocr_engine_key: Optional[tuple] = None
        self._ocr_items: List[OcrItem] = []
        self._last_ocr_ms: float = 0.0
        self._last_ocr_frame_no: int = 0
        self._ocr_running = False
        self._ocr_task_queue: queue.Queue = queue.Queue(maxsize=1)
        self._ocr_worker_started = False
        self._prev_ocr_gray: Optional[np.ndarray] = None
        self._skipped_ocr_count = 0

        self._preset_var = tk.StringVar(value="快速")
        self._ocr_max_width_var = tk.IntVar(value=480)
        self._skip_unchanged_var = tk.BooleanVar(value=True)
        self._show_text_labels_var = tk.BooleanVar(value=False)
        self._change_threshold_var = tk.DoubleVar(value=0.008)

        self._fps_counter = 0
        self._fps_last_time = time.time()
        self._fps_val = 0.0
        self._total_frame_count = 0
        self._last_canvas_coord: Optional[tuple[int, int]] = None

        self._label_font = _resolve_chinese_font(14)

        self._build_ui()
        self._load_config()

        self.root.after(33, self._schedule_frame_update)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not _SCRCPY_AVAILABLE:
            self._log("⚠ 未安装 scrcpy-client，请执行: pip install scrcpy-client")
        if not _ADB_AVAILABLE:
            self._log("⚠ 未安装 adbutils，请执行: pip install adbutils")
        if not _RAPIDOCR_AVAILABLE:
            self._log("⚠ 未安装 RapidOCR，请执行: pip install rapidocr-onnxruntime")

    # ──────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main_pane, width=300)
        right = ttk.Frame(main_pane)
        main_pane.add(left, weight=0)
        main_pane.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        parent.columnconfigure(0, weight=1)

        conn_lf = ttk.LabelFrame(parent, text="  设备连接  ", padding=6)
        conn_lf.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        conn_lf.columnconfigure(1, weight=1)

        ttk.Label(conn_lf, text="设备:").grid(row=0, column=0, sticky="w")
        dev_row = ttk.Frame(conn_lf)
        dev_row.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.device_var = tk.StringVar(value=self._default_device)
        self.device_cb = ttk.Combobox(dev_row, textvariable=self.device_var, width=16)
        self.device_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dev_row, text="刷新", width=5,
                   command=self._refresh_devices).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(conn_lf, text="最大宽度:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.max_width_var = tk.IntVar(value=800)
        ttk.Spinbox(conn_lf, from_=320, to=1920, increment=80,
                    textvariable=self.max_width_var, width=8).grid(
            row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

        ttk.Label(conn_lf, text="最大帧率:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.max_fps_var = tk.IntVar(value=30)
        ttk.Spinbox(conn_lf, from_=5, to=60, increment=5,
                    textvariable=self.max_fps_var, width=8).grid(
            row=2, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

        btn_row = ttk.Frame(conn_lf)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.connect_btn = ttk.Button(btn_row, text="▶ 连接设备",
                                       command=self._toggle_connect, width=12)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        self.conn_status_lbl = ttk.Label(btn_row, text="● 未连接", foreground="#cc4444")
        self.conn_status_lbl.pack(side=tk.LEFT, padx=(8, 0))

        ocr_lf = ttk.LabelFrame(parent, text="  OCR 参数  ", padding=6)
        ocr_lf.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        ocr_lf.columnconfigure(1, weight=1)

        ttk.Label(ocr_lf, text="速度预设:").grid(row=0, column=0, sticky="w")
        preset_cb = ttk.Combobox(
            ocr_lf, textvariable=self._preset_var,
            values=list(OCR_PRESETS.keys()), state="readonly", width=10)
        preset_cb.grid(row=0, column=1, sticky="w", padx=(4, 0))
        preset_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())

        ttk.Label(ocr_lf, text="识别每N帧:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.ocr_interval_var = tk.IntVar(value=8)
        ttk.Spinbox(ocr_lf, from_=1, to=60, textvariable=self.ocr_interval_var,
                    width=6).grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

        ttk.Label(ocr_lf, text="OCR最大宽:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Spinbox(ocr_lf, from_=0, to=1280, increment=80,
                    textvariable=self._ocr_max_width_var, width=6).grid(
            row=2, column=1, sticky="w", padx=(4, 0), pady=(4, 0))
        ttk.Label(ocr_lf, text="0=原图", font=("", 8)).grid(
            row=2, column=1, sticky="e", padx=(4, 0), pady=(4, 0))

        ttk.Label(ocr_lf, text="置信阈值:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        thresh_row = ttk.Frame(ocr_lf)
        thresh_row.grid(row=3, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.score_var = tk.DoubleVar(value=0.50)
        self.score_lbl = ttk.Label(thresh_row, text="0.50", width=5, anchor="e")
        self.score_lbl.pack(side=tk.RIGHT)
        ttk.Scale(thresh_row, from_=0.10, to=0.99, variable=self.score_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.score_lbl.configure(text=f"{float(v):.2f}"),
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        opt_row = ttk.Frame(ocr_lf)
        opt_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(opt_row, text="跳过静止画面",
                        variable=self._skip_unchanged_var).pack(anchor="w")
        ttk.Checkbutton(opt_row, text="显示文字标签",
                        variable=self._show_text_labels_var).pack(anchor="w")

        ctrl_row = ttk.Frame(ocr_lf)
        ctrl_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.ocr_btn = ttk.Button(ctrl_row, text="▶ 开始识别",
                                  command=self._toggle_recognizing, width=12)
        self.ocr_btn.pack(side=tk.LEFT, padx=2)
        self.show_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_row, text="显示标注框",
                        variable=self.show_overlay_var).pack(side=tk.LEFT, padx=(4, 0))

        stat_lf = ttk.LabelFrame(parent, text="  识别统计  ", padding=6)
        stat_lf.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        self.stat_text_var = tk.StringVar(value="尚未识别")
        ttk.Label(stat_lf, textvariable=self.stat_text_var, wraplength=260,
                  justify=tk.LEFT).pack(anchor="w")

        log_lf = ttk.LabelFrame(parent, text="  操作日志  ", padding=4)
        log_lf.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 4))
        parent.rowconfigure(3, weight=1)
        self.log_txt = scrolledtext.ScrolledText(
            log_lf, height=12, state=tk.DISABLED,
            font=("Consolas", 9), bg="#0d0d1a", fg="#cccccc",
            insertbackground="white")
        self.log_txt.pack(fill=tk.BOTH, expand=True)
        ttk.Button(log_lf, text="清空日志", command=self._clear_log,
                   width=8).pack(anchor="e", pady=(2, 0))

    def _build_right(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        view_lf = ttk.LabelFrame(
            parent,
            text="  实时屏幕  —  绿色框为 OCR 识别结果，左上角显示本帧识别耗时  ",
            padding=4)
        view_lf.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        view_lf.rowconfigure(0, weight=1)
        view_lf.columnconfigure(0, weight=1)

        canvas_frame = ttk.Frame(view_lf)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#0a0a1a", cursor="crosshair",
                                highlightthickness=0)
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        hbar.grid(row=1, column=0, sticky="ew")
        vbar.grid(row=0, column=1, sticky="ns")

        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

        toolbar = ttk.Frame(view_lf)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.coord_lbl = ttk.Label(toolbar, text="坐标: —", width=20, anchor="w")
        self.coord_lbl.pack(side=tk.LEFT)
        self.fps_lbl = ttk.Label(toolbar, text="FPS: —", width=12, anchor="w")
        self.fps_lbl.pack(side=tk.LEFT)
        self.ocr_speed_lbl = ttk.Label(toolbar, text="OCR: —", width=18, anchor="w")
        self.ocr_speed_lbl.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="保存当前帧", command=self._save_frame).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="原始大小", command=self._zoom_actual).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="适合窗口", command=lambda: self._redisplay(fit=True)).pack(
            side=tk.RIGHT, padx=2)

        res_lf = ttk.LabelFrame(parent, text="  识别结果  ", padding=4)
        res_lf.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        res_lf.columnconfigure(0, weight=1)

        cols = ("text", "score", "x", "y", "w", "h")
        self.res_tree = ttk.Treeview(res_lf, columns=cols, show="headings", height=8)
        for col, hdr, w, anchor in [
            ("text",  "文字",   260, "w"),
            ("score", "置信度",  72, "center"),
            ("x",     "X",       56, "center"),
            ("y",     "Y",       56, "center"),
            ("w",     "宽",      56, "center"),
            ("h",     "高",      56, "center"),
        ]:
            self.res_tree.heading(col, text=hdr)
            self.res_tree.column(col, width=w, anchor=anchor, stretch=(col == "text"))
        self.res_tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(res_lf, orient=tk.VERTICAL, command=self.res_tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.res_tree.configure(yscrollcommand=sb.set)

        statusbar = ttk.Frame(parent)
        statusbar.grid(row=2, column=0, sticky="ew", padx=4)
        self.status_var = tk.StringVar(value="就绪  —  请先连接设备")
        ttk.Label(statusbar, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(statusbar, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT)

    # ──────────────────────────────────────────
    # 设备连接
    # ──────────────────────────────────────────

    def _refresh_devices(self):
        def _work():
            try:
                if _ADB_AVAILABLE:
                    adb = adbutils.AdbClient()
                    devices = [d.serial for d in adb.device_list()]
                else:
                    import subprocess
                    r = subprocess.run(["adb", "devices"], capture_output=True, timeout=8)
                    lines = r.stdout.decode("utf-8", errors="ignore").splitlines()
                    devices = [ln.split("\t")[0] for ln in lines[1:] if "\tdevice" in ln]

                def _update():
                    if devices:
                        self.device_cb["values"] = devices
                        if self.device_var.get() not in devices:
                            self.device_var.set(devices[0])
                        self._log(f"发现 {len(devices)} 个设备: {', '.join(devices)}")
                    else:
                        self.device_cb["values"] = []
                        self._log("未发现 ADB 设备，请检查连接")

                self.root.after(0, _update)
            except Exception as e:
                self._log(f"刷新设备失败: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def _toggle_connect(self):
        if self.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        if not _SCRCPY_AVAILABLE:
            messagebox.showerror("缺少依赖",
                "未安装 py-scrcpy-client\n\n请执行:\n  pip install scrcpy-client adbutils")
            return

        device_id = self.device_var.get().strip()
        if not device_id:
            messagebox.showwarning("未选择设备", "请先在设备下拉框中选择或输入设备 ID")
            return

        self._set_status("正在连接…")
        self.progress.start()
        self.connect_btn.configure(state=tk.DISABLED)

        def _do_connect():
            try:
                client = scrcpy.Client(
                    device=device_id,
                    max_width=self.max_width_var.get(),
                    max_fps=self.max_fps_var.get(),
                )
                client.add_listener(scrcpy.EVENT_FRAME, self._on_scrcpy_frame)
                client.add_listener(scrcpy.EVENT_INIT, self._on_scrcpy_init)
                self.scrcpy_client = client
                self._total_frame_count = 0
                self._fps_counter = 0
                self._fps_last_time = time.time()
                client.start(threaded=False)
                if self.is_connected:
                    self.root.after(0, self._handle_disconnect)
            except Exception as e:
                err_msg = str(e)

                def _on_err(msg=err_msg):
                    self.connect_btn.configure(state=tk.NORMAL)
                    self.progress.stop()
                    self._set_status("连接失败")
                    self._log(f"连接失败: {msg}")
                    messagebox.showerror("连接失败", msg)

                self.root.after(0, _on_err)

        threading.Thread(target=_do_connect, daemon=True).start()

    def _on_scrcpy_init(self):
        self.is_connected = True
        device_id = self.device_var.get()

        def _update_ui():
            self.connect_btn.configure(text="■ 断开连接", state=tk.NORMAL)
            self.conn_status_lbl.configure(text="● 已连接", foreground="#22aa55")
            self.progress.stop()
            self._set_status(
                f"已连接: {device_id}  |  最大宽度: {self.max_width_var.get()}"
                f"  |  最大帧率: {self.max_fps_var.get()}")
            self._log(f"连接成功: {device_id}")

        self.root.after(0, _update_ui)

    def _disconnect(self):
        self.is_recognizing = False
        self.ocr_btn.configure(text="▶ 开始识别")

        if self.scrcpy_client:
            try:
                self.scrcpy_client.stop()
            except Exception:
                pass
            self.scrcpy_client = None

        self.is_connected = False
        self.connect_btn.configure(text="▶ 连接设备")
        self.conn_status_lbl.configure(text="● 未连接", foreground="#cc4444")
        self._set_status("已断开连接")
        self._log("连接已断开")

    def _handle_disconnect(self):
        if self.is_connected:
            self.is_connected = False
            self.is_recognizing = False
            self.scrcpy_client = None
            self.connect_btn.configure(text="▶ 连接设备")
            self.conn_status_lbl.configure(text="● 已断开", foreground="#cc8800")
            self.ocr_btn.configure(text="▶ 开始识别")
            self._set_status("scrcpy 连接已断开")
            self._log("scrcpy 连接已断开")

    # ──────────────────────────────────────────
    # 帧处理 & OCR
    # ──────────────────────────────────────────

    def _on_scrcpy_frame(self, frame: Optional[np.ndarray]):
        if frame is None:
            return
        self._total_frame_count += 1
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def _schedule_frame_update(self):
        self._process_frame_queue()
        self.root.after(33, self._schedule_frame_update)

    def _process_frame_queue(self):
        frame = None
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if frame is None:
            return

        self.current_frame = frame

        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_val = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = now
            self.fps_lbl.configure(text=f"FPS: {self._fps_val:.1f}")

        interval = max(1, self.ocr_interval_var.get())
        if self.is_recognizing and (self._total_frame_count % interval == 0):
            self._submit_ocr_task(frame, self._total_frame_count)

        self._redisplay()

    def _apply_preset(self):
        preset = OCR_PRESETS.get(self._preset_var.get())
        if not preset:
            return
        self.ocr_interval_var.set(preset["ocr_interval"])
        self._ocr_max_width_var.set(preset["ocr_max_width"])
        self._skip_unchanged_var.set(preset["skip_unchanged"])
        self._show_text_labels_var.set(preset["show_text_labels"])
        self._reset_ocr_engine()
        self._log(f"已切换预设: {self._preset_var.get()}")

    def _ocr_engine_config(self) -> Dict[str, Any]:
        preset = OCR_PRESETS.get(self._preset_var.get(), OCR_PRESETS["平衡"])
        threads = min(4, os.cpu_count() or 4)
        return {
            "use_cls": preset["use_cls"],
            "det_limit_side_len": preset["det_limit_side_len"],
            "intra_op_num_threads": threads,
            "inter_op_num_threads": 1,
        }

    def _reset_ocr_engine(self):
        self._ocr_engine = None
        self._ocr_engine_key = None

    def _ensure_ocr_engine(self) -> bool:
        if not _RAPIDOCR_AVAILABLE:
            return False
        cfg = self._ocr_engine_config()
        key = tuple(sorted(cfg.items()))
        if self._ocr_engine is not None and self._ocr_engine_key == key:
            return True
        try:
            if self._ocr_engine is None:
                self._log("正在加载 RapidOCR 模型…")
            self._ocr_engine = RapidOCR(**cfg)
            self._ocr_engine_key = key
            # 预热推理，避免首帧卡顿
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._ocr_engine(dummy)
            self._log(f"RapidOCR 就绪 ({self._preset_var.get()}, threads={cfg['intra_op_num_threads']})")
            return True
        except Exception as e:
            self._log(f"RapidOCR 初始化失败: {e}")
            return False

    def _ensure_ocr_worker(self):
        if self._ocr_worker_started:
            return
        self._ocr_worker_started = True
        threading.Thread(target=self._ocr_worker_loop, daemon=True).start()

    def _submit_ocr_task(self, frame: np.ndarray, frame_no: int):
        self._ensure_ocr_worker()
        task = (frame.copy(), frame_no)
        if self._ocr_task_queue.full():
            try:
                self._ocr_task_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._ocr_task_queue.put_nowait(task)
        except queue.Full:
            pass

    def _ocr_worker_loop(self):
        while True:
            frame, frame_no = self._ocr_task_queue.get()
            if frame is None:
                break
            self._run_ocr_on_frame(frame, frame_no)

    def _should_skip_ocr(self, frame: np.ndarray) -> bool:
        if not self._skip_unchanged_var.get():
            return False
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        skip = False
        if self._prev_ocr_gray is not None:
            score = _frame_change_score(gray, self._prev_ocr_gray)
            if score < self._change_threshold_var.get():
                skip = True
                self._skipped_ocr_count += 1
        self._prev_ocr_gray = gray
        return skip

    def _run_ocr_on_frame(self, frame: np.ndarray, frame_no: int):
        if not self._ocr_lock.acquire(blocking=False):
            return
        self._ocr_running = True
        try:
            if not self.is_recognizing:
                return
            if self._should_skip_ocr(frame):
                return
            if not self._ensure_ocr_engine():
                return

            ocr_frame, sx, sy = _prepare_ocr_frame(frame, self._ocr_max_width_var.get())

            t0 = time.perf_counter()
            result, elapse = self._ocr_engine(ocr_frame)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            if isinstance(elapse, list):
                engine_ms = sum(elapse) * 1000.0
            elif elapse is not None:
                engine_ms = float(elapse) * 1000.0
            else:
                engine_ms = wall_ms

            min_score = self.score_var.get()
            items: List[OcrItem] = []
            if result:
                for row in result:
                    if len(row) < 3:
                        continue
                    box, text, score = row[0], str(row[1]), float(row[2])
                    if score < min_score or not text.strip():
                        continue
                    if sx != 1.0 or sy != 1.0:
                        box = _scale_boxes(box, sx, sy)
                    items.append(OcrItem(text=text, score=score, box=box))

            self._ocr_items = items
            self._last_ocr_ms = wall_ms
            self._last_ocr_frame_no = frame_no
            skipped = self._skipped_ocr_count
            self._skipped_ocr_count = 0

            def _update_ui():
                self.ocr_speed_lbl.configure(
                    text=f"OCR: {wall_ms:.0f}ms ({len(items)}条)")
                skip_note = f"\n跳过静止: {skipped}次" if skipped else ""
                self.stat_text_var.set(
                    f"帧 #{frame_no}\n"
                    f"识别耗时: {wall_ms:.1f} ms\n"
                    f"引擎耗时: {engine_ms:.1f} ms\n"
                    f"识别条数: {len(items)}\n"
                    f"预设: {self._preset_var.get()}"
                    f"{skip_note}")
                self._refresh_results_tree()

            self.root.after(0, _update_ui)
        except Exception as e:
            self._log(f"OCR 识别失败: {e}")
        finally:
            self._ocr_running = False
            self._ocr_lock.release()

    def _toggle_recognizing(self):
        if not self.is_connected:
            messagebox.showwarning("未连接", "请先连接设备")
            return
        if not _RAPIDOCR_AVAILABLE:
            messagebox.showerror("缺少依赖",
                "未安装 RapidOCR\n\n请执行:\n  pip install rapidocr-onnxruntime")
            return

        self.is_recognizing = not self.is_recognizing
        if self.is_recognizing:
            self.ocr_btn.configure(text="■ 停止识别")
            self._set_status("实时 OCR 识别中…")
            self._prev_ocr_gray = None
            self._skipped_ocr_count = 0
            self._log("开始实时 OCR 识别")
            threading.Thread(target=self._ensure_ocr_engine, daemon=True).start()
        else:
            self.ocr_btn.configure(text="▶ 开始识别")
            self._set_status("OCR 已停止")
            self._log("停止 OCR 识别")

    # ──────────────────────────────────────────
    # 画面渲染
    # ──────────────────────────────────────────

    def _draw_ocr_overlay(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self._ocr_items:
            return self._draw_speed_banner(img_bgr)

        for idx, item in enumerate(self._ocr_items):
            color = self.BOX_COLORS[idx % len(self.BOX_COLORS)]
            pts = np.array(item.box, dtype=np.int32)
            cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)

        if self._show_text_labels_var.get():
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            for idx, item in enumerate(self._ocr_items):
                color = self.BOX_COLORS[idx % len(self.BOX_COLORS)]
                label = f"{item.text} ({item.score:.2f})"
                tx, ty = item.x1, max(item.y1 - 22, 0)
                bbox = draw.textbbox((tx, ty), label, font=self._label_font)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((tx, ty), label, fill=color, font=self._label_font)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return self._draw_speed_banner(img_bgr)

    def _draw_speed_banner(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self.is_recognizing:
            return img_bgr

        h, w = img_bgr.shape[:2]
        if self._last_ocr_ms > 0:
            status = f"OCR #{self._last_ocr_frame_no}: {self._last_ocr_ms:.1f} ms"
            sub = f"{len(self._ocr_items)} texts"
        elif self._ocr_running:
            status = "OCR 识别中…"
            sub = ""
        else:
            status = "等待首次 OCR…"
            sub = ""

        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (min(w, 360), 56 if sub else 34), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)
        cv2.putText(img_bgr, status, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 120), 2, cv2.LINE_AA)
        if sub:
            cv2.putText(img_bgr, sub, (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1, cv2.LINE_AA)
        return img_bgr

    def _redisplay(self, fit: bool = False):
        if self.current_frame is None:
            return

        img = self.current_frame
        if self.show_overlay_var.get() and (self._ocr_items or self.is_recognizing):
            img = self._draw_ocr_overlay(img.copy())
        else:
            img = img.copy()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_w, img_h = pil_img.size

        if fit:
            cw = max(self.canvas.winfo_width(), 200)
            ch = max(self.canvas.winfo_height(), 200)
            self.display_scale = min(cw / img_w, ch / img_h)

        dw = max(1, int(img_w * self.display_scale))
        dh = max(1, int(img_h * self.display_scale))
        pil_img = pil_img.resize((dw, dh), Image.NEAREST)
        self._screenshot_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._screenshot_photo)
        self.canvas.configure(scrollregion=(0, 0, dw, dh))

    def _zoom_actual(self):
        self.display_scale = 1.0
        self._redisplay(fit=False)

    def _on_canvas_resize(self):
        if self.current_frame is not None:
            self._redisplay(fit=True)

    def _on_canvas_motion(self, event):
        if self.current_frame is None:
            return
        ax = int(self.canvas.canvasx(event.x) / self.display_scale)
        ay = int(self.canvas.canvasy(event.y) / self.display_scale)
        self._last_canvas_coord = (ax, ay)
        self.coord_lbl.configure(text=f"坐标: ({ax}, {ay})")

    def _on_canvas_click(self, event):
        if self.current_frame is None:
            return
        ax = int(self.canvas.canvasx(event.x) / self.display_scale)
        ay = int(self.canvas.canvasy(event.y) / self.display_scale)
        self._last_canvas_coord = (ax, ay)

        if self.scrcpy_client and self.is_connected:
            try:
                ctrl = self.scrcpy_client.control
                ctrl.touch(ax, ay, scrcpy.ACTION_DOWN)
                time.sleep(0.05)
                ctrl.touch(ax, ay, scrcpy.ACTION_UP)
                self._log(f"触摸: ({ax}, {ay})")
            except Exception as e:
                self._log(f"触摸事件发送失败: {e}")
        else:
            self._log(f"坐标: ({ax}, {ay})  (未连接，仅记录)")

    def _refresh_results_tree(self):
        self.res_tree.delete(*self.res_tree.get_children())
        for item in self._ocr_items:
            w = item.x2 - item.x1
            h = item.y2 - item.y1
            self.res_tree.insert("", "end", values=(
                item.text, f"{item.score:.4f}", item.x1, item.y1, w, h))

    def _save_frame(self):
        if self.current_frame is None:
            messagebox.showwarning("无画面", "暂无可保存的画面")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="保存当前帧",
            defaultextension=".png",
            initialfile=f"ocr_frame_{ts}.png",
            filetypes=[("PNG 图像", "*.png"), ("JPEG 图像", "*.jpg"), ("全部", "*.*")],
        )
        if path:
            img = self.current_frame.copy()
            if self.show_overlay_var.get():
                img = self._draw_ocr_overlay(img)
            cv2.imwrite(path, img)
            self._log(f"帧已保存: {path}")

    def _load_config(self):
        cfg = _load_project_config()
        emu_cfg = cfg.get("emulator", {})
        ocr_cfg = cfg.get("scrcpy_ocr", {})

        device_id = emu_cfg.get("device_id", "")
        if device_id:
            self.device_var.set(device_id)
        self.max_width_var.set(emu_cfg.get("max_width", 800))
        self.max_fps_var.set(emu_cfg.get("max_fps", 30))

        preset = ocr_cfg.get("preset", "快速")
        if preset in OCR_PRESETS:
            self._preset_var.set(preset)
        self._apply_preset()

        self.ocr_interval_var.set(int(ocr_cfg.get("ocr_interval", self.ocr_interval_var.get())))
        self._ocr_max_width_var.set(int(ocr_cfg.get("ocr_max_width", self._ocr_max_width_var.get())))
        self._skip_unchanged_var.set(bool(ocr_cfg.get("skip_unchanged", self._skip_unchanged_var.get())))
        self._show_text_labels_var.set(bool(ocr_cfg.get("show_text_labels", self._show_text_labels_var.get())))
        score = float(ocr_cfg.get("min_score", 0.50))
        self.score_var.set(score)
        self.score_lbl.configure(text=f"{score:.2f}")
        self._log("配置已加载")

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────

    def _log(self, msg: str):
        def _do():
            self.log_txt.configure(state=tk.NORMAL)
            ts = time.strftime("%H:%M:%S")
            self.log_txt.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_txt.see(tk.END)
            self.log_txt.configure(state=tk.DISABLED)

        self.root.after(0, _do)

    def _clear_log(self):
        self.log_txt.configure(state=tk.NORMAL)
        self.log_txt.delete("1.0", tk.END)
        self.log_txt.configure(state=tk.DISABLED)

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _on_close(self):
        self._disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    ScrcpyOcrApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
