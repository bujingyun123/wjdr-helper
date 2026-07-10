#!/usr/bin/env python3
"""
实时屏幕查看与图标跟踪工具
使用 py-scrcpy-client 获取模拟器实时视频流，OpenCV 模板匹配实时跟踪图标。

依赖安装:
    pip install scrcpy-client adbutils opencv-python pillow

使用说明:
    1. 连接 ADB 设备后，点击「刷新」选择设备，再点击「连接设备」
    2. 通过「添加模板」、「粘贴」或「截帧裁图」添加要跟踪的图标
    3. 点击「开始跟踪」，画面上会实时标注匹配到的图标位置
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import shutil
import threading
import time
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable

try:
    import av as _av
    # PyAV >= 10.0 将 av.AVError 重命名为 av.FFmpegError，scrcpy-client 仍使用旧名称
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

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _load_project_config() -> dict:
    """加载 config.yaml，返回完整配置字典"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists() or not _YAML_AVAILABLE:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

@dataclass
class Template:
    name: str
    path: str
    enabled: bool = True
    last_confidence: float = 0.0
    last_x: int = 0
    last_y: int = 0
    last_found: bool = False


@dataclass
class MatchResult:
    confidence: float
    x: int
    y: int
    w: int
    h: int
    found: bool


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

class ScrcpyTrackerApp:
    # 模板图片统一保存到 assets/templates（与主流程共用）
    # 独立运行时 Path(__file__) = tools/scrcpy_tracker.py → 上两级为项目根
    TEMPLATES_DIR = Path(__file__).parent.parent / "assets" / "templates"

    CV_METHODS = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR_NORMED":  cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }

    PREVIEW_AREA_HEIGHT = 180

    def __init__(self, root: tk.Tk, shared_ctx=None):
        """
        :param root:       Tk 根窗口或 Toplevel
        :param shared_ctx: 可选 SharedContext。
                           传入后进入「集成模式」：帧来自共享上下文，
                           连接区隐藏，模板列表与主流程共用。
                           不传则保持原有独立运行行为不变。
        """
        self.root = root
        self._shared_ctx = shared_ctx  # None → 独立模式；非 None → 集成模式
        self.root.title("实时屏幕查看与图标跟踪  —  py-scrcpy-client + OpenCV")
        self.root.geometry("1440x880")
        self.root.minsize(1000, 680)

        proj_cfg = _load_project_config()
        self._default_device = proj_cfg.get("emulator", {}).get("device_id", "")

        # 状态
        self.templates: List[Template] = []
        self.current_frame: Optional[np.ndarray] = None  # BGR 格式
        self.display_scale: float = 1.0
        self.match_results: Dict[str, MatchResult] = {}
        self._screenshot_photo = None
        self._preview_photo = None
        self._last_canvas_coord: Optional[tuple[int, int]] = None

        # scrcpy 相关
        self.scrcpy_client = None
        self.is_connected: bool = False
        self.is_tracking: bool = False
        # 集成模式触摸代理：由主程序注入（例如 Device.click）
        self.external_click_handler: Optional[Callable[[int, int], bool]] = None

        # 帧队列：scrcpy 回调线程 → UI 线程
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)
        self._matching_lock = threading.Lock()

        # FPS 统计
        self._fps_counter: int = 0
        self._fps_last_time: float = time.time()
        self._fps_val: float = 0.0
        self._total_frame_count: int = 0

        # 框选裁图时的临时标记
        self._crop_active: bool = False

        # 集成模式：首帧标记（用于自动 fit）
        self._integrated_first_frame: bool = True

        # 模板图像内存缓存：path → BGR ndarray，避免每次 matchTemplate 前从磁盘读取
        self._tmpl_cache: Dict[str, np.ndarray] = {}

        self._build_ui()
        self._load_config()

        # 集成模式：把已加载的模板合并到 SharedContext，并共用同一份列表
        if self._shared_ctx is not None:
            existing_paths = {t.path for t in self._shared_ctx.templates}
            for t in self.templates:
                if t.path not in existing_paths:
                    self._shared_ctx.templates.append(t)
            self.templates = self._shared_ctx.templates  # 共用同一份引用

        self._schedule_frame_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not _SCRCPY_AVAILABLE:
            self._log("⚠ 未安装 scrcpy-client，请执行: pip install scrcpy-client")
        if not _ADB_AVAILABLE:
            self._log("⚠ 未安装 adbutils，请执行: pip install adbutils")

    # ──────────────────────────────────────────
    # UI 构建
    # ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main_pane, width=330)
        right = ttk.Frame(main_pane)
        main_pane.add(left, weight=0)
        main_pane.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

    # ---- 左侧面板 ----

    def _build_left(self, parent):
        parent.columnconfigure(0, weight=1)

        # 连接控制
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

        # 集成模式：隐藏设备连接区（连接由主流程统一管理）
        if self._shared_ctx is not None:
            conn_lf.grid_remove()

        # 模板列表
        tmpl_lf = ttk.LabelFrame(parent, text="  模板图像  ", padding=6)
        tmpl_lf.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        parent.rowconfigure(1, weight=1)

        cols = ("on", "name", "conf")
        self.tmpl_tree = ttk.Treeview(tmpl_lf, columns=cols, show="headings", height=10)
        self.tmpl_tree.heading("on",   text="启")
        self.tmpl_tree.heading("name", text="模板名称")
        self.tmpl_tree.heading("conf", text="置信度")
        self.tmpl_tree.column("on",   width=28, anchor="center", stretch=False)
        self.tmpl_tree.column("name", width=160)
        self.tmpl_tree.column("conf", width=65, anchor="center", stretch=False)

        vsb = ttk.Scrollbar(tmpl_lf, orient=tk.VERTICAL, command=self.tmpl_tree.yview)
        self.tmpl_tree.configure(yscrollcommand=vsb.set)
        self.tmpl_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        tmpl_lf.rowconfigure(0, weight=1)
        tmpl_lf.columnconfigure(0, weight=1)
        self.tmpl_tree.bind("<<TreeviewSelect>>", self._on_tmpl_select)
        self.tmpl_tree.bind("<Double-1>", lambda e: self._rename_template())

        btn_row2 = ttk.Frame(tmpl_lf)
        btn_row2.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(btn_row2, text="添加图像",   command=self._add_template,   width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="粘贴 Ctrl+V", command=self._paste_template, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="截帧裁图",   command=self._crop_from_frame, width=8).pack(side=tk.LEFT, padx=2)

        btn_row3 = ttk.Frame(tmpl_lf)
        btn_row3.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(btn_row3, text="删除",   command=self._remove_template, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row3, text="切换启用", command=self._toggle_template, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row3, text="重命名",  command=self._rename_template,  width=7).pack(side=tk.LEFT, padx=2)

        # 模板预览
        prev_lf = ttk.LabelFrame(parent, text="  模板预览  ", padding=4)
        prev_lf.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        parent.rowconfigure(2, weight=0, minsize=self.PREVIEW_AREA_HEIGHT)

        preview_frame = ttk.Frame(prev_lf, height=self.PREVIEW_AREA_HEIGHT - 28)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        preview_frame.pack_propagate(False)

        self.preview_lbl = ttk.Label(
            preview_frame,
            text="← 选择模板查看预览",
            anchor="center",
            relief="sunken",
            background="#1e1e1e",
            foreground="#aaaaaa",
        )
        self.preview_lbl.pack(fill=tk.BOTH, expand=True)

        # 跟踪参数
        cfg_lf = ttk.LabelFrame(parent, text="  跟踪参数  ", padding=6)
        cfg_lf.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 4))
        cfg_lf.columnconfigure(1, weight=1)

        ttk.Label(cfg_lf, text="匹配方法:").grid(row=0, column=0, sticky="w")
        self.method_var = tk.StringVar(value="TM_CCOEFF_NORMED")
        self.method_var.trace_add("write", self._on_method_change)
        ttk.Combobox(cfg_lf, textvariable=self.method_var,
                     values=list(self.CV_METHODS.keys()),
                     state="readonly", width=20).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(cfg_lf, text="置信阈值:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        thresh_row = ttk.Frame(cfg_lf)
        thresh_row.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.threshold_var = tk.DoubleVar(value=0.80)
        self.thresh_lbl = ttk.Label(thresh_row, text="0.80", width=5, anchor="e")
        self.thresh_lbl.pack(side=tk.RIGHT)
        ttk.Scale(thresh_row, from_=0.10, to=1.00, variable=self.threshold_var,
                  orient=tk.HORIZONTAL,
                  command=self._on_threshold_change,
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(cfg_lf, text="模板缩放:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        scale_row = ttk.Frame(cfg_lf)
        scale_row.grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.scale_var = tk.DoubleVar(value=1.00)
        self.scale_lbl = ttk.Label(scale_row, text="1.00", width=5, anchor="e")
        self.scale_lbl.pack(side=tk.RIGHT)
        ttk.Scale(scale_row, from_=0.25, to=3.00, variable=self.scale_var,
                  orient=tk.HORIZONTAL,
                  command=self._on_scale_change,
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(cfg_lf, text="跟踪每N帧:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.track_interval_var = tk.IntVar(value=3)
        ttk.Spinbox(cfg_lf, from_=1, to=30, textvariable=self.track_interval_var,
                    width=6).grid(row=3, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

        # 跟踪控制按钮
        track_row = ttk.Frame(cfg_lf)
        track_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.track_btn = ttk.Button(track_row, text="▶ 开始跟踪",
                                     command=self._toggle_tracking, width=12)
        self.track_btn.pack(side=tk.LEFT, padx=2)
        self.show_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(track_row, text="显示标注框",
                        variable=self.show_overlay_var).pack(side=tk.LEFT, padx=(4, 0))

        # Ctrl+V 全局绑定
        self.root.bind("<Control-v>", lambda e: self._paste_template())

    # ---- 右侧画面区域 ----

    def _build_right(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        view_lf = ttk.LabelFrame(
            parent,
            text="  实时屏幕  —  鼠标悬停显示坐标，左键点击发送触摸事件  ",
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
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,   command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        hbar.grid(row=1, column=0, sticky="ew")
        vbar.grid(row=0, column=1, sticky="ns")

        self.canvas.bind("<Motion>",    self._on_canvas_motion)
        self.canvas.bind("<Button-1>",  self._on_canvas_click)
        self.canvas.bind("<Button-3>",  self._copy_coord_from_event)
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

        # 画面工具栏
        toolbar = ttk.Frame(view_lf)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.coord_lbl = ttk.Label(toolbar, text="坐标: —", width=20, anchor="w")
        self.coord_lbl.pack(side=tk.LEFT)
        self.fps_lbl = ttk.Label(toolbar, text="FPS: —", width=12, anchor="w")
        self.fps_lbl.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="复制坐标", command=self._copy_current_coord).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="保存当前帧", command=self._save_frame).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="原始大小",   command=self._zoom_actual).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="适合窗口",   command=lambda: self._redisplay(fit=True)).pack(side=tk.RIGHT, padx=2)

        # 下方面板：跟踪结果 + 日志
        bottom = ttk.Frame(parent)
        bottom.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        res_lf = ttk.LabelFrame(bottom, text="  跟踪结果  ", padding=4)
        res_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        res_lf.columnconfigure(0, weight=1)

        rcols = ("template", "confidence", "x", "y", "status")
        self.res_tree = ttk.Treeview(res_lf, columns=rcols, show="headings", height=5)
        for col, hdr, w, anchor in [
            ("template",   "模板",   130, "w"),
            ("confidence", "置信度",  72, "center"),
            ("x",          "X",       60, "center"),
            ("y",          "Y",       60, "center"),
            ("status",     "状态",    80, "center"),
        ]:
            self.res_tree.heading(col, text=hdr)
            self.res_tree.column(col, width=w, anchor=anchor, stretch=(col == "template"))
        self.res_tree.grid(row=0, column=0, sticky="nsew")
        res_sb = ttk.Scrollbar(res_lf, orient=tk.VERTICAL, command=self.res_tree.yview)
        res_sb.grid(row=0, column=1, sticky="ns")
        self.res_tree.configure(yscrollcommand=res_sb.set)
        self.res_tree.tag_configure("found",    foreground="#00aa44")
        self.res_tree.tag_configure("notfound", foreground="#cc4444")

        log_lf = ttk.LabelFrame(bottom, text="  操作日志  ", padding=4)
        log_lf.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        self.log_txt = scrolledtext.ScrolledText(
            log_lf, height=5, state=tk.DISABLED,
            font=("Consolas", 9), bg="#0d0d1a", fg="#cccccc",
            insertbackground="white")
        self.log_txt.pack(fill=tk.BOTH, expand=True)
        ttk.Button(log_lf, text="清空日志", command=self._clear_log,
                   width=8).pack(anchor="e", pady=(2, 0))

        # 状态栏
        statusbar = ttk.Frame(parent)
        statusbar.grid(row=2, column=0, sticky="ew", padx=4)
        self.status_var = tk.StringVar(value="就绪  —  请先连接设备")
        ttk.Label(statusbar, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(statusbar, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT)

    # ──────────────────────────────────────────
    # 设备刷新 & 连接
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
                # 直接传 serial 字符串，scrcpy 内部通过 adb.device(serial=...) 查找设备
                client = scrcpy.Client(
                    device=device_id,
                    max_width=self.max_width_var.get(),
                    max_fps=self.max_fps_var.get(),
                )
                client.add_listener(scrcpy.EVENT_FRAME, self._on_scrcpy_frame)
                # EVENT_INIT 在连接建立、第一帧前触发，用于更新 UI 连接状态
                client.add_listener(scrcpy.EVENT_INIT,  self._on_scrcpy_init)

                self.scrcpy_client = client
                self._total_frame_count = 0
                self._fps_counter = 0
                self._fps_last_time = time.time()

                # 阻塞运行直到流结束，线程结束时自动检测断线
                client.start(threaded=False)

                # start() 正常返回 → 流已结束（主动 stop 或异常）
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
        """scrcpy 连接建立后（第一帧前）回调，在工作线程中调用"""
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
        self.is_tracking = False
        self.track_btn.configure(text="▶ 开始跟踪")

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
        """流线程意外结束时调用（非主动断开）"""
        if self.is_connected:
            self.is_connected = False
            self.is_tracking = False
            self.scrcpy_client = None
            self.connect_btn.configure(text="▶ 连接设备")
            self.conn_status_lbl.configure(text="● 已断开", foreground="#cc8800")
            self.track_btn.configure(text="▶ 开始跟踪")
            self._set_status("scrcpy 连接已断开")
            self._log("scrcpy 连接已断开")

    # ──────────────────────────────────────────
    # 帧接收与处理
    # ──────────────────────────────────────────

    def _on_scrcpy_frame(self, frame: Optional[np.ndarray]):
        """scrcpy 工作线程回调，接收 BGR 帧"""
        if frame is None:
            return

        self._total_frame_count += 1

        # 非阻塞入队：队列满时丢弃最旧帧
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
        """每 33ms 驱动一次 UI 帧刷新（约 30fps）"""
        self._process_frame_queue()
        self.root.after(33, self._schedule_frame_update)

    def _process_frame_queue(self):
        """从队列取最新帧，更新画面并触发跟踪"""
        if self._shared_ctx is not None:
            # 集成模式：直接从共享上下文读取最新帧
            frame = self._shared_ctx.get_frame()
            if frame is None:
                return
            frame = frame.copy()
        else:
            # 独立模式：从队列取帧
            frame = None
            while not self._frame_queue.empty():
                try:
                    frame = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            if frame is None:
                return

        self.current_frame = frame

        # 集成模式：帧计数由此处驱动（独立模式由 scrcpy 回调驱动）
        if self._shared_ctx is not None:
            self._total_frame_count += 1

        # 集成模式：首帧自动 fit 到窗口，并根据 active_template_names 控制跟踪开关
        if self._shared_ctx is not None:
            if self._integrated_first_frame:
                self._integrated_first_frame = False
                self.root.after(50, lambda: self._redisplay(fit=True))

            has_active = bool(self._shared_ctx.active_template_names)
            if has_active and not self.is_tracking:
                self.is_tracking = True
                self.track_btn.configure(text="■ 停止跟踪")
                self._set_status("实时跟踪中（主流程驱动）…")
            elif not has_active and self.is_tracking:
                self.is_tracking = False
                self.track_btn.configure(text="▶ 开始跟踪")
                self._set_status("无活跃任务，跟踪已暂停")

        # FPS 统计
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_val = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = now
            self.fps_lbl.configure(text=f"FPS: {self._fps_val:.1f}")

        # 每 N 帧触发一次模板匹配（异步）
        interval = max(1, self.track_interval_var.get())
        if self.is_tracking and (self._total_frame_count % interval == 0):
            threading.Thread(
                target=self._run_matching_on_frame,
                args=(frame.copy(),),
                daemon=True
            ).start()

        self._redisplay()

    def _run_matching_on_frame(self, frame: np.ndarray):
        """后台线程：对帧执行模板匹配"""
        if not self._matching_lock.acquire(blocking=False):
            return  # 上次匹配还在进行，跳过本帧
        try:
            # 集成模式：根据主流程激活的模板名称更新 enabled 状态
            if self._shared_ctx is not None and self._shared_ctx.active_template_names:
                active = self._shared_ctx.active_template_names
                for t in self.templates:
                    t.enabled = t.name in active

            enabled = [t for t in self.templates if t.enabled]
            if not enabled:
                return

            method_id  = self.CV_METHODS.get(self.method_var.get(), cv2.TM_CCOEFF_NORMED)
            threshold  = self.threshold_var.get()
            user_scale = self.scale_var.get()
            results: Dict[str, MatchResult] = {}

            for tmpl in enabled:
                try:
                    # 优先从内存缓存取，避免每次从磁盘读取
                    tmpl_img = self._tmpl_cache.get(tmpl.path)
                    if tmpl_img is None:
                        raw = open(tmpl.path, "rb").read()
                        tmpl_img = cv2.imdecode(
                            np.frombuffer(raw, np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if tmpl_img is None:
                            continue
                        self._tmpl_cache[tmpl.path] = tmpl_img

                    th_s, tw_s = tmpl_img.shape[:2]
                    tw_new = max(1, int(tw_s * user_scale))
                    th_new = max(1, int(th_s * user_scale))
                    scaled = cv2.resize(tmpl_img, (tw_new, th_new))

                    if scaled.shape[0] > frame.shape[0] or scaled.shape[1] > frame.shape[1]:
                        continue

                    res_map = cv2.matchTemplate(frame, scaled, method_id)
                    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res_map)

                    if method_id == cv2.TM_SQDIFF_NORMED:
                        conf, loc = 1.0 - min_v, min_loc
                    else:
                        conf, loc = max_v, max_loc

                    cx = loc[0] + tw_new // 2
                    cy = loc[1] + th_new // 2
                    found = conf >= threshold

                    tmpl.last_confidence = conf
                    tmpl.last_x = cx
                    tmpl.last_y = cy
                    tmpl.last_found = found

                    results[tmpl.name] = MatchResult(
                        confidence=conf, x=cx, y=cy,
                        w=tw_new, h=th_new, found=found
                    )
                except Exception:
                    pass

            self.match_results = results

            # 集成模式：将识别结果同步回 SharedContext 供主流程读取
            if self._shared_ctx is not None:
                self._shared_ctx.set_match_results(results)

            self.root.after(0, self._refresh_results_tree)
            self.root.after(0, self._refresh_tmpl_list)

        finally:
            self._matching_lock.release()

    # ──────────────────────────────────────────
    # 画面渲染
    # ──────────────────────────────────────────

    def _draw_matches(self, img: np.ndarray) -> np.ndarray:
        for name, r in self.match_results.items():
            color = (0, 200, 80) if r.found else (80, 80, 220)
            x1 = max(0, r.x - r.w // 2)
            y1 = max(0, r.y - r.h // 2)
            x2 = min(img.shape[1], r.x + r.w // 2)
            y2 = min(img.shape[0], r.y + r.h // 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.drawMarker(img, (r.x, r.y), color, cv2.MARKER_CROSS, 18, 2)
            label = f"{name}  {r.confidence:.3f}"
            ly = max(y1 - 6, 14)
            cv2.putText(img, label, (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        return img

    def _redisplay(self, fit: bool = False):
        if self.current_frame is None:
            return

        img = self.current_frame.copy()
        if self.show_overlay_var.get() and self.match_results:
            img = self._draw_matches(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_w, img_h = pil_img.size

        if fit:
            cw = max(self.canvas.winfo_width(),  200)
            ch = max(self.canvas.winfo_height(), 200)
            self.display_scale = min(cw / img_w, ch / img_h)

        dw = max(1, int(img_w * self.display_scale))
        dh = max(1, int(img_h * self.display_scale))
        # NEAREST 速度最快，适合实时渲染
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
        """左键点击：通过 scrcpy control 发送触摸事件"""
        if self.current_frame is None:
            return
        ax = int(self.canvas.canvasx(event.x) / self.display_scale)
        ay = int(self.canvas.canvasy(event.y) / self.display_scale)
        self._last_canvas_coord = (ax, ay)

        # 集成模式优先走主流程 Device 的点击通道
        if self._shared_ctx is not None and self.external_click_handler is not None:
            try:
                ok = self.external_click_handler(ax, ay)
                if ok:
                    self._log(f"触摸: ({ax}, {ay})")
                else:
                    self._log(f"触摸失败: ({ax}, {ay})  (主流程未连接)")
            except Exception as e:
                self._log(f"触摸事件发送失败: {e}")
            return

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

    def _copy_coord_from_event(self, event):
        """右键复制当前鼠标位置坐标，格式: [x, y]"""
        if self.current_frame is None:
            return
        ax = int(self.canvas.canvasx(event.x) / self.display_scale)
        ay = int(self.canvas.canvasy(event.y) / self.display_scale)
        self._last_canvas_coord = (ax, ay)
        self.coord_lbl.configure(text=f"坐标: ({ax}, {ay})")
        self._copy_current_coord()

    def _copy_current_coord(self):
        """复制最近一次画布坐标到剪贴板，格式: [x, y]"""
        if self._last_canvas_coord is None:
            self._log("暂无可复制坐标，请先将鼠标移动到画面上")
            return
        ax, ay = self._last_canvas_coord
        text = f"[{ax}, {ay}]"
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
            self._log(f"已复制坐标: {text}")
        except Exception as e:
            self._log(f"复制坐标失败: {e}")

    # ──────────────────────────────────────────
    # 跟踪控制
    # ──────────────────────────────────────────

    def _toggle_tracking(self):
        if not self.is_connected:
            messagebox.showwarning("未连接", "请先连接设备")
            return
        if not self.templates:
            messagebox.showwarning("无模板", "请先添加模板图像")
            return

        self.is_tracking = not self.is_tracking
        if self.is_tracking:
            self.track_btn.configure(text="■ 停止跟踪")
            self._set_status("实时跟踪中…")
            self._log("开始实时图标跟踪")
        else:
            self.track_btn.configure(text="▶ 开始跟踪")
            self._set_status("跟踪已停止")
            self._log("停止图标跟踪")

    # ──────────────────────────────────────────
    # 模板管理
    # ──────────────────────────────────────────

    def _add_template(self):
        paths = filedialog.askopenfilenames(
            title="选择模板图像（可多选）",
            filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp *.webp"), ("全部", "*.*")]
        )
        added = 0
        self.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        for path in paths:
            src = Path(path)
            dst = self.TEMPLATES_DIR / src.name
            if any(t.path == str(dst) for t in self.templates):
                self._log(f"已存在，跳过: {src.name}")
                continue
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            self.templates.append(Template(
                name=dst.stem,
                path=str(dst),
                enabled=self._default_template_enabled(),
            ))
            added += 1
        if added:
            self._refresh_tmpl_list()
            self._log(f"添加了 {added} 个模板")

    def _paste_template(self):
        """从剪贴板粘贴图像或文件到模板列表（Ctrl+V）"""
        from PIL import ImageGrab
        try:
            cb = ImageGrab.grabclipboard()
        except Exception as e:
            self._log(f"读取剪贴板失败: {e}")
            return

        if cb is None:
            self._log("剪贴板为空或不含图像/文件")
            return

        if isinstance(cb, list):
            img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            added = 0
            self.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
            for p in cb:
                src = Path(p)
                if src.suffix.lower() not in img_exts or not src.exists():
                    continue
                dst = self.TEMPLATES_DIR / src.name
                if not any(t.path == str(dst) for t in self.templates):
                    if src.resolve() != dst.resolve():
                        shutil.copy2(src, dst)
                    self.templates.append(Template(
                        name=dst.stem,
                        path=str(dst),
                        enabled=self._default_template_enabled(),
                    ))
                    added += 1
            if added:
                self._refresh_tmpl_list()
                self._log(f"从文件列表粘贴了 {added} 个模板")
            else:
                self._log("剪贴板文件中未发现支持的图像格式")
            return

        if isinstance(cb, Image.Image):
            save_dir = self.TEMPLATES_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"paste_{ts}"
            name = simpledialog.askstring(
                "命名模板", "输入模板名称（留空使用时间戳）:",
                initialvalue=default_name, parent=self.root)
            if name is None:
                return
            name = name.strip() or default_name
            save_path = save_dir / f"{name}.png"
            counter = 1
            while save_path.exists():
                save_path = save_dir / f"{name}_{counter}.png"
                counter += 1
            cb.convert("RGBA").save(save_path, "PNG")
            self.templates.append(Template(
                name=save_path.stem,
                path=str(save_path),
                enabled=self._default_template_enabled(),
            ))
            self._refresh_tmpl_list()
            self._log(f"粘贴模板已保存: {save_path.name}")
            self._show_preview(self.templates[-1])
            return

        self._log(f"剪贴板内容无法识别为图像 (type={type(cb).__name__})")

    def _crop_from_frame(self):
        """在当前帧上框选区域，裁剪为模板"""
        if self.current_frame is None:
            messagebox.showwarning("无画面", "请先连接设备获取实时画面")
            return

        frame_copy = self.current_frame.copy()
        was_tracking = self.is_tracking
        self.is_tracking = False  # 暂停跟踪，避免画面更新干扰框选

        crop_win = tk.Toplevel(self.root)
        crop_win.title("框选模板区域  —  左键拖拽选择，松开确认")
        crop_win.grab_set()
        crop_win.resizable(False, False)

        h, w = frame_copy.shape[:2]
        max_w, max_h = 960, 720
        scale = min(max_w / w, max_h / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)

        canvas = tk.Canvas(crop_win, width=dw, height=dh, cursor="crosshair",
                           highlightthickness=0)
        canvas.pack()

        img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((dw, dh), Image.NEAREST)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas._photo_ref = photo  # 防 GC

        tip = ttk.Label(crop_win, text="左键拖拽框选区域，松开后输入名称保存为模板",
                        foreground="#555555")
        tip.pack(pady=(2, 4))

        state: dict = {"x0": 0, "y0": 0, "rect": None}

        def on_press(evt):
            state["x0"], state["y0"] = evt.x, evt.y
            if state["rect"]:
                canvas.delete(state["rect"])

        def on_drag(evt):
            if state["rect"]:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                state["x0"], state["y0"], evt.x, evt.y,
                outline="#00ff88", width=2, dash=(4, 2))

        def on_release(evt):
            x0, y0 = state["x0"], state["y0"]
            x1, y1 = evt.x, evt.y
            rx0 = int(min(x0, x1) / scale)
            ry0 = int(min(y0, y1) / scale)
            rx1 = int(max(x0, x1) / scale)
            ry1 = int(max(y0, y1) / scale)

            if rx1 - rx0 < 4 or ry1 - ry0 < 4:
                messagebox.showwarning("区域太小", "请框选更大的区域", parent=crop_win)
                return

            cropped = frame_copy[ry0:ry1, rx0:rx1]
            save_dir = self.TEMPLATES_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = simpledialog.askstring(
                "命名裁剪模板", "输入模板名称:",
                initialvalue=f"crop_{ts}", parent=crop_win)
            if name is None:
                crop_win.destroy()
                if was_tracking:
                    self.is_tracking = True
                return

            name = name.strip() or f"crop_{ts}"
            save_path = save_dir / f"{name}.png"
            cv2.imwrite(str(save_path), cropped)

            self.templates.append(Template(
                name=name,
                path=str(save_path),
                enabled=self._default_template_enabled(),
            ))
            self._refresh_tmpl_list()
            self._log(f"裁剪模板已保存: {save_path.name}  ({rx1-rx0}×{ry1-ry0}px)")
            self._show_preview(self.templates[-1])
            crop_win.destroy()
            if was_tracking:
                self.is_tracking = True

        canvas.bind("<ButtonPress-1>",   on_press)
        canvas.bind("<B1-Motion>",       on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        def on_close():
            crop_win.destroy()
            if was_tracking:
                self.is_tracking = True

        crop_win.protocol("WM_DELETE_WINDOW", on_close)

    def _remove_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        tmpl = self.templates[idx]
        if not messagebox.askyesno("确认删除", f"删除模板「{tmpl.name}」？\n（同时从磁盘删除图像文件）"):
            return
        try:
            Path(tmpl.path).unlink(missing_ok=True)
        except Exception as e:
            self._log(f"删除文件失败: {e}")
        self._tmpl_cache.pop(tmpl.path, None)
        self.templates.pop(idx)
        self._refresh_tmpl_list()
        self._log(f"已删除模板: {tmpl.name}")

    def _toggle_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        self.templates[idx].enabled = not self.templates[idx].enabled
        self._refresh_tmpl_list()

    def _rename_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        new_name = simpledialog.askstring(
            "重命名", "新名称:",
            initialvalue=self.templates[idx].name, parent=self.root)
        if new_name and new_name.strip():
            self.templates[idx].name = new_name.strip()
            self._refresh_tmpl_list()

    def _on_tmpl_select(self, _event):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        if 0 <= idx < len(self.templates):
            self._show_preview(self.templates[idx])

    def _show_preview(self, tmpl: Template):
        try:
            img = Image.open(tmpl.path)
            img.thumbnail((310, 160), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_lbl.configure(image=self._preview_photo, text="")
        except Exception as e:
            self.preview_lbl.configure(image="", text=f"无法加载: {e}")
            self._preview_photo = None

    def _refresh_tmpl_list(self):
        self.tmpl_tree.delete(*self.tmpl_tree.get_children())
        for t in self.templates:
            flag = "✓" if t.enabled else "✗"
            if t.last_found:
                conf = f"★{t.last_confidence:.3f}"
            elif t.last_confidence > 0:
                conf = f"{t.last_confidence:.3f}"
            else:
                conf = "—"
            tag = "found" if t.last_found else ("on" if t.enabled else "off")
            self.tmpl_tree.insert("", "end", values=(flag, t.name, conf), tags=(tag,))
        self.tmpl_tree.tag_configure("found", foreground="#007733")
        self.tmpl_tree.tag_configure("on",    foreground="#000000")
        self.tmpl_tree.tag_configure("off",   foreground="#888888")

    # ──────────────────────────────────────────
    # 跟踪结果表格
    # ──────────────────────────────────────────

    def _refresh_results_tree(self):
        self.res_tree.delete(*self.res_tree.get_children())
        for name, r in self.match_results.items():
            status = "✓ 找到" if r.found else "✗ 未找"
            self.res_tree.insert("", "end",
                                  values=(name, f"{r.confidence:.4f}", r.x, r.y, status),
                                  tags=("found" if r.found else "notfound",))

    # ──────────────────────────────────────────
    # 帧保存
    # ──────────────────────────────────────────

    def _save_frame(self):
        if self.current_frame is None:
            messagebox.showwarning("无画面", "暂无可保存的画面")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="保存当前帧",
            defaultextension=".png",
            initialfile=f"frame_{ts}.png",
            filetypes=[("PNG 图像", "*.png"), ("JPEG 图像", "*.jpg"), ("全部", "*.*")],
        )
        if path:
            cv2.imwrite(path, self.current_frame)
            self._log(f"帧已保存: {path}")

    # ──────────────────────────────────────────
    # 配置持久化
    # ──────────────────────────────────────────

    def _save_config(self):
        # 配置统一由 config.yaml 管理，模板通过文件系统持久化，无需额外保存
        pass

    def _load_config(self):
        """从 config.yaml 读取初始参数，并自动扫描 TEMPLATES_DIR 加载所有模板图像"""
        cfg = _load_project_config()
        emu_cfg     = cfg.get("emulator", {})
        tracker_cfg = cfg.get("scrcpy_tracker", {})

        # 设备连接参数
        device_id = emu_cfg.get("device_id", "")
        if device_id:
            self.device_var.set(device_id)
        self.max_width_var.set(emu_cfg.get("max_width", 800))
        self.max_fps_var.set(emu_cfg.get("max_fps", 30))

        # 跟踪参数
        threshold     = float(tracker_cfg.get("threshold", 0.80))
        method        = tracker_cfg.get("method", "TM_CCOEFF_NORMED")
        scale         = float(tracker_cfg.get("scale", 1.0))
        track_interval = int(tracker_cfg.get("track_interval", 3))

        self.threshold_var.set(threshold)
        self.thresh_lbl.configure(text=f"{threshold:.2f}")
        self.method_var.set(method)
        self.scale_var.set(scale)
        self.scale_lbl.configure(text=f"{scale:.2f}")
        self.track_interval_var.set(track_interval)

        # 自动从 TEMPLATES_DIR 扫描所有图像文件
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        loaded = 0
        if self.TEMPLATES_DIR.exists():
            for f in sorted(self.TEMPLATES_DIR.iterdir()):
                if f.suffix.lower() in img_exts:
                    if not any(t.path == str(f) for t in self.templates):
                        self.templates.append(Template(
                            name=f.stem,
                            path=str(f),
                            enabled=self._default_template_enabled(),
                        ))
                        loaded += 1

        self._refresh_tmpl_list()
        self._log(f"配置已加载，从 {self.TEMPLATES_DIR.name}/ 读取 {loaded} 个模板")

        # 集成模式：把参数同步到 SharedContext
        if self._shared_ctx is not None:
            self._shared_ctx.vision_threshold = threshold
            self._shared_ctx.vision_scale     = scale
            self._shared_ctx.vision_method    = method

    # ──────────────────────────────────────────
    # 参数变更回调（同步到 SharedContext）
    # ──────────────────────────────────────────

    def _on_threshold_change(self, v):
        val = float(v)
        self.thresh_lbl.config(text=f"{val:.2f}")
        if self._shared_ctx is not None:
            self._shared_ctx.vision_threshold = val

    def _on_scale_change(self, v):
        val = float(v)
        self.scale_lbl.config(text=f"{val:.2f}")
        # 清除模板缓存，避免使用旧的缩放图像
        if self._shared_ctx is not None:
            self._shared_ctx.vision_scale = val
        try:
            from core.vision import clear_cache
            clear_cache()
        except ImportError:
            pass

    def _on_method_change(self, *_):
        if self._shared_ctx is not None:
            self._shared_ctx.vision_method = self.method_var.get()

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────

    def _default_template_enabled(self) -> bool:
        """独立运行默认禁用模板，集成模式保持启用。"""
        return self._shared_ctx is not None

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
        if self._shared_ctx is None:
            self._disconnect()
        self.root.destroy()


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    root = tk.Tk()
    ScrcpyTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
