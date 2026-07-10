#!/usr/bin/env python3
"""
图标找图点击测试工具
使用 OpenCV 模板匹配，连接 ADB 模拟器截图并自动点击
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import json
import os
import threading
import time
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _load_project_config() -> dict:
    """尝试读取项目根目录的 config.yaml，返回其中 emulator 节"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists() or not _YAML_AVAILABLE:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("emulator", {})
    except Exception:
        return {}


def _cv_imread_compat(path: str, flags=cv2.IMREAD_COLOR):
    """
    兼容 Windows 中文路径的读图：
    优先使用 np.fromfile + cv2.imdecode，失败时再回退到 cv2.imread。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        return cv2.imread(path, flags)
    except Exception:
        return None


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

class IconClickerApp:
    CONFIG_FILE = Path(__file__).parent / "icon_clicker_config.json"

    CV_METHODS = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR_NORMED":  cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("图标找图点击测试工具  —  OpenCV Template Matching")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)

        # 从项目 config.yaml 读取默认 adb 配置
        proj_cfg = _load_project_config()
        default_adb_path = proj_cfg.get("adb_path", "adb")
        default_device   = proj_cfg.get("device_id", "")

        # 状态
        self.templates: list[Template] = []
        self.screenshot: Optional[np.ndarray] = None
        self.display_scale: float = 1.0
        self.actual_img_size: Tuple[int, int] = (0, 0)
        self.match_results: Dict[str, MatchResult] = {}
        self.devices: list[str] = []
        self.is_matching: bool = False
        self.batch_running: bool = False
        self._screenshot_photo = None   # 防止 GC 回收
        self._preview_photo = None

        # adb 路径（可在 UI 中修改）
        self._default_adb_path = default_adb_path
        self._default_device   = default_device

        self._build_ui()
        self._load_config()
        self._refresh_devices()
        # 全局 Ctrl+V 粘贴到模板（返回 break 防止事件冒泡导致重复触发）
        self.root.bind("<Control-v>", self._on_paste_shortcut)

    # ──────────────────────────────────────────
    # UI 构建
    # ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # 菜单栏
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="加载截图",         command=self._load_local_screenshot)
        file_menu.add_command(label="导出截图（原图）", command=self._export_raw)
        file_menu.add_command(label="导出截图（含标注）", command=self._export_annotated)
        file_menu.add_command(label="保存配置", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="使用说明", command=self._show_help)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

        # 主布局：左侧面板 + 右侧主区域
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main_pane, width=310)
        right = ttk.Frame(main_pane)
        main_pane.add(left, weight=0)
        main_pane.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

    # ---- 左侧面板 ----

    def _build_left(self, parent):
        parent.columnconfigure(0, weight=1)

        # 模板列表
        tmpl_lf = ttk.LabelFrame(parent, text="  模板图像管理  ", padding=6)
        tmpl_lf.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        parent.rowconfigure(0, weight=1)

        cols = ("on", "name", "conf")
        self.tmpl_tree = ttk.Treeview(tmpl_lf, columns=cols, show="headings", height=12)
        self.tmpl_tree.heading("on",   text="启")
        self.tmpl_tree.heading("name", text="模板名称")
        self.tmpl_tree.heading("conf", text="置信度")
        self.tmpl_tree.column("on",   width=28, anchor="center", stretch=False)
        self.tmpl_tree.column("name", width=140)
        self.tmpl_tree.column("conf", width=62, anchor="center", stretch=False)

        vsb = ttk.Scrollbar(tmpl_lf, orient=tk.VERTICAL, command=self.tmpl_tree.yview)
        self.tmpl_tree.configure(yscrollcommand=vsb.set)
        self.tmpl_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        tmpl_lf.rowconfigure(0, weight=1)
        tmpl_lf.columnconfigure(0, weight=1)

        self.tmpl_tree.bind("<<TreeviewSelect>>", self._on_tmpl_select)
        self.tmpl_tree.bind("<Double-1>", self._on_tmpl_dblclick)
        self.tmpl_tree.bind("<Control-v>", self._on_paste_shortcut)

        btn_row = ttk.Frame(tmpl_lf)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(btn_row, text="添加图像", command=self._add_template, width=9).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="粘贴 Ctrl+V", command=self._paste_template, width=11).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="删除",     command=self._remove_template, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="切换启用", command=self._toggle_template, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="重命名",   command=self._rename_template, width=7).pack(side=tk.LEFT, padx=2)

        # 模板预览
        prev_lf = ttk.LabelFrame(parent, text="  模板预览  ", padding=6)
        prev_lf.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        self.preview_lbl = ttk.Label(prev_lf, text="← 选择模板查看预览",
                                     anchor="center", relief="sunken",
                                     background="#1e1e1e", foreground="#aaaaaa")
        self.preview_lbl.pack(fill=tk.BOTH, ipady=30)

        # 匹配设置
        cfg_lf = ttk.LabelFrame(parent, text="  匹配参数  ", padding=6)
        cfg_lf.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        cfg_lf.columnconfigure(1, weight=1)

        # 匹配方法
        ttk.Label(cfg_lf, text="方法:").grid(row=0, column=0, sticky="w")
        self.method_var = tk.StringVar(value="TM_CCOEFF_NORMED")
        ttk.Combobox(cfg_lf, textvariable=self.method_var,
                     values=list(self.CV_METHODS.keys()),
                     state="readonly", width=20).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # 阈值
        ttk.Label(cfg_lf, text="阈值:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        thresh_row = ttk.Frame(cfg_lf)
        thresh_row.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.threshold_var = tk.DoubleVar(value=0.80)
        self.thresh_lbl = ttk.Label(thresh_row, text="0.80", width=5, anchor="e")
        self.thresh_lbl.pack(side=tk.RIGHT)
        ttk.Scale(thresh_row, from_=0.10, to=1.00, variable=self.threshold_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.thresh_lbl.config(text=f"{float(v):.2f}")
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 模板缩放
        ttk.Label(cfg_lf, text="缩放:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        scale_row = ttk.Frame(cfg_lf)
        scale_row.grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.scale_var = tk.DoubleVar(value=1.00)
        self.scale_lbl = ttk.Label(scale_row, text="1.00", width=5, anchor="e")
        self.scale_lbl.pack(side=tk.RIGHT)
        ttk.Scale(scale_row, from_=0.25, to=3.00, variable=self.scale_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.scale_lbl.config(text=f"{float(v):.2f}")
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 多尺度匹配
        self.multiscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg_lf, text="多尺度匹配 (±30%, 步长10%)",
                        variable=self.multiscale_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ADB 区域
        adb_lf = ttk.LabelFrame(parent, text="  ADB 控制  ", padding=6)
        adb_lf.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 4))
        adb_lf.columnconfigure(1, weight=1)

        # adb 可执行路径
        ttk.Label(adb_lf, text="adb:").grid(row=0, column=0, sticky="w")
        adb_path_row = ttk.Frame(adb_lf)
        adb_path_row.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.adb_path_var = tk.StringVar(value=getattr(self, "_default_adb_path", "adb"))
        adb_path_entry = ttk.Entry(adb_path_row, textvariable=self.adb_path_var)
        adb_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(adb_path_row, text="…", width=3,
                   command=self._browse_adb_path).pack(side=tk.LEFT, padx=(2, 0))

        # 设备
        ttk.Label(adb_lf, text="设备:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        dev_row = ttk.Frame(adb_lf)
        dev_row.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.device_var = tk.StringVar()
        self.device_cb = ttk.Combobox(dev_row, textvariable=self.device_var,
                                       state="readonly", width=18)
        self.device_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dev_row, text="刷新", width=5,
                   command=self._refresh_devices).pack(side=tk.LEFT, padx=(4, 0))

        # 自动点击
        opt_row = ttk.Frame(adb_lf)
        opt_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.auto_click_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_row, text="匹配后自动点击",
                        variable=self.auto_click_var).pack(side=tk.LEFT)
        ttk.Label(opt_row, text="延迟(s):").pack(side=tk.LEFT, padx=(8, 2))
        self.delay_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(opt_row, from_=0, to=5, increment=0.1,
                    textvariable=self.delay_var, width=5).pack(side=tk.LEFT)

        # 操作按钮
        act_row1 = ttk.Frame(adb_lf)
        act_row1.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(act_row1, text="截图", command=self._capture_screenshot, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(act_row1, text="开始匹配", command=self._run_matching, width=9).pack(side=tk.LEFT, padx=2)
        ttk.Button(act_row1, text="截图+匹配", command=self._capture_and_match, width=10).pack(side=tk.LEFT, padx=2)

        act_row2 = ttk.Frame(adb_lf)
        act_row2.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.click_sel_btn = ttk.Button(act_row2, text="点击选中结果",
                                         command=self._click_selected, width=12, state=tk.DISABLED)
        self.click_sel_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(act_row2, text="点击全部找到", command=self._click_all_found, width=12).pack(side=tk.LEFT, padx=2)

        act_row3 = ttk.Frame(adb_lf)
        act_row3.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.batch_btn = ttk.Button(act_row3, text="▶ 批量循环测试",
                                     command=self._toggle_batch, width=14)
        self.batch_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(act_row3, text="间隔(s):").pack(side=tk.LEFT, padx=(8, 2))
        self.batch_interval_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(act_row3, from_=0.5, to=30, increment=0.5,
                    textvariable=self.batch_interval_var, width=5).pack(side=tk.LEFT)

    # ---- 右侧主区域 ----

    def _build_right(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # 截图预览
        ss_lf = ttk.LabelFrame(parent, text="  截图预览  —  鼠标悬停显示坐标，左键单击可手动触发点击  ", padding=4)
        ss_lf.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ss_lf.rowconfigure(0, weight=1)
        ss_lf.columnconfigure(0, weight=1)

        canvas_frame = ttk.Frame(ss_lf)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#1a1a2e", cursor="crosshair",
                                highlightthickness=0)
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,   command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        hbar.grid(row=1, column=0, sticky="ew")
        vbar.grid(row=0, column=1, sticky="ns")

        self.canvas.bind("<Button-1>",  self._on_canvas_click)
        self.canvas.bind("<Motion>",    self._on_canvas_motion)
        self.canvas.bind("<Configure>", lambda e: self._redisplay())

        # 工具栏
        toolbar = ttk.Frame(ss_lf)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.coord_lbl = ttk.Label(toolbar, text="坐标: —", width=18, anchor="w")
        self.coord_lbl.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="加载本地截图",   command=self._load_local_screenshot).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="导出截图▾",     command=self._show_export_menu).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="原始大小",       command=self._zoom_actual).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="适合窗口",       command=lambda: self._redisplay(fit=True)).pack(side=tk.RIGHT, padx=2)

        # 下方：结果 + 日志
        bottom = ttk.Frame(parent)
        bottom.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        # 匹配结果表格
        res_lf = ttk.LabelFrame(bottom, text="  匹配结果  ", padding=4)
        res_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        res_lf.columnconfigure(0, weight=1)

        rcols = ("template", "confidence", "x", "y", "status")
        self.res_tree = ttk.Treeview(res_lf, columns=rcols, show="headings", height=5)
        for col, hdr, w, anchor in [
            ("template",   "模板",   110, "w"),
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
        self.res_tree.tag_configure("found",    foreground="#00c853")
        self.res_tree.tag_configure("notfound", foreground="#ff5252")
        self.res_tree.bind("<<TreeviewSelect>>", self._on_result_select)

        res_btn_row = ttk.Frame(res_lf)
        res_btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(res_btn_row, text="清空结果", command=self._clear_results, width=8).pack(side=tk.RIGHT)

        # 操作日志
        log_lf = ttk.LabelFrame(bottom, text="  操作日志  ", padding=4)
        log_lf.grid(row=0, column=1, sticky="nsew", padx=(3, 0))

        self.log_txt = scrolledtext.ScrolledText(
            log_lf, height=6, state=tk.DISABLED,
            font=("Consolas", 9), bg="#0d0d1a", fg="#cccccc",
            insertbackground="white")
        self.log_txt.pack(fill=tk.BOTH, expand=True)

        log_btn = ttk.Frame(log_lf)
        log_btn.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(log_btn, text="清空日志", command=self._clear_log, width=8).pack(side=tk.RIGHT)

        # 状态栏
        statusbar = ttk.Frame(parent)
        statusbar.grid(row=2, column=0, sticky="ew", padx=4)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(statusbar, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(statusbar, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT)

    # ──────────────────────────────────────────
    # 模板管理
    # ──────────────────────────────────────────

    def _add_template(self):
        paths = filedialog.askopenfilenames(
            title="选择模板图像（可多选）",
            filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp *.webp"), ("全部", "*.*")]
        )
        added = 0
        for path in paths:
            if any(t.path == path for t in self.templates):
                self._log(f"已存在，跳过: {Path(path).name}")
                continue
            self.templates.append(Template(name=Path(path).stem, path=path))
            added += 1
        if added:
            self._refresh_tmpl_list()
            self._save_config()
            self._log(f"添加了 {added} 个模板")

    def _paste_template(self):
        """从剪贴板粘贴图像或文件路径到模板列表（支持 Ctrl+V）"""
        from PIL import ImageGrab
        try:
            cb = ImageGrab.grabclipboard()
        except Exception as e:
            self._log(f"读取剪贴板失败: {e}")
            return

        if cb is None:
            self._log("剪贴板为空或不包含图像/文件")
            return

        # ── 情况 1：剪贴板里是文件列表（如资源管理器复制的文件）
        if isinstance(cb, list):
            img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            added = 0
            for p in cb:
                p = Path(p)
                if p.suffix.lower() in img_exts and p.exists():
                    if any(t.path == str(p) for t in self.templates):
                        self._log(f"已存在，跳过: {p.name}")
                        continue
                    self.templates.append(Template(name=p.stem, path=str(p)))
                    added += 1
            if added:
                self._refresh_tmpl_list()
                self._save_config()
                self._log(f"从文件列表粘贴了 {added} 个模板")
            else:
                self._log("剪贴板文件列表中未发现支持的图像格式")
            return

        # ── 情况 2：剪贴板里是图像数据（截图、QQ/微信粘贴等）
        if isinstance(cb, Image.Image):
            # 保存到工具目录下的 pasted_templates 子文件夹
            save_dir = Path(__file__).parent / "pasted_templates"
            save_dir.mkdir(exist_ok=True)

            # 自动生成默认名称
            ts = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"paste_{ts}"

            name = simpledialog.askstring(
                "为粘贴的模板命名",
                "输入模板名称（留空使用时间戳）:",
                initialvalue=default_name,
                parent=self.root,
            )
            if name is None:        # 用户点了取消
                return
            name = name.strip() or default_name

            save_path = save_dir / f"{name}.png"
            # 若同名文件已存在，加序号避免覆盖
            counter = 1
            while save_path.exists():
                save_path = save_dir / f"{name}_{counter}.png"
                counter += 1

            # 保证是 RGB 再保存（避免 RGBA 写入 PNG 时出现透明通道问题）
            img_to_save = cb.convert("RGBA") if cb.mode in ("P", "LA") else cb
            img_to_save.save(save_path, "PNG")

            if any(t.path == str(save_path) for t in self.templates):
                self._log(f"已存在，跳过: {save_path.name}")
                return

            self.templates.append(Template(name=save_path.stem, path=str(save_path)))
            self._refresh_tmpl_list()
            self._save_config()
            self._log(f"粘贴模板已保存: {save_path}")

            # 自动显示预览
            self._show_preview(self.templates[-1])
            return

        self._log(f"剪贴板内容无法识别为图像 (type={type(cb).__name__})")

    def _on_paste_shortcut(self, _event=None):
        """统一处理 Ctrl+V，阻止同一按键事件冒泡造成重复粘贴。"""
        self._paste_template()
        return "break"

    def _remove_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        name = self.templates[idx].name
        if messagebox.askyesno("确认", f"删除模板「{name}」？"):
            self.templates.pop(idx)
            self._refresh_tmpl_list()
            self._save_config()
            self._log(f"已删除: {name}")

    def _toggle_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        self.templates[idx].enabled = not self.templates[idx].enabled
        self._refresh_tmpl_list()
        self._save_config()

    def _rename_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        new_name = simpledialog.askstring("重命名", "新名称:",
                                           initialvalue=self.templates[idx].name,
                                           parent=self.root)
        if new_name and new_name.strip():
            self.templates[idx].name = new_name.strip()
            self._refresh_tmpl_list()
            self._save_config()

    def _on_tmpl_select(self, _event):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        idx = self.tmpl_tree.index(sel[0])
        if 0 <= idx < len(self.templates):
            self._show_preview(self.templates[idx])

    def _on_tmpl_dblclick(self, _event):
        self._rename_template()

    def _show_preview(self, tmpl: Template):
        try:
            img = Image.open(tmpl.path)
            img.thumbnail((290, 160), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_lbl.configure(image=self._preview_photo, text="")
        except Exception as e:
            self.preview_lbl.configure(image="", text=f"无法加载: {e}")
            self._preview_photo = None

    def _refresh_tmpl_list(self):
        self.tmpl_tree.delete(*self.tmpl_tree.get_children())
        for t in self.templates:
            flag = "✓" if t.enabled else "✗"
            conf = f"{t.last_confidence:.3f}" if t.last_confidence > 0 else "—"
            tag  = "on" if t.enabled else "off"
            self.tmpl_tree.insert("", "end", values=(flag, t.name, conf), tags=(tag,))
        self.tmpl_tree.tag_configure("on",  foreground="#000000")
        self.tmpl_tree.tag_configure("off", foreground="#888888")

    # ──────────────────────────────────────────
    # ADB 操作
    # ──────────────────────────────────────────

    @property
    def _adb_exe(self) -> str:
        """返回当前界面设置的 adb 可执行路径"""
        return self.adb_path_var.get().strip() or "adb"

    def _adb(self, args: list[str], timeout=15) -> subprocess.CompletedProcess:
        return subprocess.run([self._adb_exe] + args, capture_output=True, timeout=timeout)

    def _browse_adb_path(self):
        path = filedialog.askopenfilename(
            title="选择 adb.exe",
            filetypes=[("adb 可执行文件", "adb.exe adb"), ("全部", "*.*")],
            initialdir=str(Path(self._adb_exe).parent) if Path(self._adb_exe).exists() else "C:\\"
        )
        if path:
            self.adb_path_var.set(path)
            self._save_config()
            self._log(f"ADB 路径已更新: {path}")

    def _refresh_devices(self):
        def _work():
            try:
                r = self._adb(["devices"])
                lines = r.stdout.decode("utf-8", errors="ignore").splitlines()
                devs = [l.split("\t")[0] for l in lines[1:] if "\tdevice" in l]
                self.devices = devs
                self.root.after(0, self._update_device_list)
                if devs:
                    self._log(f"ADB({self._adb_exe}) 设备: {', '.join(devs)}")
                else:
                    self._log(f"未发现 ADB 设备  [adb={self._adb_exe}]")
            except FileNotFoundError:
                self._log(f"找不到 adb: {self._adb_exe}  请检查路径或修改 config.yaml")
            except Exception as e:
                self._log(f"刷新设备失败: {e}")
        threading.Thread(target=_work, daemon=True).start()

    def _update_device_list(self):
        if self.devices:
            self.device_cb["values"] = self.devices
            if self.device_var.get() not in self.devices:
                self.device_var.set(self.devices[0])
        else:
            self.device_cb["values"] = ["(无设备)"]
            self.device_var.set("(无设备)")

    def _current_device(self) -> Optional[str]:
        dev = self.device_var.get()
        if not dev or dev == "(无设备)":
            messagebox.showwarning("无设备", "请先连接 ADB 设备并刷新")
            return None
        return dev

    def _capture_screenshot(self, callback=None):
        device = self._current_device()
        if not device:
            return
        self._set_status("正在截图…")
        self.progress.start()

        def _work():
            try:
                r = self._adb(["-s", device, "exec-out", "screencap", "-p"], timeout=20)
                if r.returncode != 0:
                    self._log(f"截图失败: {r.stderr.decode(errors='ignore')}")
                    return
                img = Image.open(io.BytesIO(r.stdout))
                self.screenshot = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                h, w = self.screenshot.shape[:2]
                self._log(f"截图成功 {w}×{h}")
                self.root.after(0, lambda: self._redisplay(fit=True))
                if callback:
                    self.root.after(100, callback)
            except Exception as e:
                self._log(f"截图异常: {e}")
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self._set_status("就绪"))

        threading.Thread(target=_work, daemon=True).start()

    def _capture_and_match(self):
        self._capture_screenshot(callback=self._run_matching)

    def _click_adb(self, device: str, x: int, y: int):
        try:
            self._adb(["-s", device, "shell", "input", "tap", str(x), str(y)], timeout=6)
            self._log(f"已点击 ({x}, {y})")
        except Exception as e:
            self._log(f"点击失败: {e}")

    def _click_selected(self):
        sel = self.res_tree.selection()
        if not sel:
            return
        vals = self.res_tree.item(sel[0])["values"]
        if not vals or len(vals) < 4:
            return
        device = self._current_device()
        if device:
            self._click_adb(device, int(vals[2]), int(vals[3]))

    def _click_all_found(self):
        device = self._current_device()
        if not device:
            return
        clicked = 0
        for r in self.match_results.values():
            if r.found:
                self._click_adb(device, r.x, r.y)
                clicked += 1
                time.sleep(0.15)
        self._log(f"共点击 {clicked} 个目标" if clicked else "没有找到可点击的目标")

    # ──────────────────────────────────────────
    # 模板匹配
    # ──────────────────────────────────────────

    def _run_matching(self):
        if self.screenshot is None:
            messagebox.showwarning("无截图", "请先截图或加载本地截图")
            return
        enabled = [t for t in self.templates if t.enabled]
        if not enabled:
            messagebox.showwarning("无模板", "请先添加并启用模板")
            return
        if self.is_matching:
            return

        self.is_matching = True
        self._set_status("正在匹配…")
        self.progress.start()

        def _work():
            results: Dict[str, MatchResult] = {}
            method_id  = self.CV_METHODS.get(self.method_var.get(), cv2.TM_CCOEFF_NORMED)
            threshold  = self.threshold_var.get()
            user_scale = self.scale_var.get()
            multiscale = self.multiscale_var.get()

            src = self.screenshot.copy()

            for tmpl in enabled:
                try:
                    tmpl_img = _cv_imread_compat(tmpl.path)
                    if tmpl_img is None:
                        self._log(f"[{tmpl.name}] 无法读取图像文件")
                        continue

                    best_conf, best_loc, best_scale = 0.0, (0, 0), user_scale

                    # 构造搜索的缩放列表
                    if multiscale:
                        scales = [round(user_scale * s, 2)
                                  for s in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
                    else:
                        scales = [user_scale]

                    for sc in scales:
                        th_s, tw_s = tmpl_img.shape[:2]
                        tw_new = max(1, int(tw_s * sc))
                        th_new = max(1, int(th_s * sc))
                        scaled_tmpl = cv2.resize(tmpl_img, (tw_new, th_new))

                        if scaled_tmpl.shape[0] > src.shape[0] or scaled_tmpl.shape[1] > src.shape[1]:
                            continue

                        res_map = cv2.matchTemplate(src, scaled_tmpl, method_id)
                        min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res_map)

                        if method_id == cv2.TM_SQDIFF_NORMED:
                            conf = 1.0 - min_v
                            loc  = min_loc
                        else:
                            conf = max_v
                            loc  = max_loc

                        if conf > best_conf:
                            best_conf  = conf
                            best_loc   = loc
                            best_scale = sc

                    # 换算实际坐标（以原始截图坐标系）
                    tmpl_h, tmpl_w = tmpl_img.shape[:2]
                    tw_final = max(1, int(tmpl_w * best_scale))
                    th_final = max(1, int(tmpl_h * best_scale))
                    cx = best_loc[0] + tw_final // 2
                    cy = best_loc[1] + th_final // 2

                    found = best_conf >= threshold
                    tmpl.last_confidence = best_conf
                    tmpl.last_x = cx
                    tmpl.last_y = cy

                    results[tmpl.name] = MatchResult(
                        confidence=best_conf, x=cx, y=cy,
                        w=tw_final, h=th_final, found=found
                    )
                    status_str = "✓ 找到" if found else "✗ 未找"
                    scale_info = f" (scale={best_scale:.2f})" if multiscale else ""
                    self._log(
                        f"[{tmpl.name}] 置信度 {best_conf:.4f}  坐标 ({cx}, {cy}){scale_info}  {status_str}"
                    )
                except Exception as e:
                    self._log(f"[{tmpl.name}] 匹配异常: {e}")

            self.match_results = results
            found_n = sum(1 for r in results.values() if r.found)
            self.root.after(0, self._refresh_tmpl_list)
            self.root.after(0, self._refresh_results_tree)
            self.root.after(0, lambda: self._redisplay(fit=False))
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self._set_status(
                f"匹配完成：{found_n}/{len(results)} 个找到"))
            self.is_matching = False

            if self.auto_click_var.get() and found_n > 0:
                delay_ms = int(self.delay_var.get() * 1000)
                self.root.after(delay_ms, self._click_all_found)

        threading.Thread(target=_work, daemon=True).start()

    # ──────────────────────────────────────────
    # 批量循环测试
    # ──────────────────────────────────────────

    def _toggle_batch(self):
        if self.batch_running:
            self.batch_running = False
            self.batch_btn.configure(text="▶ 批量循环测试")
            self._log("批量测试已停止")
        else:
            self.batch_running = True
            self.batch_btn.configure(text="■ 停止循环测试")
            self._log("开始批量循环测试…")
            self._batch_step()

    def _batch_step(self):
        if not self.batch_running:
            return

        def after_screenshot():
            self._run_matching()
            # 等待匹配完成后再排队下次截图
            interval_ms = int(self.batch_interval_var.get() * 1000)
            self.root.after(interval_ms + 500, self._batch_step)

        self._capture_screenshot(callback=after_screenshot)

    # ──────────────────────────────────────────
    # 截图显示
    # ──────────────────────────────────────────

    def _draw_matches(self, img: np.ndarray) -> np.ndarray:
        """在图像副本上绘制匹配框和标签"""
        for name, r in self.match_results.items():
            color = (0, 200, 60) if r.found else (60, 100, 255)
            x1 = max(0, r.x - r.w // 2)
            y1 = max(0, r.y - r.h // 2)
            x2 = min(img.shape[1], r.x + r.w // 2)
            y2 = min(img.shape[0], r.y + r.h // 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.drawMarker(img, (r.x, r.y), color, cv2.MARKER_CROSS, 20, 2)
            label = f"{name}  {r.confidence:.3f}"
            ly = max(y1 - 6, 14)
            # 黑色背景文字
            cv2.putText(img, label, (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return img

    def _redisplay(self, fit: bool = False):
        if self.screenshot is None:
            return
        img = self._draw_matches(self.screenshot.copy())
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_w, img_h = pil_img.size
        self.actual_img_size = (img_w, img_h)

        if fit:
            cw = max(self.canvas.winfo_width(), 200)
            ch = max(self.canvas.winfo_height(), 200)
            self.display_scale = min(cw / img_w, ch / img_h)

        dw = max(1, int(img_w * self.display_scale))
        dh = max(1, int(img_h * self.display_scale))
        pil_img = pil_img.resize((dw, dh), Image.LANCZOS)
        self._screenshot_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._screenshot_photo)
        self.canvas.configure(scrollregion=(0, 0, dw, dh))

    def _zoom_actual(self):
        self.display_scale = 1.0
        self._redisplay(fit=False)

    def _on_canvas_click(self, event):
        if self.screenshot is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ax = int(cx / self.display_scale)
        ay = int(cy / self.display_scale)
        device = self.device_var.get()
        if device and device != "(无设备)":
            if messagebox.askyesno("手动点击", f"在设备上点击坐标 ({ax}, {ay})？"):
                threading.Thread(
                    target=lambda: self._click_adb(device, ax, ay),
                    daemon=True
                ).start()

    def _on_canvas_motion(self, event):
        if self.screenshot is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ax = int(cx / self.display_scale)
        ay = int(cy / self.display_scale)
        self.coord_lbl.configure(text=f"坐标: ({ax}, {ay})")

    # ──────────────────────────────────────────
    # 结果表格
    # ──────────────────────────────────────────

    def _clear_results(self):
        self.match_results = {}
        for t in self.templates:
            t.last_confidence = 0.0
            t.last_x = 0
            t.last_y = 0
        self.res_tree.delete(*self.res_tree.get_children())
        self.click_sel_btn.configure(state=tk.DISABLED)
        self._refresh_tmpl_list()
        self._redisplay(fit=False)
        self._set_status("结果已清空")
        self._log("匹配结果已清空")

    def _refresh_results_tree(self):
        self.res_tree.delete(*self.res_tree.get_children())
        for name, r in self.match_results.items():
            status = "✓ 找到" if r.found else "✗ 未找"
            self.res_tree.insert("", "end",
                                  values=(name, f"{r.confidence:.4f}", r.x, r.y, status),
                                  tags=("found" if r.found else "notfound",))

    def _on_result_select(self, _event):
        sel = self.res_tree.selection()
        self.click_sel_btn.configure(state=tk.NORMAL if sel else tk.DISABLED)

    # ──────────────────────────────────────────
    # 本地截图加载
    # ──────────────────────────────────────────

    def _load_local_screenshot(self):
        path = filedialog.askopenfilename(
            title="选择截图文件",
            filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp"), ("全部", "*.*")]
        )
        if not path:
            return
        try:
            self.screenshot = _cv_imread_compat(path)
            if self.screenshot is None:
                raise ValueError("读取截图失败（文件不存在、损坏或路径包含特殊字符）")
            self.match_results = {}
            self._redisplay(fit=True)
            self._log(f"加载截图: {path}")
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    # ──────────────────────────────────────────
    # 截图导出
    # ──────────────────────────────────────────

    def _show_export_menu(self):
        """在「导出截图▾」按钮下方弹出下拉菜单"""
        if self.screenshot is None:
            messagebox.showwarning("无截图", "请先截图或加载本地截图")
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="导出原始截图",     command=self._export_raw)
        menu.add_command(label="导出含标注截图",   command=self._export_annotated)
        menu.add_separator()
        menu.add_command(label="复制原图到剪贴板",   command=self._copy_raw_to_clipboard)
        menu.add_command(label="复制标注图到剪贴板", command=self._copy_annotated_to_clipboard)
        # 定位到按钮右下角弹出
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def _export_raw(self):
        """导出无标注的原始截图"""
        self._export_image(self.screenshot, suffix="raw")

    def _export_annotated(self):
        """导出带匹配框标注的截图"""
        if self.screenshot is None:
            messagebox.showwarning("无截图", "请先截图或加载本地截图")
            return
        annotated = self._draw_matches(self.screenshot.copy())
        self._export_image(annotated, suffix="annotated")

    def _export_image(self, img_bgr: np.ndarray, suffix: str = ""):
        if img_bgr is None:
            messagebox.showwarning("无截图", "请先截图或加载本地截图")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"screenshot_{ts}_{suffix}.png" if suffix else f"screenshot_{ts}.png"
        path = filedialog.asksaveasfilename(
            title="导出截图",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG 图像",  "*.png"),
                ("JPEG 图像", "*.jpg"),
                ("BMP 图像",  "*.bmp"),
                ("全部",      "*.*"),
            ],
        )
        if not path:
            return
        try:
            cv2.imwrite(path, img_bgr)
            self._log(f"截图已导出: {path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _copy_raw_to_clipboard(self):
        self._copy_image_to_clipboard(self.screenshot)

    def _copy_annotated_to_clipboard(self):
        if self.screenshot is None:
            return
        self._copy_image_to_clipboard(self._draw_matches(self.screenshot.copy()))

    def _copy_image_to_clipboard(self, img_bgr: np.ndarray):
        """将图像复制到系统剪贴板（Windows）"""
        if img_bgr is None:
            return
        try:
            from PIL import ImageGrab
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            # 借助 io 把图存成 BMP 再写入剪贴板
            import io as _io
            import ctypes
            buf = _io.BytesIO()
            pil_img.save(buf, format="BMP")
            bmp_data = buf.getvalue()[14:]          # 去掉 BITMAPFILEHEADER
            CF_DIB = 8
            ctypes.windll.user32.OpenClipboard(0)
            ctypes.windll.user32.EmptyClipboard()
            hMem = ctypes.windll.kernel32.GlobalAlloc(0x0042, len(bmp_data))
            ptr  = ctypes.windll.kernel32.GlobalLock(hMem)
            ctypes.memmove(ptr, bmp_data, len(bmp_data))
            ctypes.windll.kernel32.GlobalUnlock(hMem)
            ctypes.windll.user32.SetClipboardData(CF_DIB, hMem)
            ctypes.windll.user32.CloseClipboard()
            self._log("截图已复制到剪贴板")
        except Exception as e:
            self._log(f"复制到剪贴板失败: {e}")

    # ──────────────────────────────────────────
    # 配置持久化
    # ──────────────────────────────────────────

    def _save_config(self):
        cfg = {
            "templates": [
                {"name": t.name, "path": t.path, "enabled": t.enabled}
                for t in self.templates
            ],
            "adb_path":       self.adb_path_var.get(),
            "threshold":      self.threshold_var.get(),
            "method":         self.method_var.get(),
            "device":         self.device_var.get(),
            "scale":          self.scale_var.get(),
            "auto_click":     self.auto_click_var.get(),
            "delay":          self.delay_var.get(),
            "multiscale":     self.multiscale_var.get(),
            "batch_interval": self.batch_interval_var.get(),
        }
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"保存配置失败: {e}")

    def _load_config(self):
        if not self.CONFIG_FILE.exists():
            return
        try:
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.templates = []
            for t in cfg.get("templates", []):
                p = t.get("path", "")
                if os.path.exists(p):
                    self.templates.append(
                        Template(name=t.get("name", Path(p).stem),
                                 path=p, enabled=t.get("enabled", True))
                    )
            # adb 路径：优先用上次保存的值，其次才是 config.yaml 的默认值
            saved_adb = cfg.get("adb_path", "")
            self.adb_path_var.set(saved_adb if saved_adb else self._default_adb_path)

            self.threshold_var.set(cfg.get("threshold", 0.80))
            self.thresh_lbl.configure(text=f"{cfg.get('threshold', 0.80):.2f}")
            self.method_var.set(cfg.get("method", "TM_CCOEFF_NORMED"))
            self.scale_var.set(cfg.get("scale", 1.00))
            self.scale_lbl.configure(text=f"{cfg.get('scale', 1.00):.2f}")
            self.auto_click_var.set(cfg.get("auto_click", False))
            self.delay_var.set(cfg.get("delay", 0.3))
            self.multiscale_var.set(cfg.get("multiscale", False))
            self.batch_interval_var.set(cfg.get("batch_interval", 2.0))
            self._refresh_tmpl_list()
            self._log(f"配置已加载，共 {len(self.templates)} 个模板")
        except Exception as e:
            self._log(f"加载配置失败: {e}")

    # ──────────────────────────────────────────
    # 帮助
    # ──────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title("使用说明")
        win.geometry("560x480")
        win.resizable(False, False)
        text = scrolledtext.ScrolledText(win, font=("微软雅黑", 10), wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, """\
图标找图点击测试工具 — 使用说明
================================

【基本流程】
1. 在「模板图像管理」面板点击「添加图像」，选择要查找的图标截图
   — 或者直接截图后按 Ctrl+V（或点击「粘贴」按钮）粘贴到模板列表
2. 在「ADB 控制」区选择已连接的模拟器设备（点击「刷新」获取设备列表）
3. 点击「截图」从模拟器抓取当前画面，或点击「加载本地截图」使用本地文件
4. 点击「开始匹配」执行模板匹配，结果显示在右下角表格中
5. 匹配成功的项目（绿色）可点击「点击全部找到」自动发送 ADB 点击命令

【快捷操作】
• 「截图+匹配」：一键完成截图和匹配
• 「批量循环测试」：自动循环截图→匹配→（可选点击），用于测试稳定性
• 截图区域左键点击：直接发送对应坐标的 ADB tap 命令

【匹配参数说明】
• 方法：
  - TM_CCOEFF_NORMED（推荐）：归一化相关系数，对亮度变化鲁棒
  - TM_CCORR_NORMED：归一化互相关，速度略快
  - TM_SQDIFF_NORMED：差值平方和，置信度逻辑相反（越低越匹配）
• 阈值：0.80 以上通常认为可靠匹配，可根据实际情况调整
• 缩放：如模板尺寸与截图不一致（如不同 DPI 屏幕），调整此值
• 多尺度匹配：在 ±30% 范围内自动寻找最佳缩放，但速度较慢

【测试 OpenCV 可靠性建议】
• 对比不同「匹配方法」在同一场景下的置信度
• 调整「阈值」观察误检率和漏检率的变化
• 开启「多尺度匹配」测试尺度不变性
• 使用「批量循环测试」观察多次结果的稳定性（置信度波动范围）

【ADB 环境要求】
• adb 路径会自动从项目根目录 config.yaml 的 emulator.adb_path 读取
  （如 D:/leidian/LDPlayer9/adb.exe），也可在界面「adb:」栏手动修改
• 点击「…」按钮可通过文件选择器定位 adb.exe
• 修改后点击「刷新」重新扫描设备
""")
        text.configure(state=tk.DISABLED)

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
        self.batch_running = False
        self._save_config()
        self.root.destroy()


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = IconClickerApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
