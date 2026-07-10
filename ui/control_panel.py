"""
ControlPanel：主控制面板 UI。
- 设备连接管理（连接/断开）
- 任务开关（抢红包开启/停止）
- 实时状态展示（当前任务、Tick 计数、设备状态）
- 日志输出
"""
from __future__ import annotations

import logging
import threading
import time
import ast
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext, messagebox
from typing import Callable, Optional

from core.device import Device
from core.shared_state import SharedContext
from engine.scheduler import Scheduler
from logic.common_steps import HomeNavigator, try_go_back
from logic.tasks.alliance_mobilization import AllianceMobilizationTask
from logic.tasks.alliance_monitor import AllianceMonitorTask
from logic.tasks.bear_attack import BearAttackTask
from logic.tasks.bear_attack_debug import BearAttackDebugTask
from logic.tasks.giant_beast_rally import GiantBeastRallyTask
from logic.tasks.red_packet import RedPacketTask

logger = logging.getLogger(__name__)


class ControlPanel:
    """主控制面板，嵌入到主 Tk 窗口中"""

    REFRESH_MS = 500  # 状态刷新间隔
    # 启动时自动连接 scrcpy：首次延迟后尝试，失败后每隔 interval 再试，共 max 次
    AUTO_CONNECT_INITIAL_MS = 1_000
    AUTO_CONNECT_INTERVAL_MS = 10_000
    AUTO_CONNECT_MAX_ATTEMPTS = 5
    # 无尽冬日自动启动：竖屏约 864×1920 视为已在游戏内，横屏约 1920×864 视为系统桌面
    GAME_AUTOLAUNCH_SIZE_TOLERANCE = 48
    GAME_LAUNCH_TEMPLATE_NAME = "无尽冬日"
    GAME_LAUNCH_RETRY_INTERVAL_SEC = 5.0
    GAME_LAUNCH_MAX_RETRIES = 5
    GAME_LAUNCH_POST_CLICK_WAIT_SEC = 10.0
    GAME_LAUNCH_FRAME_WAIT_SEC = 45.0

    def __init__(
        self,
        root: tk.Tk,
        scheduler: Scheduler,
        device: Device,
        ctx: SharedContext,
        config: dict,
    ) -> None:
        self._root = root
        self._scheduler = scheduler
        self._device = device
        self._ctx = ctx
        self._config = config

        self._red_packet_task: Optional[RedPacketTask] = None
        self._bear_attack_task: Optional[BearAttackTask] = None
        self._alliance_mob_task: Optional[AllianceMobilizationTask] = None
        self._exc_monitor_task: Optional[AllianceMonitorTask] = None
        self._pub_monitor_task: Optional[AllianceMonitorTask] = None
        self._giant_beast_rally_task: Optional[GiantBeastRallyTask] = None
        self._bear_debug_task: Optional[BearAttackDebugTask] = None
        self._feishu_notify: Optional[Callable[[str], None]] = None
        self._manual_nav_lock = threading.Lock()
        self._event_listeners: list[Callable[[str], None]] = []
        self._bear_team_template_options: list[str] = []
        self._team_checkbox_vars: dict[int, dict[str, tk.BooleanVar]] = {}
        self._ba_cd_labels: dict[int, ttk.Label] = {}
        self._ba_delay_group_var: Optional[tk.DoubleVar] = None
        self._ba_delay_expedition_var: Optional[tk.DoubleVar] = None
        # 抢红包 UI 状态变量
        self._rp_mode_var: Optional[tk.StringVar] = None
        self._rp_scan_interval_var: Optional[tk.StringVar] = None
        self._rp_click_delay_var: Optional[tk.StringVar] = None
        # 启动自动连设备（每 interval 一次，最多 max 次）
        self._auto_connect_job: Optional[str] = None
        self._auto_connect_active = True
        self._auto_connect_attempts_used = 0
        # 自动进入游戏工作线程与会话号（断开时递增，避免旧线程 finally 误解除暂停）
        self._game_autolaunch_session: int = 0

        root.title("小橙碎多任务自动化  —  控制面板")
        root.geometry("880x920")
        root.minsize(760, 600)
        root.resizable(True, True)

        self._build_ui()
        self._setup_device_callbacks()
        self._schedule_refresh()
        self._schedule_initial_auto_connect()

    # ── UI 构建 ────────────────────────────────────────────

    def _build_ui(self):
        root = self._root
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(3, weight=1)

        # ── 左列 row=0：设备连接区 ─────────────────────────
        conn_lf = ttk.LabelFrame(root, text="  设备连接  ", padding=8)
        conn_lf.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=(8, 4))
        conn_lf.columnconfigure(1, weight=1)

        emu_cfg = self._config.get("emulator", {})

        ttk.Label(conn_lf, text="设备 ID:").grid(row=0, column=0, sticky="w")
        dev_row = ttk.Frame(conn_lf)
        dev_row.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        dev_row.columnconfigure(0, weight=1)

        self._device_var = tk.StringVar(value=emu_cfg.get("device_id", ""))
        self._device_cb = ttk.Combobox(dev_row, textvariable=self._device_var)
        self._device_cb.grid(row=0, column=0, sticky="ew")
        ttk.Button(dev_row, text="刷新", width=5,
                   command=self._refresh_devices).grid(row=0, column=1, padx=(4, 0))

        param_row = ttk.Frame(conn_lf)
        param_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(param_row, text="最大宽度:").pack(side=tk.LEFT)
        self._max_width_var = tk.IntVar(value=emu_cfg.get("max_width", 1920))
        ttk.Spinbox(param_row, from_=320, to=3840, increment=160,
                    textvariable=self._max_width_var, width=6).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(param_row, text="最大帧率:").pack(side=tk.LEFT)
        self._max_fps_var = tk.IntVar(value=emu_cfg.get("max_fps", 30))
        ttk.Spinbox(param_row, from_=5, to=60, increment=5,
                    textvariable=self._max_fps_var, width=5).pack(side=tk.LEFT, padx=(4, 0))

        btn_row = ttk.Frame(conn_lf)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._conn_btn = ttk.Button(btn_row, text="▶ 连接设备",
                                    command=self._toggle_connect, width=14)
        self._conn_btn.pack(side=tk.LEFT)
        self._conn_status_lbl = ttk.Label(btn_row, text="● 未连接", foreground="#cc4444")
        self._conn_status_lbl.pack(side=tk.LEFT, padx=(10, 0))

        # ── 左列 row=1：任务控制区（仅任务开关）──────────────
        task_lf = ttk.LabelFrame(root, text="  任务控制  ", padding=8)
        task_lf.grid(row=1, column=0, sticky="ew", padx=(8, 4), pady=4)
        task_lf.columnconfigure(1, weight=1)

        # 抢红包
        rp_row = ttk.Frame(task_lf)
        rp_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        rp_row.columnconfigure(1, weight=1)
        ttk.Label(rp_row, text="抢红包").grid(row=0, column=0, sticky="w")
        self._rp_state_lbl = ttk.Label(rp_row, text="— 未运行 —",
                                        foreground="#888888", width=16)
        self._rp_state_lbl.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._rp_btn = ttk.Button(rp_row, text="开 启", width=8,
                                   command=self._toggle_red_packet)
        self._rp_btn.grid(row=0, column=2, sticky="e")

        # 抢红包：模式切换行
        rp_cfg_init = self._config.get("tasks", {}).get("red_packet_task", {})
        rp_mode_row = ttk.Frame(rp_row)
        rp_mode_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(rp_mode_row, text="模式:").pack(side=tk.LEFT)
        self._rp_mode_var = tk.StringVar(value=rp_cfg_init.get("chat_mode", "alliance"))
        ttk.Radiobutton(
            rp_mode_row, text="世界红包", value="world",
            variable=self._rp_mode_var, command=self._on_rp_mode_changed,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            rp_mode_row, text="联盟红包", value="alliance",
            variable=self._rp_mode_var, command=self._on_rp_mode_changed,
        ).pack(side=tk.LEFT, padx=(8, 0))

        # 抢红包：扫描间隔 & 点击延迟行
        rp_params_row = ttk.Frame(rp_row)
        rp_params_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(rp_params_row, text="扫描间隔:").pack(side=tk.LEFT)
        self._rp_scan_interval_var = tk.StringVar(
            value=str(rp_cfg_init.get("scan_interval_secs", 1.0))
        )
        ttk.Entry(rp_params_row, textvariable=self._rp_scan_interval_var, width=6).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Label(rp_params_row, text="s").pack(side=tk.LEFT)
        ttk.Button(
            rp_params_row, text="确认", width=5,
            command=self._on_rp_scan_interval_confirm,
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(rp_params_row, text="点击延迟:").pack(side=tk.LEFT)
        self._rp_click_delay_var = tk.StringVar(
            value=str(rp_cfg_init.get("click_delay_secs", 0.3))
        )
        ttk.Entry(rp_params_row, textvariable=self._rp_click_delay_var, width=6).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Label(rp_params_row, text="s").pack(side=tk.LEFT)
        ttk.Button(
            rp_params_row, text="确认", width=5,
            command=self._on_rp_click_delay_confirm,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # 打熊
        ba_row = ttk.Frame(task_lf)
        ba_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ba_row.columnconfigure(1, weight=1)
        ttk.Label(ba_row, text="打  熊").grid(row=0, column=0, sticky="w")
        self._ba_state_lbl = ttk.Label(ba_row, text="— 未运行 —",
                                        foreground="#888888", width=16)
        self._ba_state_lbl.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._ba_btn = ttk.Button(ba_row, text="开 启", width=8,
                                   command=self._toggle_bear_attack)
        self._ba_btn.grid(row=0, column=2, sticky="e")
        self._ba_mode_var = tk.StringVar(value="bear")
        ba_mode_row = ttk.Frame(ba_row)
        ba_mode_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        ttk.Radiobutton(
            ba_mode_row, text="打熊", value="bear",
            variable=self._ba_mode_var, command=self._on_ba_mode_changed,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            ba_mode_row, text="打怪", value="monster",
            variable=self._ba_mode_var, command=self._on_ba_mode_changed,
        ).pack(side=tk.LEFT, padx=(10, 0))

        # 联盟总动员
        am_row = ttk.Frame(task_lf)
        am_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        am_row.columnconfigure(1, weight=1)
        ttk.Label(am_row, text="联盟总动员").grid(row=0, column=0, sticky="w")
        self._am_state_lbl = ttk.Label(am_row, text="— 未运行 —",
                                       foreground="#888888", width=16)
        self._am_state_lbl.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._am_btn = ttk.Button(am_row, text="开 启", width=8,
                                  command=self._toggle_alliance_mobilization)
        self._am_btn.grid(row=0, column=2, sticky="e")

        # 专属任务刷新
        exc_row = ttk.Frame(task_lf)
        exc_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        exc_row.columnconfigure(1, weight=1)
        ttk.Label(exc_row, text="专属任务刷新").grid(row=0, column=0, sticky="w")
        self._exc_state_lbl = ttk.Label(exc_row, text="— 未运行 —",
                                        foreground="#888888", width=16)
        self._exc_state_lbl.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._exc_btn = ttk.Button(exc_row, text="开 启", width=8,
                                   command=self._toggle_exc_monitor)
        self._exc_btn.grid(row=0, column=2, sticky="e")

        exc_refresh_row = ttk.Frame(task_lf)
        exc_refresh_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(exc_refresh_row, text="专属刷新").pack(side=tk.LEFT)
        ttk.Label(exc_refresh_row, text="坑位1").pack(side=tk.LEFT, padx=(8, 0))
        self._exc_slot1_refresh_var = tk.StringVar()
        self._exc_slot1_refresh_cb = ttk.Combobox(
            exc_refresh_row, width=6, state="readonly",
            values=["2分钟", "5分钟"], textvariable=self._exc_slot1_refresh_var,
        )
        self._exc_slot1_refresh_cb.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(exc_refresh_row, text="坑位2").pack(side=tk.LEFT)
        self._exc_slot2_refresh_var = tk.StringVar()
        self._exc_slot2_refresh_cb = ttk.Combobox(
            exc_refresh_row, width=6, state="readonly",
            values=["2分钟", "5分钟"], textvariable=self._exc_slot2_refresh_var,
        )
        self._exc_slot2_refresh_cb.pack(side=tk.LEFT, padx=(4, 0))
        self._init_exc_refresh_ui_vars()
        self._exc_slot1_refresh_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_exc_slot_refresh_changed(0))
        self._exc_slot2_refresh_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_exc_slot_refresh_changed(1))

        # 联盟监控：并发 / 公共自动抢（写入 tasks.alliance_monitor_task，新启任务生效）
        am_mon_cfg_row = ttk.Frame(task_lf)
        am_mon_cfg_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        _am_cfg = self._get_alliance_monitor_cfg()
        self._am_concurrent_var = tk.BooleanVar(
            value=bool(_am_cfg.get("concurrent_exclusive_public", False))
        )
        ttk.Checkbutton(
            am_mon_cfg_row,
            text="专属+公共同时开",
            variable=self._am_concurrent_var,
            command=self._on_am_concurrent_changed,
        ).pack(side=tk.LEFT)
        self._pub_auto_accept_var = tk.BooleanVar(value=bool(_am_cfg.get("pub_auto_accept", True)))
        ttk.Checkbutton(
            am_mon_cfg_row,
            text="公共自动抢",
            variable=self._pub_auto_accept_var,
            command=self._on_pub_auto_accept_changed,
        ).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(
            am_mon_cfg_row,
            text="（关=仅飞书；并发开时默认同屏 y>1023 匹配公共模板）",
            foreground="#666666",
            font=("", 8),
        ).pack(side=tk.LEFT, padx=(10, 0))

        # 公共任务抢
        pub_row = ttk.Frame(task_lf)
        pub_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        pub_row.columnconfigure(1, weight=1)
        ttk.Label(pub_row, text="公共任务抢").grid(row=0, column=0, sticky="w")
        self._pub_state_lbl = ttk.Label(pub_row, text="— 未运行 —",
                                        foreground="#888888", width=16)
        self._pub_state_lbl.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._pub_btn = ttk.Button(pub_row, text="开 启", width=8,
                                   command=self._toggle_pub_monitor)
        self._pub_btn.grid(row=0, column=2, sticky="e")

        # 集结巨兽
        gbr_row = ttk.Frame(task_lf)
        gbr_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        gbr_row.columnconfigure(1, weight=1)
        ttk.Label(gbr_row, text="集结巨兽").grid(row=0, column=0, sticky="w")
        self._gbr_state_lbl = ttk.Label(gbr_row, text="— 未运行 —",
                                        foreground="#888888", width=16)
        self._gbr_state_lbl.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._gbr_btn = ttk.Button(gbr_row, text="开 启", width=8,
                                   command=self._toggle_giant_beast_rally)
        self._gbr_btn.grid(row=0, column=2, sticky="e")

        # ── 左列 row=2：调度器状态区 ───────────────────────
        sched_lf = ttk.LabelFrame(root, text="  调度器  ", padding=8)
        sched_lf.grid(row=2, column=0, sticky="ew", padx=(8, 4), pady=4)
        sched_lf.columnconfigure(1, weight=1)

        for i, (label, attr) in enumerate([
            ("状态:",    "_sched_running_lbl"),
            ("当前任务:", "_sched_task_lbl"),
            ("Tick 计数:", "_sched_tick_lbl"),
        ]):
            ttk.Label(sched_lf, text=label).grid(row=i, column=0, sticky="w", pady=2)
            lbl = ttk.Label(sched_lf, text="—")
            lbl.grid(row=i, column=1, sticky="w", padx=(8, 0), pady=2)
            setattr(self, attr, lbl)

        sched_ctrl = ttk.Frame(sched_lf)
        sched_ctrl.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._sched_btn = ttk.Button(sched_ctrl, text="▶ 启动调度器",
                                      command=self._toggle_scheduler, width=14)
        self._sched_btn.pack(side=tk.LEFT)

        # ── 右列：容器（row=0..2 rowspan=3）──────────────────
        right_frame = ttk.Frame(root)
        right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(4, 8), pady=(8, 4))
        right_frame.columnconfigure(0, weight=1)

        # ── 右列：打熊配置区 ──────────────────────────────
        bear_cfg_lf = ttk.LabelFrame(right_frame, text="  打熊配置  ", padding=8)
        bear_cfg_lf.pack(fill=tk.X, expand=False, pady=(0, 4))
        bear_cfg_lf.columnconfigure(1, weight=1)
        bear_cfg_lf.columnconfigure(3, weight=1)
        bear_cfg_lf.columnconfigure(5, weight=1)

        self._bear_team_template_options = self._collect_bear_team_template_options()
        teams_cfg = self._get_bear_attack_teams_cfg()

        ttk.Label(bear_cfg_lf, text="队伍模板").grid(row=0, column=0, columnspan=5, sticky="w")
        ttk.Button(
            bear_cfg_lf, text="更新模板", width=8,
            command=self._refresh_bear_team_templates,
        ).grid(row=0, column=5, sticky="e")

        self._teams_frame = teams_frame = ttk.Frame(bear_cfg_lf)
        teams_frame.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(4, 0))
        teams_frame.columnconfigure(0, weight=1)
        teams_frame.columnconfigure(1, weight=1)
        teams_frame.columnconfigure(2, weight=1)
        teams_frame.columnconfigure(3, weight=1)

        for idx, team_id in enumerate((1, 2, 3, 4)):
            team_lf = ttk.LabelFrame(teams_frame, text=f"  队伍{team_id}  ", padding=4)
            team_lf.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0))

            self._team_checkbox_vars[team_id] = {}
            selected = set(teams_cfg.get(team_id, []))

            for name in self._bear_team_template_options:
                var = tk.BooleanVar(value=(name in selected))
                self._team_checkbox_vars[team_id][name] = var
                ttk.Checkbutton(
                    team_lf, text=name, variable=var,
                    command=lambda tid=team_id: self._on_bear_team_templates_changed(tid),
                ).pack(anchor="w", pady=1)

        ba_cooldown_row = ttk.Frame(bear_cfg_lf)
        ba_cooldown_row.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        ttk.Label(ba_cooldown_row, text="打熊清冷却").pack(side=tk.LEFT)
        ttk.Button(
            ba_cooldown_row, text="队伍1", width=8,
            command=lambda: self._clear_bear_attack_team_cooldown(1),
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            ba_cooldown_row, text="队伍2", width=8,
            command=lambda: self._clear_bear_attack_team_cooldown(2),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            ba_cooldown_row, text="队伍3", width=8,
            command=lambda: self._clear_bear_attack_team_cooldown(3),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            ba_cooldown_row, text="队伍4", width=8,
            command=lambda: self._clear_bear_attack_team_cooldown(4),
        ).pack(side=tk.LEFT, padx=(6, 0))

        ba_cd_info_row = ttk.Frame(bear_cfg_lf)
        ba_cd_info_row.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(4, 0))
        ttk.Label(ba_cd_info_row, text="冷却剩余").pack(side=tk.LEFT)
        for team_id in (1, 2, 3, 4):
            ttk.Label(ba_cd_info_row, text=f"  队{team_id}:").pack(side=tk.LEFT)
            lbl = ttk.Label(ba_cd_info_row, text="—", foreground="#888888", width=5, anchor="w")
            lbl.pack(side=tk.LEFT)
            self._ba_cd_labels[team_id] = lbl

        # ── 打熊延时配置行 ─────────────────────────────────
        ba_delay_row = ttk.Frame(bear_cfg_lf)
        ba_delay_row.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        ttk.Label(ba_delay_row, text="出征延时(s)").pack(side=tk.LEFT)

        ba_cfg_init = self._config.get("tasks", {}).get("bear_attack_task", {})
        self._ba_delay_group_var = tk.DoubleVar(
            value=round(float(ba_cfg_init.get("delay_before_click_group_secs", 0.1)), 1)
        )
        self._ba_delay_expedition_var = tk.DoubleVar(
            value=round(float(ba_cfg_init.get("delay_before_click_expedition_secs", 0.1)), 1)
        )

        ttk.Label(ba_delay_row, text="  编队:").pack(side=tk.LEFT)
        self._build_step_spinner(
            ba_delay_row, self._ba_delay_group_var,
            step=0.1, min_val=0.0, max_val=5.0,
            on_change=self._on_ba_delays_changed,
        )
        ttk.Label(ba_delay_row, text="  出征:").pack(side=tk.LEFT)
        self._build_step_spinner(
            ba_delay_row, self._ba_delay_expedition_var,
            step=0.1, min_val=0.0, max_val=5.0,
            on_change=self._on_ba_delays_changed,
        )

        # ── 右列：打熊调试区 ──────────────────────────────
        bad_lf = ttk.LabelFrame(right_frame, text="  打熊调试（最小等待时间）  ", padding=6)
        bad_lf.pack(fill=tk.X, expand=False, pady=(0, 4))
        bad_lf.columnconfigure(1, weight=1)
        bad_lf.columnconfigure(3, weight=1)

        ttk.Label(bad_lf, text="进入编队点:").grid(row=0, column=0, sticky="w")
        self._bad_enter_group_var = tk.StringVar(
            value=str(list(
                self._config.get("tasks", {})
                .get("bear_attack_debug_task", {})
                .get("enter_group_point", [766, 1448])
            ))
        )
        ttk.Entry(bad_lf, textvariable=self._bad_enter_group_var, width=14).grid(
            row=0, column=1, sticky="ew", padx=(4, 12)
        )

        ttk.Label(bad_lf, text="编队点:").grid(row=0, column=2, sticky="w")
        self._bad_group_var = tk.StringVar(
            value=str(list(
                self._config.get("tasks", {})
                .get("bear_attack_debug_task", {})
                .get("group_point", [81, 146])
            ))
        )
        ttk.Entry(bad_lf, textvariable=self._bad_group_var, width=14).grid(
            row=0, column=3, sticky="ew", padx=(4, 0)
        )

        bad_delay_row = ttk.Frame(bad_lf)
        bad_delay_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Label(bad_delay_row, text="延时(s) 进入编队→编队:").pack(side=tk.LEFT)
        self._bad_delay1_var = tk.StringVar(
            value=str(
                self._config.get("tasks", {})
                .get("bear_attack_debug_task", {})
                .get("delay_before_click_group_secs", 0.1)
            )
        )
        ttk.Entry(bad_delay_row, textvariable=self._bad_delay1_var, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(bad_delay_row, text="编队→出征:").pack(side=tk.LEFT)
        self._bad_delay2_var = tk.StringVar(
            value=str(
                self._config.get("tasks", {})
                .get("bear_attack_debug_task", {})
                .get("delay_before_click_expedition_secs", 0.1)
            )
        )
        ttk.Entry(bad_delay_row, textvariable=self._bad_delay2_var, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(bad_delay_row, text="出征→检测:").pack(side=tk.LEFT)
        self._bad_delay3_var = tk.StringVar(
            value=str(
                self._config.get("tasks", {})
                .get("bear_attack_debug_task", {})
                .get("delay_after_click_expedition_check_secs", 0.3)
            )
        )
        ttk.Entry(bad_delay_row, textvariable=self._bad_delay3_var, width=6).pack(side=tk.LEFT, padx=(4, 0))

        bad_ctrl_row = ttk.Frame(bad_lf)
        bad_ctrl_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self._bad_state_lbl = ttk.Label(bad_ctrl_row, text="— 待执行 —", foreground="#888888", width=18)
        self._bad_state_lbl.pack(side=tk.LEFT)
        self._bad_btn = ttk.Button(
            bad_ctrl_row, text="▶ 执行调试", width=12, command=self._run_bear_attack_debug
        )
        self._bad_btn.pack(side=tk.RIGHT)

        # ── 右列：手动控制区 ──────────────────────────────
        manual_lf = ttk.LabelFrame(right_frame, text="  手动控制  ", padding=8)
        manual_lf.pack(fill=tk.X, expand=False)

        manual_row = ttk.Frame(manual_lf)
        manual_row.pack(fill=tk.X, expand=False)
        ttk.Button(
            manual_row, text="返回上一步", width=10, command=self._manual_go_back
        ).pack(side=tk.RIGHT)
        ttk.Button(
            manual_row, text="返回首页", width=10, command=self._manual_go_home
        ).pack(side=tk.RIGHT, padx=(0, 6))

        manual_swipe_row = ttk.Frame(manual_lf)
        manual_swipe_row.pack(fill=tk.X, expand=False, pady=(6, 0))
        ttk.Label(manual_swipe_row, text="起点").pack(side=tk.LEFT)
        self._manual_swipe_start_var = tk.StringVar(value="[571, 1445]")
        ttk.Entry(
            manual_swipe_row, textvariable=self._manual_swipe_start_var, width=12,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(manual_swipe_row, text="终点").pack(side=tk.LEFT)
        self._manual_swipe_end_var = tk.StringVar(value="[571, 1200]")
        ttk.Entry(
            manual_swipe_row, textvariable=self._manual_swipe_end_var, width=12,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(manual_swipe_row, text="时长ms").pack(side=tk.LEFT)
        self._manual_swipe_duration_var = tk.IntVar(value=300)
        ttk.Spinbox(
            manual_swipe_row, from_=1, to=10000, increment=50,
            textvariable=self._manual_swipe_duration_var, width=6,
        ).pack(side=tk.LEFT, padx=(4, 8))

        manual_swipe_ctrl_row = ttk.Frame(manual_lf)
        manual_swipe_ctrl_row.pack(fill=tk.X, expand=False, pady=(4, 0))
        ttk.Label(manual_swipe_ctrl_row, text="控制方式").pack(side=tk.LEFT)
        self._manual_swipe_mode_var = tk.StringVar(value="scrcpy")
        ttk.Radiobutton(
            manual_swipe_ctrl_row, text="Scrcpy", value="scrcpy",
            variable=self._manual_swipe_mode_var,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            manual_swipe_ctrl_row, text="ADB", value="adb",
            variable=self._manual_swipe_mode_var,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            manual_swipe_ctrl_row, text="执行滑动", width=10, command=self._manual_swipe,
        ).pack(side=tk.RIGHT)

        # ── 底部通栏：日志区 ──────────────────────────────
        log_lf = ttk.LabelFrame(root, text="  日志  ", padding=4)
        log_lf.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=8, pady=(4, 8))
        log_lf.rowconfigure(0, weight=1)
        log_lf.columnconfigure(0, weight=1)

        self._log_txt = scrolledtext.ScrolledText(
            log_lf, height=8, state=tk.DISABLED,
            font=("Consolas", 9), bg="#0d0d1a", fg="#cccccc",
            insertbackground="white",
        )
        self._log_txt.grid(row=0, column=0, sticky="nsew")
        ttk.Button(log_lf, text="清空", command=self._clear_log,
                   width=6).grid(row=1, column=0, sticky="e", pady=(2, 0))

        # 挂载日志 handler
        self._log_handler = _TkLogHandler(self._log)
        self._log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)s | %(message)s")
        self._log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self._log_handler)

    # ── 设备连接 ───────────────────────────────────────────

    def _setup_device_callbacks(self):
        self._device.on_connect = self._on_device_connect
        self._device.on_disconnect = self._on_device_disconnect

    def _refresh_devices(self):
        def _work():
            devices = Device.list_devices()
            def _update():
                if devices:
                    self._device_cb["values"] = devices
                    if self._device_var.get() not in devices:
                        self._device_var.set(devices[0])
                    self._log(f"发现 {len(devices)} 个设备: {', '.join(devices)}")
                else:
                    self._log("未发现 ADB 设备")
            self._root.after(0, _update)
        threading.Thread(target=_work, daemon=True).start()

    def _toggle_connect(self):
        if self._device.is_connected:
            self._stop_auto_connect("用户断开连接")
            self._device.disconnect()
        else:
            self._stop_auto_connect("用户手动连接")
            device_id = self._device_var.get().strip()
            if not device_id:
                messagebox.showwarning("未选择设备", "请先输入或选择设备 ID", parent=self._root)
                return
            # 把控制面板的连接参数同步写回 config，Device 连接时从此读取
            emu = self._config.setdefault("emulator", {})
            emu["device_id"] = device_id
            emu["max_width"] = self._max_width_var.get()
            emu["max_fps"]   = self._max_fps_var.get()
            self._conn_btn.configure(state=tk.DISABLED)
            self._log(f"正在连接: {device_id}  max_width={emu['max_width']} max_fps={emu['max_fps']}…")
            self._device.connect()

    def _cancel_auto_connect_timer(self) -> None:
        if self._auto_connect_job is not None:
            try:
                self._root.after_cancel(self._auto_connect_job)
            except Exception:
                pass
            self._auto_connect_job = None

    def _stop_auto_connect(self, reason: str = "") -> None:
        """停止启动自动连接序列（手动连/断时调用）。"""
        self._auto_connect_active = False
        self._cancel_auto_connect_timer()
        if reason:
            logger.debug("自动连接已停止: %s", reason)

    def _schedule_initial_auto_connect(self) -> None:
        if not self._auto_connect_active:
            return
        self._cancel_auto_connect_timer()
        self._auto_connect_job = self._root.after(
            self.AUTO_CONNECT_INITIAL_MS, self._auto_connect_try,
        )

    def _schedule_next_auto_connect(self) -> None:
        if not self._auto_connect_active:
            return
        self._cancel_auto_connect_timer()
        self._auto_connect_job = self._root.after(
            self.AUTO_CONNECT_INTERVAL_MS, self._auto_connect_try,
        )

    def _sync_emulator_config_from_ui(self) -> None:
        emu = self._config.setdefault("emulator", {})
        emu["device_id"] = self._device_var.get().strip()
        emu["max_width"] = self._max_width_var.get()
        emu["max_fps"] = self._max_fps_var.get()

    def _auto_connect_try(self) -> None:
        """自动连接单次尝试（主线程）。"""
        self._auto_connect_job = None
        if not self._auto_connect_active:
            return
        if self._device.is_connected:
            self._stop_auto_connect("")
            return
        th = getattr(self._device, "_thread", None)
        if th is not None and th.is_alive():
            self._log("自动连接：已有连接线程进行中，稍后再试…")
            self._auto_connect_job = self._root.after(2_000, self._auto_connect_try)
            return
        if self._auto_connect_attempts_used >= self.AUTO_CONNECT_MAX_ATTEMPTS:
            self._log(
                f"自动连接：已连续尝试 {self.AUTO_CONNECT_MAX_ATTEMPTS} 次仍未成功，"
                f"已停止。请检查模拟器或点击「连接设备」。"
            )
            self._stop_auto_connect("")
            return
        device_id = self._device_var.get().strip()
        if not device_id:
            self._log("自动连接跳过：未配置设备 ID")
            self._stop_auto_connect("")
            return
        self._auto_connect_attempts_used += 1
        n = self._auto_connect_attempts_used
        self._sync_emulator_config_from_ui()
        self._conn_btn.configure(state=tk.DISABLED)
        self._log(
            f"自动连接 第 {n}/{self.AUTO_CONNECT_MAX_ATTEMPTS} 次 "
            f"（间隔 {self.AUTO_CONNECT_INTERVAL_MS // 1000}s）: {device_id} …"
        )
        self._device.connect()

    def _on_device_connect(self):
        self._game_autolaunch_session += 1
        launch_sid = self._game_autolaunch_session
        self._ctx.scheduler_paused_for_game_autolaunch = True
        threading.Thread(
            target=self._run_game_autolaunch_worker,
            args=(launch_sid,),
            daemon=True,
        ).start()

        def _ui():
            self._stop_auto_connect("")
            self._conn_btn.configure(text="■ 断开连接", state=tk.NORMAL)
            self._conn_status_lbl.configure(text="● 已连接", foreground="#22aa55")
            self._log(f"设备已连接: {self._device_var.get()}")
            self._notify_event("设备已连接")
        self._root.after(0, _ui)

    def _on_device_disconnect(self):
        def _ui():
            self._game_autolaunch_session += 1
            self._ctx.scheduler_paused_for_game_autolaunch = False
            self._conn_btn.configure(text="▶ 连接设备", state=tk.NORMAL)
            self._conn_status_lbl.configure(text="● 未连接", foreground="#cc4444")
            self._log("设备连接已断开")
            self._notify_event("设备已断开")
            if (
                self._auto_connect_active
                and not self._device.is_connected
                and self._auto_connect_attempts_used < self.AUTO_CONNECT_MAX_ATTEMPTS
            ):
                self._log(
                    f"将在 {self.AUTO_CONNECT_INTERVAL_MS // 1000}s 后自动重试连接 "
                    f"（剩余 {self.AUTO_CONNECT_MAX_ATTEMPTS - self._auto_connect_attempts_used} 次）"
                )
                self._schedule_next_auto_connect()
            elif (
                self._auto_connect_active
                and not self._device.is_connected
                and self._auto_connect_attempts_used >= self.AUTO_CONNECT_MAX_ATTEMPTS
            ):
                self._log("自动连接：已达尝试上限，不再重试。")
                self._stop_auto_connect("")
        self._root.after(0, _ui)

    @staticmethod
    def _classify_endless_winter_screen(width: int, height: int, tol: int) -> str:
        """
        返回 "game" | "launcher" | "unknown"
        width/height 为视频帧 shape[1] / shape[0]（与常见「宽×高」表述一致）。
        """
        w, h = width, height
        if abs(w - 864) <= tol and abs(h - 1920) <= tol:
            return "game"
        if abs(w - 1920) <= tol and abs(h - 864) <= tol:
            return "launcher"
        if h > w * 1.2:
            return "game"
        if w > h * 1.2:
            return "launcher"
        return "unknown"

    def _run_game_autolaunch_worker(self, session_id: int) -> None:
        """连接建立后在后台线程执行：必要时从桌面点击「无尽冬日」再等待进入游戏。"""
        tmpl = self.GAME_LAUNCH_TEMPLATE_NAME
        tol = self.GAME_AUTOLAUNCH_SIZE_TOLERANCE

        def _ui_log(msg: str) -> None:
            self._root.after(0, lambda m=msg: self._log(m))

        def _ui_notify(msg: str) -> None:
            self._root.after(0, lambda m=msg: self._notify_event(m))

        def _maybe_unpause() -> None:
            def _u() -> None:
                if self._game_autolaunch_session == session_id:
                    self._ctx.scheduler_paused_for_game_autolaunch = False

            try:
                self._root.after(0, _u)
            except Exception:
                if self._game_autolaunch_session == session_id:
                    self._ctx.scheduler_paused_for_game_autolaunch = False

        try:
            deadline = time.monotonic() + self.GAME_LAUNCH_FRAME_WAIT_SEC
            frame = None
            while self._device.is_connected and time.monotonic() < deadline:
                frame = self._ctx.get_frame()
                if frame is not None and getattr(frame, "size", 0) > 0:
                    break
                time.sleep(0.15)

            if not self._device.is_connected:
                return

            if frame is None:
                _ui_log("自动进入游戏：超时未收到画面，已恢复任务调度。")
                return

            h, w = frame.shape[:2]
            kind = self._classify_endless_winter_screen(w, h, tol)

            if kind == "game":
                _ui_log(f"检测到竖屏画面（{w}×{h}），判定已在无尽冬日内，跳过桌面启动。")
                _ui_notify("已自动进入无尽冬日游戏")
                return

            if kind == "unknown":
                msg = (
                    f"无法根据分辨率判断是否在桌面或游戏内（当前 {w}×{h}），"
                    "请手动进入无尽冬日后再启用任务。"
                )
                _ui_log(msg)
                _ui_notify(msg)
                return

            # launcher：在桌面查找图标
            paths = self._scheduler._build_template_paths([tmpl])
            icon_path = paths.get(tmpl)
            if not icon_path:
                msg = (
                    f"未找到「{tmpl}」模板文件，请在 assets/templates 下放置 "
                    f"{tmpl}.png（或 .jpg / .jpeg）。"
                )
                _ui_log(msg)
                _ui_notify(msg)
                return

            _ui_log(f"检测到横屏桌面（{w}×{h}），开始查找「{tmpl}」图标…")
            clicked = False
            for attempt in range(1, self.GAME_LAUNCH_MAX_RETRIES + 1):
                if not self._device.is_connected:
                    return
                frame2 = self._ctx.get_frame()
                if frame2 is None or getattr(frame2, "size", 0) == 0:
                    if attempt < self.GAME_LAUNCH_MAX_RETRIES:
                        time.sleep(self.GAME_LAUNCH_RETRY_INTERVAL_SEC)
                    continue

                res = self._scheduler._vision.match(frame2, icon_path)
                if res.found and self._device.click(res.x, res.y):
                    clicked = True
                    _ui_log(
                        f"已点击「{tmpl}」图标（第 {attempt}/{self.GAME_LAUNCH_MAX_RETRIES} 次尝试，"
                        f"置信度 {res.confidence:.3f}）。"
                    )
                    break

                _ui_log(
                    f"第 {attempt}/{self.GAME_LAUNCH_MAX_RETRIES} 次未匹配到「{tmpl}」，"
                    f"{int(self.GAME_LAUNCH_RETRY_INTERVAL_SEC)}s 后重试…"
                )
                if attempt < self.GAME_LAUNCH_MAX_RETRIES:
                    time.sleep(self.GAME_LAUNCH_RETRY_INTERVAL_SEC)

            if not clicked:
                msg = (
                    f"自动进入无尽冬日失败：桌面未找到「{tmpl}」图标"
                    f"（已重试 {self.GAME_LAUNCH_MAX_RETRIES} 次，间隔 "
                    f"{int(self.GAME_LAUNCH_RETRY_INTERVAL_SEC)}s），请手动打开游戏。"
                )
                _ui_log(msg)
                _ui_notify(msg)
                return

            # 点击后等待游戏加载
            waited = 0.0
            step = 0.2
            while waited < self.GAME_LAUNCH_POST_CLICK_WAIT_SEC and self._device.is_connected:
                time.sleep(step)
                waited += step

            if not self._device.is_connected:
                return

            _ui_log(
                f"已等待 {int(self.GAME_LAUNCH_POST_CLICK_WAIT_SEC)}s，假定已进入游戏，恢复任务调度。"
            )
            _ui_notify("已自动进入无尽冬日游戏")
        except Exception:
            logger.exception("自动进入无尽冬日流程异常")
            _ui_log("自动进入无尽冬日流程异常，已恢复任务调度（详见日志）。")
            _ui_notify("自动进入无尽冬日流程异常，请查看控制面板日志。")
        finally:
            _maybe_unpause()

    # ── 任务开关 ───────────────────────────────────────────

    def _toggle_red_packet(self):
        if self._red_packet_task and self._red_packet_task.is_enabled:
            # 停止
            self._red_packet_task.is_enabled = False
            self._scheduler.remove_task(self._red_packet_task)
            self._red_packet_task = None
            self._rp_btn.configure(text="开 启")
            self._rp_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")
            self._log("抢红包任务已停止")
            self._notify_event("抢红包任务已停止")
        else:
            # 首次开启前执行断线重连检查
            reconnect_msg = self._check_and_reconnect_if_disconnected()
            if reconnect_msg:
                self._log(f"断线重连检查：{reconnect_msg}")
            # 启动时将当前 UI 参数写入 config
            rp_cfg = self._config.setdefault("tasks", {}).setdefault("red_packet_task", {})
            if self._rp_mode_var:
                rp_cfg["chat_mode"] = self._rp_mode_var.get()
            if self._rp_scan_interval_var:
                try:
                    rp_cfg["scan_interval_secs"] = float(self._rp_scan_interval_var.get())
                except ValueError:
                    pass
            if self._rp_click_delay_var:
                try:
                    rp_cfg["click_delay_secs"] = float(self._rp_click_delay_var.get())
                except ValueError:
                    pass
            task = RedPacketTask(self._config, notify_callback=self._feishu_notify)
            self._red_packet_task = task
            self._scheduler.add_task(task)
            self._rp_btn.configure(text="停 止")
            self._rp_state_lbl.configure(text="运行中…", foreground="#22aa55")
            self._log("抢红包任务已启动")
            self._notify_event("抢红包任务已启动")

    def _on_rp_mode_changed(self) -> None:
        """模式单选按钮切换时回调，若任务运行中则实时生效。"""
        if not self._rp_mode_var:
            return
        mode = self._rp_mode_var.get()
        if self._red_packet_task and self._red_packet_task.is_enabled:
            self._red_packet_task.set_chat_mode(mode)
            self._log(f"抢红包模式切换为：{'世界' if mode == 'world' else '联盟'}")

    def _on_rp_scan_interval_confirm(self) -> None:
        """扫描间隔确认按钮回调，实时更新任务参数。"""
        if not self._rp_scan_interval_var:
            return
        try:
            val = float(self._rp_scan_interval_var.get())
        except ValueError:
            self._log("扫描间隔输入无效，请输入数字")
            return
        if val < 0.1:
            self._log("扫描间隔最小为 0.1s")
            self._rp_scan_interval_var.set("0.1")
            val = 0.1
        if self._red_packet_task and self._red_packet_task.is_enabled:
            self._red_packet_task.set_scan_interval(val)
        self._log(f"抢红包扫描间隔更新为 {val}s")

    def _on_rp_click_delay_confirm(self) -> None:
        """点击延迟确认按钮回调，实时更新任务参数。"""
        if not self._rp_click_delay_var:
            return
        try:
            val = float(self._rp_click_delay_var.get())
        except ValueError:
            self._log("点击延迟输入无效，请输入数字")
            return
        if val < 0.0:
            self._log("点击延迟最小为 0s")
            self._rp_click_delay_var.set("0.0")
            val = 0.0
        if self._red_packet_task and self._red_packet_task.is_enabled:
            self._red_packet_task.set_click_delay(val)
        self._log(f"抢红包点击延迟更新为 {val}s")

    def _get_bear_attack_cfg(self) -> dict:
        return self._config.setdefault("tasks", {}).setdefault("bear_attack_task", {})

    def _get_alliance_monitor_cfg(self) -> dict:
        return self._config.setdefault("tasks", {}).setdefault("alliance_monitor_task", {})

    def _alliance_monitor_concurrent_mode(self) -> bool:
        return bool(self._get_alliance_monitor_cfg().get("concurrent_exclusive_public", False))

    def _on_am_concurrent_changed(self) -> None:
        v = bool(self._am_concurrent_var.get())
        self._get_alliance_monitor_cfg()["concurrent_exclusive_public"] = v
        self._log(f"联盟监控：专属+公共同时开 → {'开' if v else '关'}（下次开启任务时生效）")

    def _on_pub_auto_accept_changed(self) -> None:
        v = bool(self._pub_auto_accept_var.get())
        self._get_alliance_monitor_cfg()["pub_auto_accept"] = v
        self._log(f"联盟监控：公共自动抢 → {'开' if v else '关（仅飞书）'}（下次开启公共任务时生效）")

    @staticmethod
    def _seconds_to_refresh_label(seconds: float) -> str:
        return "5分钟" if float(seconds) >= 300 else "2分钟"

    @staticmethod
    def _refresh_label_to_seconds(label: str) -> int:
        return 300 if (label or "").strip() == "5分钟" else 120

    def _init_exc_refresh_ui_vars(self) -> None:
        cfg = self._get_alliance_monitor_cfg()
        slot_refresh = cfg.get("slot_refresh_secs", [])
        if isinstance(slot_refresh, (list, tuple)) and len(slot_refresh) >= 2:
            slot1_secs = float(slot_refresh[0])
            slot2_secs = float(slot_refresh[1])
        else:
            default_secs = float(cfg.get("check_interval_secs", 120))
            slot1_secs = float(cfg.get("slot1_refresh_secs", default_secs))
            slot2_secs = float(cfg.get("slot2_refresh_secs", default_secs))
        self._exc_slot1_refresh_var.set(self._seconds_to_refresh_label(slot1_secs))
        self._exc_slot2_refresh_var.set(self._seconds_to_refresh_label(slot2_secs))

    def _on_exc_slot_refresh_changed(self, slot_idx: int) -> None:
        cfg = self._get_alliance_monitor_cfg()
        slot1_secs = self._refresh_label_to_seconds(self._exc_slot1_refresh_var.get())
        slot2_secs = self._refresh_label_to_seconds(self._exc_slot2_refresh_var.get())
        cfg["slot_refresh_secs"] = [slot1_secs, slot2_secs]
        if slot_idx == 0:
            cfg["slot1_refresh_secs"] = slot1_secs
        else:
            cfg["slot2_refresh_secs"] = slot2_secs

        task = self._exc_monitor_task
        if task and task.is_enabled:
            task.update_slot_refresh_interval(0, slot1_secs)
            task.update_slot_refresh_interval(1, slot2_secs)
            self._log(f"专属任务刷新时间已更新：坑位1={slot1_secs // 60}分钟，坑位2={slot2_secs // 60}分钟（运行中已生效）")
        else:
            self._log(f"专属任务刷新时间已更新：坑位1={slot1_secs // 60}分钟，坑位2={slot2_secs // 60}分钟")

    def _get_bear_attack_teams_cfg(self) -> dict[int, list[str]]:
        cfg = self._get_bear_attack_cfg()
        raw_teams = cfg.get("teams", {})
        if not isinstance(raw_teams, dict):
            raw_teams = {}
            cfg["teams"] = raw_teams
        normalized: dict[int, list[str]] = {}
        for team_id in (1, 2, 3, 4):
            values = raw_teams.get(team_id, raw_teams.get(str(team_id), []))
            if not isinstance(values, list):
                values = []
            normalized[team_id] = [str(v) for v in values if isinstance(v, str)]
        cfg["teams"] = normalized
        return normalized

    def _collect_bear_team_template_options(self) -> list[str]:
        all_names: set[str] = set()

        def _collect(name: str):
            n = str(name).strip()
            if n and n.endswith("队伍"):
                all_names.add(n)

        teams_cfg = self._get_bear_attack_teams_cfg()
        for names in teams_cfg.values():
            for name in names:
                _collect(name)

        template_dir = Path.cwd() / "assets" / "templates"
        if template_dir.exists():
            for p in sorted(template_dir.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                _collect(p.stem)

        fixed_order: list[str] = (
            self._config.get("tasks", {})
            .get("bear_attack_task", {})
            .get("team_display_order", [])
        )
        options: list[str] = []
        seen: set[str] = set()
        for name in fixed_order:
            n = str(name).strip()
            if n in all_names and n not in seen:
                options.append(n)
                seen.add(n)
        for name in sorted(all_names):
            if name not in seen:
                options.append(name)
        return options

    @staticmethod
    def _build_step_spinner(
        parent: tk.Widget,
        var: tk.DoubleVar,
        step: float,
        min_val: float,
        max_val: float,
        on_change: Callable[[], None],
    ) -> None:
        """构建带 − / + 按钮的步进控件，步长为 step，值范围 [min_val, max_val]。"""
        def _dec():
            v = round(var.get() - step, 10)
            var.set(round(max(min_val, v), 1))
            on_change()

        def _inc():
            v = round(var.get() + step, 10)
            var.set(round(min(max_val, v), 1))
            on_change()

        ttk.Button(parent, text="−", width=2, command=_dec).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(parent, textvariable=var, width=4, anchor="center").pack(side=tk.LEFT, padx=1)
        ttk.Button(parent, text="+", width=2, command=_inc).pack(side=tk.LEFT)

    def _refresh_bear_team_templates(self) -> None:
        """重新扫描模板文件夹并刷新队伍模板选项，保留已选状态，同时清空图像识别缓存。"""
        # 保存各队伍当前已勾选的模板名
        current_selected: dict[int, set[str]] = {}
        for team_id, vars_dict in self._team_checkbox_vars.items():
            current_selected[team_id] = {name for name, var in vars_dict.items() if var.get()}

        # 清空图像识别缓存，让新图片立即生效
        from core.vision import clear_cache
        clear_cache()

        # 重新扫描模板目录
        self._bear_team_template_options = self._collect_bear_team_template_options()

        # 销毁旧的 Checkbutton 并重建
        for child in self._teams_frame.winfo_children():
            child.destroy()
        self._team_checkbox_vars.clear()

        for idx, team_id in enumerate((1, 2, 3, 4)):
            team_lf = ttk.LabelFrame(self._teams_frame, text=f"  队伍{team_id}  ", padding=4)
            team_lf.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0))

            self._team_checkbox_vars[team_id] = {}
            selected = current_selected.get(team_id, set())

            for name in self._bear_team_template_options:
                var = tk.BooleanVar(value=(name in selected))
                self._team_checkbox_vars[team_id][name] = var
                ttk.Checkbutton(
                    team_lf, text=name, variable=var,
                    command=lambda tid=team_id: self._on_bear_team_templates_changed(tid),
                ).pack(anchor="w", pady=1)

        self._log(f"打熊队伍模板已更新，共 {len(self._bear_team_template_options)} 个选项（已保留原有选择）")

    def _on_ba_delays_changed(self) -> None:
        d_group = round(self._ba_delay_group_var.get(), 1)
        d_exp = round(self._ba_delay_expedition_var.get(), 1)
        cfg = self._get_bear_attack_cfg()
        cfg["delay_before_click_group_secs"] = d_group
        cfg["delay_before_click_expedition_secs"] = d_exp
        task = self._bear_attack_task
        if task and task.is_enabled:
            task.update_click_delays(d_group, d_exp)
            self._log(f"打熊延时已更新（实时生效）：编队={d_group}s  出征={d_exp}s")
        else:
            self._log(f"打熊延时已更新：编队={d_group}s  出征={d_exp}s")

    def _on_ba_mode_changed(self) -> None:
        monster_mode = self._ba_mode_var.get() == "monster"
        task = self._bear_attack_task
        if task and task.is_enabled:
            task.set_monster_mode(monster_mode)
            mode_text = "打怪" if monster_mode else "打熊"
            self._log(f"打熊任务模式已切换为：{mode_text}（运行中立即生效）")

    def _on_bear_team_templates_changed(self, team_id: int) -> None:
        vars_dict = self._team_checkbox_vars.get(team_id)
        if vars_dict is None:
            return
        selected = [name for name, var in vars_dict.items() if var.get()]
        teams_cfg = self._get_bear_attack_teams_cfg()
        before = list(teams_cfg.get(team_id, []))
        teams_cfg[team_id] = selected
        self._get_bear_attack_cfg()["teams"] = teams_cfg
        self._log(f"打熊队伍配置变更：队伍{team_id} {before} -> {selected}")

        task = self._bear_attack_task
        if task and task.is_enabled:
            applied = task.force_refresh_teams()
            self._log(f"打熊运行中已立即应用队伍配置：{applied}")

    def _toggle_bear_attack(self):
        if self._bear_attack_task and self._bear_attack_task.is_enabled:
            # 停止
            self._bear_attack_task.is_enabled = False
            self._scheduler.remove_task(self._bear_attack_task)
            self._bear_attack_task = None
            self._ba_btn.configure(text="开 启")
            self._ba_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")
            self._log("打熊任务已停止")
            self._notify_event("打熊任务已停止")
        else:
            # 启动
            task = BearAttackTask(self._config)
            task.set_monster_mode(self._ba_mode_var.get() == "monster")
            self._bear_attack_task = task
            self._scheduler.add_task(task)
            self._ba_btn.configure(text="停 止")
            self._ba_state_lbl.configure(text="运行中…", foreground="#22aa55")
            mode_text = "打怪" if self._ba_mode_var.get() == "monster" else "打熊"
            self._log(f"打熊任务已启动（{mode_text}模式）")
            self._notify_event("打熊任务已启动")

    def _toggle_alliance_mobilization(self):
        if self._alliance_mob_task and self._alliance_mob_task.is_enabled:
            self._alliance_mob_task.is_enabled = False
            self._scheduler.remove_task(self._alliance_mob_task)
            self._alliance_mob_task = None
            self._am_btn.configure(text="开 启")
            self._am_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")
            self._log("联盟总动员任务已停止")
            self._notify_event("联盟总动员任务已停止")
        else:
            task = AllianceMobilizationTask(self._config)
            self._alliance_mob_task = task
            self._scheduler.add_task(task)
            self._am_btn.configure(text="停 止")
            self._am_state_lbl.configure(text="运行中…", foreground="#22aa55")
            self._log("联盟总动员任务已启动")
            self._notify_event("联盟总动员任务已启动")

    def set_feishu_notify(self, callback: Callable[[str], None]) -> None:
        """设置飞书通知回调，供监控任务使用。"""
        self._feishu_notify = callback

    def _stop_exc_monitor(self) -> None:
        if self._exc_monitor_task:
            self._exc_monitor_task.is_enabled = False
            self._scheduler.remove_task(self._exc_monitor_task)
            self._exc_monitor_task = None
        self._exc_btn.configure(text="开 启")
        self._exc_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

    def _stop_pub_monitor(self) -> None:
        if self._pub_monitor_task:
            self._pub_monitor_task.is_enabled = False
            self._scheduler.remove_task(self._pub_monitor_task)
            self._pub_monitor_task = None
        self._pub_btn.configure(text="开 启")
        self._pub_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

    def _toggle_exc_monitor(self):
        if self._exc_monitor_task and self._exc_monitor_task.is_enabled:
            self._stop_exc_monitor()
            self._log("专属任务刷新已停止")
            self._notify_event("专属任务刷新已停止")
        else:
            if not self._alliance_monitor_concurrent_mode():
                if self._pub_monitor_task and self._pub_monitor_task.is_enabled:
                    self._stop_pub_monitor()
                    self._log("公共任务抢已自动停止（与专属任务互斥）")
            task = AllianceMonitorTask(
                mode=AllianceMonitorTask.MODE_EXCLUSIVE,
                config=self._config,
                notify_callback=self._feishu_notify,
            )
            self._exc_monitor_task = task
            self._scheduler.add_task(task)
            self._exc_btn.configure(text="停 止")
            self._exc_state_lbl.configure(text="运行中…", foreground="#22aa55")
            self._log("专属任务刷新已启动")
            self._notify_event("专属任务刷新已启动")

    def _toggle_pub_monitor(self):
        if self._pub_monitor_task and self._pub_monitor_task.is_enabled:
            self._stop_pub_monitor()
            self._log("公共任务抢已停止")
            self._notify_event("公共任务抢已停止")
        else:
            if not self._alliance_monitor_concurrent_mode():
                if self._exc_monitor_task and self._exc_monitor_task.is_enabled:
                    self._stop_exc_monitor()
                    self._log("专属任务刷新已自动停止（与公共任务互斥）")
            skip_pub_scroll = (
                self._alliance_monitor_concurrent_mode()
                and self._exc_monitor_task is not None
                and self._exc_monitor_task.is_enabled
            )
            task = AllianceMonitorTask(
                mode=AllianceMonitorTask.MODE_PUBLIC,
                config=self._config,
                notify_callback=self._feishu_notify,
                skip_initial_pub_scroll=skip_pub_scroll,
            )
            self._pub_monitor_task = task
            self._scheduler.add_task(task)
            self._pub_btn.configure(text="停 止")
            self._pub_state_lbl.configure(text="运行中…", foreground="#22aa55")
            self._log("公共任务抢已启动")
            self._notify_event("公共任务抢已启动")

    def _stop_giant_beast_rally(self) -> None:
        if self._giant_beast_rally_task:
            self._giant_beast_rally_task.is_enabled = False
            self._scheduler.remove_task(self._giant_beast_rally_task)
            self._giant_beast_rally_task = None
        self._gbr_btn.configure(text="开 启")
        self._gbr_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

    def _toggle_giant_beast_rally(self):
        if self._giant_beast_rally_task and self._giant_beast_rally_task.is_enabled:
            self._stop_giant_beast_rally()
            self._log("集结巨兽任务已停止")
            self._notify_event("集结巨兽任务已停止")
        else:
            task = GiantBeastRallyTask(
                config=self._config,
                notify_callback=self._feishu_notify,
            )
            self._giant_beast_rally_task = task
            self._scheduler.add_task(task)
            self._gbr_btn.configure(text="停 止")
            self._gbr_state_lbl.configure(text="运行中…", foreground="#22aa55")
            self._log("集结巨兽任务已启动")
            self._notify_event("集结巨兽任务已启动")

    def _get_bear_attack_debug_cfg(self) -> dict:
        return self._config.setdefault("tasks", {}).setdefault("bear_attack_debug_task", {})

    def _run_bear_attack_debug(self) -> None:
        if not self._device.is_connected:
            self._log("打熊调试失败：设备未连接")
            return

        # 若上次调试任务仍在调度器中，先移除
        if self._bear_debug_task is not None:
            self._scheduler.remove_task(self._bear_debug_task)
            self._bear_debug_task = None

        # 把 UI 输入同步到 config
        cfg = self._get_bear_attack_debug_cfg()
        enter_pt = self._parse_manual_point(self._bad_enter_group_var.get())
        if enter_pt is None:
            self._log("打熊调试失败：「进入编队点」格式错误，请用 [x, y]")
            return
        group_pt = self._parse_manual_point(self._bad_group_var.get())
        if group_pt is None:
            self._log("打熊调试失败：「编队点」格式错误，请用 [x, y]")
            return

        def _parse_float(s: str, default: float) -> float | None:
            try:
                return float(s.strip())
            except Exception:
                return None

        d1 = _parse_float(self._bad_delay1_var.get(), 0.1)
        d2 = _parse_float(self._bad_delay2_var.get(), 0.1)
        d3 = _parse_float(self._bad_delay3_var.get(), 0.3)
        if d1 is None or d2 is None or d3 is None:
            self._log("打熊调试失败：延时字段必须是数字（秒）")
            return

        cfg["enter_group_point"] = list(enter_pt)
        cfg["group_point"] = list(group_pt)
        cfg["delay_before_click_group_secs"] = d1
        cfg["delay_before_click_expedition_secs"] = d2
        cfg["delay_after_click_expedition_check_secs"] = d3

        task = BearAttackDebugTask(self._config)
        self._bear_debug_task = task
        self._scheduler.add_task(task)
        self._bad_state_lbl.configure(text="执行中…", foreground="#22aa55")
        self._bad_btn.configure(state=tk.DISABLED)
        self._log(
            f"打熊调试已启动：进入编队点={list(enter_pt)}  编队点={list(group_pt)}"
            f"  延时 {d1}s / {d2}s / {d3}s"
        )

    def _clear_bear_attack_team_cooldown(self, team_id: int):
        task = self._bear_attack_task
        if not task or not task.is_enabled:
            self._log(f"清空队伍{team_id}冷却失败：打熊未运行")
            return
        had_cooldown = task.clear_team_cooldown(team_id)
        if had_cooldown:
            self._log(f"已清空打熊队伍{team_id}冷却")
        else:
            self._log(f"队伍{team_id}当前无冷却（已确保可用）")

    def _manual_go_back(self):
        if not self._device.is_connected:
            self._log("返回上一步失败：设备未连接")
            return
        frame = self._ctx.get_frame()
        if frame is None:
            self._log("返回上一步失败：当前无画面帧")
            return
        ok = try_go_back(
            frame=frame,
            device=self._device,
            vision=self._scheduler._vision,
            template_paths={},
            tag="手动",
        )
        self._log("已执行返回上一步" if ok else "返回上一步失败")

    def _manual_go_home(self):
        if not self._device.is_connected:
            self._log("返回首页失败：设备未连接")
            return

        # 防止重复触发多个返回首页线程
        if not self._manual_nav_lock.acquire(blocking=False):
            self._log("返回首页进行中，请稍后")
            return

        def _work():
            try:
                nav = HomeNavigator(
                    max_retries=self._config.get("tasks", {})
                    .get("red_packet_task", {})
                    .get("max_home_retries", 5),
                    tag="手动返回首页",
                )
                nav.reset()
                self._log("开始返回首页…")
                while True:
                    frame = self._ctx.get_frame()
                    if frame is None:
                        self._root.after(0, lambda: self._log("返回首页中断：当前无画面帧"))
                        return
                    template_paths = self._scheduler._build_template_paths(
                        ["世界地图", "城镇大门"]
                    )
                    status = nav.step(
                        frame,
                        self._device,
                        self._scheduler._vision,
                        template_paths,
                    )
                    if status == HomeNavigator.STATUS_HOME:
                        self._root.after(0, lambda: self._log("返回首页成功"))
                        return
                    if status == HomeNavigator.STATUS_FAILED:
                        self._root.after(0, lambda: self._log("返回首页失败：超过重试次数"))
                        return
                    time.sleep(0.35)
            finally:
                self._manual_nav_lock.release()

        threading.Thread(target=_work, daemon=True).start()

    @staticmethod
    def _parse_manual_point(point_text: str):
        text = point_text.strip()
        try:
            value = ast.literal_eval(text)
        except Exception:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return int(value[0]), int(value[1])
            except Exception:
                return None
        return None

    def _manual_swipe(self):
        if not self._device.is_connected:
            self._log("手动滑动失败：设备未连接")
            return

        start = self._parse_manual_point(self._manual_swipe_start_var.get())
        if start is None:
            self._log("手动滑动失败：起点格式错误，请用 [x, y]")
            return
        end = self._parse_manual_point(self._manual_swipe_end_var.get())
        if end is None:
            self._log("手动滑动失败：终点格式错误，请用 [x, y]")
            return

        try:
            duration_ms = max(1, int(self._manual_swipe_duration_var.get()))
        except Exception:
            self._log("手动滑动失败：时长必须是整数毫秒")
            return

        mode = self._manual_swipe_mode_var.get().strip().lower()
        x1, y1 = start
        x2, y2 = end

        def _work():
            if mode == "adb":
                ok = self._device.adb_swipe(x1, y1, x2, y2, duration_ms=duration_ms)
            else:
                ok = self._device.swipe(
                    x1,
                    y1,
                    x2,
                    y2,
                    duration=duration_ms / 1000.0,
                    blocking=True,
                )
            self._root.after(
                0,
                lambda: self._log(
                    f"手动滑动{'成功' if ok else '失败'}：{mode} {start} -> {end} {duration_ms}ms"
                ),
            )

        threading.Thread(target=_work, daemon=True).start()

    # ── 调度器开关 ─────────────────────────────────────────

    def _toggle_scheduler(self):
        if self._scheduler._running:
            self._scheduler.stop()
            self._sched_btn.configure(text="▶ 启动调度器")
            self._log("调度器已停止")
            self._notify_event("调度器已停止")
        else:
            self._scheduler.start()
            self._sched_btn.configure(text="■ 停止调度器")
            self._log("调度器已启动")
            self._notify_event("调度器已启动")

    # ── 状态刷新 ───────────────────────────────────────────

    def _schedule_refresh(self):
        self._refresh_status()
        self._root.after(self.REFRESH_MS, self._schedule_refresh)

    def _refresh_status(self):
        running = self._scheduler._running
        self._sched_running_lbl.configure(
            text="运行中" if running else "已停止",
            foreground="#22aa55" if running else "#cc4444",
        )
        self._sched_btn.configure(
            text="■ 停止调度器" if running else "▶ 启动调度器"
        )
        self._sched_task_lbl.configure(text=self._scheduler.current_task_name)
        self._sched_tick_lbl.configure(text=str(self._scheduler.tick_count))

        # 抢红包任务状态
        if self._red_packet_task and self._red_packet_task.is_enabled and not self._red_packet_task.is_finished:
            self._rp_state_lbl.configure(
                text=self._red_packet_task.state_label,
                foreground="#22aa55",
            )
        else:
            self._rp_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

        # 打熊任务状态
        if self._bear_attack_task and self._bear_attack_task.is_enabled and not self._bear_attack_task.is_finished:
            self._ba_state_lbl.configure(
                text=self._bear_attack_task.state_label,
                foreground="#22aa55",
            )
            cooldowns = self._bear_attack_task.get_team_cooldowns()
            for team_id, remaining in cooldowns.items():
                lbl = self._ba_cd_labels.get(team_id)
                if lbl is None:
                    continue
                if remaining > 0:
                    lbl.configure(text=f"{remaining:.0f}s", foreground="#e6a020")
                else:
                    lbl.configure(text="就绪", foreground="#22aa55")
        else:
            self._ba_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")
            for lbl in self._ba_cd_labels.values():
                lbl.configure(text="—", foreground="#888888")

        # 联盟总动员任务状态
        if self._alliance_mob_task and self._alliance_mob_task.is_enabled and not self._alliance_mob_task.is_finished:
            self._am_state_lbl.configure(
                text=self._alliance_mob_task.state_label,
                foreground="#22aa55",
            )
        else:
            self._am_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

        # 专属任务刷新状态
        if self._exc_monitor_task and self._exc_monitor_task.is_enabled and not self._exc_monitor_task.is_finished:
            self._exc_state_lbl.configure(
                text=self._exc_monitor_task.state_label,
                foreground="#22aa55",
            )
        else:
            self._exc_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

        # 公共任务抢状态（单次命中后任务自标记 is_finished，需同步 UI 引用）
        if self._pub_monitor_task is not None:
            if self._pub_monitor_task.is_finished:
                self._scheduler.remove_task(self._pub_monitor_task)
                self._pub_monitor_task = None
                self._pub_btn.configure(text="开 启")
                self._pub_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")
            elif self._pub_monitor_task.is_enabled:
                self._pub_state_lbl.configure(
                    text=self._pub_monitor_task.state_label,
                    foreground="#22aa55",
                )
            else:
                self._pub_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")
        else:
            self._pub_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

        # 集结巨兽任务状态
        if self._giant_beast_rally_task and self._giant_beast_rally_task.is_enabled:
            if self._giant_beast_rally_task.is_finished:
                self._gbr_state_lbl.configure(text="— 已完成 —", foreground="#888888")
                self._gbr_btn.configure(text="开 启")
                self._scheduler.remove_task(self._giant_beast_rally_task)
                self._giant_beast_rally_task = None
            else:
                self._gbr_state_lbl.configure(
                    text=self._giant_beast_rally_task.state_label,
                    foreground="#22aa55",
                )
        else:
            self._gbr_state_lbl.configure(text="— 已停止 —", foreground="#cc4444")

        # 打熊调试任务状态
        if self._bear_debug_task is not None:
            if self._bear_debug_task.is_finished:
                self._bad_state_lbl.configure(text="— 已完成 —", foreground="#888888")
                self._bad_btn.configure(state=tk.NORMAL)
                self._scheduler.remove_task(self._bear_debug_task)
                self._bear_debug_task = None
            else:
                self._bad_state_lbl.configure(
                    text=self._bear_debug_task.state_label, foreground="#22aa55"
                )

    # ── 日志 ───────────────────────────────────────────────

    def _log(self, msg: str):
        def _do():
            self._log_txt.configure(state=tk.NORMAL)
            ts = time.strftime("%H:%M:%S")
            self._log_txt.insert(tk.END, f"[{ts}] {msg}\n")
            self._log_txt.see(tk.END)
            self._log_txt.configure(state=tk.DISABLED)
        self._root.after(0, _do)

    def _clear_log(self):
        self._log_txt.configure(state=tk.NORMAL)
        self._log_txt.delete("1.0", tk.END)
        self._log_txt.configure(state=tk.DISABLED)

    def on_close(self):
        """主窗口关闭时调用"""
        self._stop_auto_connect("窗口关闭")
        self._scheduler.stop()
        self._device.disconnect()
        logging.getLogger().removeHandler(self._log_handler)

    def add_event_listener(self, listener: Callable[[str], None]) -> None:
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)

    def get_status_text(self) -> str:
        lines = [
            f"设备: {'已连接' if self._device.is_connected else '未连接'}",
            f"调度器: {'运行中' if self._scheduler._running else '已停止'}",
            f"当前任务: {self._scheduler.current_task_name}",
            f"Tick: {self._scheduler.tick_count}",
            f"抢红包: {self._rp_state_lbl.cget('text')}",
            f"打熊: {self._ba_state_lbl.cget('text')}",
            f"联盟总动员: {self._am_state_lbl.cget('text')}",
            f"专属任务刷新: {self._exc_state_lbl.cget('text')}",
            f"公共任务抢: {self._pub_state_lbl.cget('text')}",
            f"集结巨兽: {self._gbr_state_lbl.cget('text')}",
        ]
        return "\n".join(lines)

    def _check_and_reconnect_if_disconnected(self) -> str:
        """
        飞书远程开启任务前执行一次断线重连检查。
        规则：
        - 查找“重新连接”模板
        - 找到则点击
        - 点击后等待 2 秒再次检查
        - 第二次未找到，视为已恢复（或本就未断线）
        """
        if not self._device.is_connected:
            msg = "断线重连检查跳过：设备未连接。"
            self._log(msg)
            return msg

        template_name = "重新连接"
        template_paths = self._scheduler._build_template_paths([template_name])
        path = template_paths.get(template_name)
        if not path:
            msg = "断线重连检查：未找到“重新连接”模板，视为未断线。"
            self._log(msg)
            return msg

        frame = self._ctx.get_frame()
        if frame is None:
            msg = "断线重连检查跳过：当前无画面帧。"
            self._log(msg)
            return msg

        first_match = self._scheduler._vision.match(frame, path)
        if not first_match.found:
            msg = "断线重连检查：未检测到“重新连接”，当前未断线。"
            self._log(msg)
            return msg

        self._log(f"断线重连检查：检测到“重新连接”，点击 ({first_match.x}, {first_match.y})")
        self._device.click(first_match.x, first_match.y)
        time.sleep(2.0)

        frame_after = self._ctx.get_frame()
        if frame_after is None:
            msg = "断线重连检查完成：点击后当前无画面帧，继续执行任务启动。"
            self._log(msg)
            return msg

        second_match = self._scheduler._vision.match(frame_after, path)
        if second_match.found:
            msg = "断线重连检查：2秒后仍检测到“重新连接”，可能尚未恢复。"
            self._log(msg)
            return msg

        msg = "断线重连检查：2秒后未检测到“重新连接”，已恢复或未断线。"
        self._log(msg)
        return msg

    def handle_remote_command(self, text: str) -> str:
        cmd = (text or "").strip().lower().replace(" ", "")
        if not cmd:
            return "空指令，发送“帮助”查看可用命令。"
        if cmd in {"帮助", "help", "?"}:
            return (
                "可用命令：\n"
                "1) 状态\n"
                "2) 开启抢红包 / 停止抢红包\n"
                "3) 抢联盟红包 / 抢世界红包（切换红包模式）\n"
                "4) 开启打熊 / 停止打熊\n"
                "5) 开启联盟总动员 / 停止联盟总动员\n"
                "6) 开启专属任务刷新 / 停止专属任务刷新\n"
                "7) 开启公共任务抢 / 停止公共任务抢\n"
                "8) 开启集结巨兽 / 停止集结巨兽\n"
                "9) 开启调度器 / 停止调度器"
            )
        if cmd in {"状态", "status"}:
            return self.get_status_text()

        if cmd in {"抢联盟红包", "切换联盟红包", "联盟红包"}:
            if self._rp_mode_var:
                self._rp_mode_var.set("alliance")
            if self._red_packet_task and self._red_packet_task.is_enabled:
                self._red_packet_task.set_chat_mode("alliance")
            return "已处理：抢红包模式切换为联盟红包。"
        if cmd in {"抢世界红包", "切换世界红包", "世界红包"}:
            if self._rp_mode_var:
                self._rp_mode_var.set("world")
            if self._red_packet_task and self._red_packet_task.is_enabled:
                self._red_packet_task.set_chat_mode("world")
            return "已处理：抢红包模式切换为世界红包。"

        if cmd in {"开启抢红包", "启动抢红包"}:
            if not (self._red_packet_task and self._red_packet_task.is_enabled):
                reconnect_msg = self._check_and_reconnect_if_disconnected()
                self._toggle_red_packet()
                return f"{reconnect_msg}\n已处理：抢红包任务开启。"
            return "抢红包任务已在运行中。"
        if cmd in {"停止抢红包", "关闭抢红包"}:
            if self._red_packet_task and self._red_packet_task.is_enabled:
                self._toggle_red_packet()
            return "已处理：抢红包任务停止。"

        if cmd in {"开启打熊", "启动打熊"}:
            if not (self._bear_attack_task and self._bear_attack_task.is_enabled):
                reconnect_msg = self._check_and_reconnect_if_disconnected()
                self._toggle_bear_attack()
                return f"{reconnect_msg}\n已处理：打熊任务开启。"
            return "打熊任务已在运行中。"
        if cmd in {"停止打熊", "关闭打熊"}:
            if self._bear_attack_task and self._bear_attack_task.is_enabled:
                self._toggle_bear_attack()
            return "已处理：打熊任务停止。"

        if cmd in {"开启联盟总动员", "启动联盟总动员"}:
            if not (self._alliance_mob_task and self._alliance_mob_task.is_enabled):
                reconnect_msg = self._check_and_reconnect_if_disconnected()
                self._toggle_alliance_mobilization()
                return f"{reconnect_msg}\n已处理：联盟总动员任务开启。"
            return "联盟总动员任务已在运行中。"
        if cmd in {"停止联盟总动员", "关闭联盟总动员"}:
            if self._alliance_mob_task and self._alliance_mob_task.is_enabled:
                self._toggle_alliance_mobilization()
            return "已处理：联盟总动员任务停止。"

        if cmd in {"开启专属任务刷新", "启动专属任务刷新"}:
            if not (self._exc_monitor_task and self._exc_monitor_task.is_enabled):
                reconnect_msg = self._check_and_reconnect_if_disconnected()
                self._toggle_exc_monitor()
                return f"{reconnect_msg}\n已处理：专属任务刷新开启。"
            return "专属任务刷新已在运行中。"
        if cmd in {"停止专属任务刷新", "关闭专属任务刷新"}:
            if self._exc_monitor_task and self._exc_monitor_task.is_enabled:
                self._toggle_exc_monitor()
            return "已处理：专属任务刷新停止。"

        if cmd in {"开启公共任务抢", "启动公共任务抢"}:
            if not (self._pub_monitor_task and self._pub_monitor_task.is_enabled):
                reconnect_msg = self._check_and_reconnect_if_disconnected()
                self._toggle_pub_monitor()
                return f"{reconnect_msg}\n已处理：公共任务抢开启。"
            return "公共任务抢已在运行中。"
        if cmd in {"停止公共任务抢", "关闭公共任务抢"}:
            if self._pub_monitor_task and self._pub_monitor_task.is_enabled:
                self._toggle_pub_monitor()
            return "已处理：公共任务抢停止。"

        if cmd in {"开启集结巨兽", "启动集结巨兽"}:
            if not (self._giant_beast_rally_task and self._giant_beast_rally_task.is_enabled):
                reconnect_msg = self._check_and_reconnect_if_disconnected()
                self._toggle_giant_beast_rally()
                return f"{reconnect_msg}\n已处理：集结巨兽任务开启。"
            return "集结巨兽任务已在运行中。"
        if cmd in {"停止集结巨兽", "关闭集结巨兽"}:
            if self._giant_beast_rally_task and self._giant_beast_rally_task.is_enabled:
                self._toggle_giant_beast_rally()
            return "已处理：集结巨兽任务停止。"

        if cmd in {"开启调度器", "启动调度器"}:
            if not self._scheduler._running:
                self._toggle_scheduler()
            return "已处理：调度器开启。"
        if cmd in {"停止调度器", "关闭调度器"}:
            if self._scheduler._running:
                self._toggle_scheduler()
            return "已处理：调度器停止。"
        return f"未识别命令：{text}。发送“帮助”查看可用命令。"

    def _notify_event(self, message: str) -> None:
        for listener in list(self._event_listeners):
            try:
                listener(message)
            except Exception:
                logger.exception("事件监听器回调失败: %s", message)


class _TkLogHandler(logging.Handler):
    """将 logging 日志路由到控制面板日志框的处理器"""

    def __init__(self, log_func):
        super().__init__()
        self._log_func = log_func

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self._log_func(msg)
        except Exception:
            pass
