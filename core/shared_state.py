"""
SharedContext：主流程与 Tracker UI 之间的唯一通信桥梁。
持有最新帧、模板列表、匹配结果和当前任务激活的模板名称集合。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


@dataclass
class Template:
    """模板图像描述，与 scrcpy_tracker 中的 Template 保持字段兼容"""
    name: str
    path: str
    enabled: bool = True
    last_confidence: float = 0.0
    last_x: int = 0
    last_y: int = 0
    last_found: bool = False


@dataclass
class MatchResult:
    """单次模板匹配结果"""
    confidence: float
    x: int
    y: int
    w: int
    h: int
    found: bool


class SharedContext:
    """
    线程安全的共享状态容器。

    - 主流程（Device 线程）写入 current_frame
    - Scheduler 读取帧、写入 active_template_names
    - TrackerUI 读取帧、templates、active_template_names、match_results
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # 最新视频帧（BGR numpy array）
        self._current_frame: Optional[np.ndarray] = None

        # 所有已注册模板（tracker 与主流程共用同一份列表）
        self.templates: List[Template] = []
        self._templates_lock = threading.Lock()

        # 当前运行任务所需的模板名称集合（由 Scheduler 写入）
        self.active_template_names: Set[str] = set()

        # 最新一轮模板匹配结果（由 TrackerUI 或 Scheduler 写入）
        self.match_results: Dict[str, MatchResult] = {}
        self._results_lock = threading.Lock()

        # 供 UI 观察用的状态文本
        self.status_text: str = "就绪"

        # 设备连接后「自动进入无尽冬日」流程未完成前，调度器跳过任务 tick（避免在桌面误跑任务）
        self.scheduler_paused_for_game_autolaunch: bool = False

        # Tracker 可调参数（由 Tracker UI 写入，Vision 读取）
        self.vision_threshold: float = 0.72
        self.vision_scale: float = 1.0
        self.vision_method: str = "TM_CCOEFF_NORMED"

    # ── 帧读写 ────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._current_frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._current_frame

    # ── 模板列表 ──────────────────────────────────────────

    def get_templates(self) -> List[Template]:
        with self._templates_lock:
            return list(self.templates)

    def set_templates(self, templates: List[Template]) -> None:
        with self._templates_lock:
            self.templates = templates

    def sync_enabled_from_active(self) -> None:
        """根据 active_template_names 更新每个模板的 enabled 状态。
        若 active_template_names 为空则全部启用（不干预独立模式）。"""
        if not self.active_template_names:
            return
        with self._templates_lock:
            for t in self.templates:
                t.enabled = t.name in self.active_template_names

    # ── 匹配结果 ──────────────────────────────────────────

    def set_match_results(self, results: Dict[str, MatchResult]) -> None:
        with self._results_lock:
            self.match_results = results

    def get_match_results(self) -> Dict[str, MatchResult]:
        with self._results_lock:
            return dict(self.match_results)

    def get_result(self, name: str) -> Optional[MatchResult]:
        with self._results_lock:
            return self.match_results.get(name)
