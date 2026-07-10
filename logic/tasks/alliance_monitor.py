# -*- coding: utf-8 -*-
"""
AllianceMonitorTask — 联盟总动员活动页监控任务。

先导航到联盟总动员页面，之后根据 mode 进入对应监控循环：

  MODE_EXCLUSIVE（专属任务自动刷新）:
    - 滑动回顶部（ADB）
    - 每 check_interval_secs（默认 2 分钟）对两个坑位进行截图 + 模板匹配 + OCR 积分
    - 识别到"拳头"/"专精石"且积分 > score_threshold → 发飞书提醒

  模板匹配（tasks.alliance_monitor_task，可与 tasks.defaults 合并）：
    - pub_templates: 仅公共模式，全屏/条带内按顺序匹配的模板名列表，默认 ["拳头"]
    - vision_grayscale: 默认 true，灰度模板匹配；false 时使用 BGR 三通道匹配
    - vision_threshold: 可选，本任务专用匹配置信度阈值；省略则沿用 scrcpy_tracker.threshold / Tracker
    - 否则点击坑位中心，等 1 秒，查找"刷新任务"模板并点击
    - 坑位1 结束后等待 post_slot1_wait_secs（默认 1.5 秒）再检查坑位2
    - 冷却 refresh_cooldown_secs（默认 5 分钟）后进入下一轮

  MODE_PUBLIC（公共任务）:
    - 默认：滑动到公共任务区域（ADB）；与专属并行且 skip_initial_pub_scroll 时跳过该初始滑动
    - pub_templates：公共池要匹配的模板名列表，默认 ["拳头"]；按顺序匹配，先命中者优先
    - concurrent_exclusive_public 且未在 yaml 写 pub_fist_center_y_min 时：仅在 y > 1023 的条带内匹配（任意 pub_templates）
    - pub_auto_accept：true 时找到目标模板→点击→找「接受任务」点击；false 时仅飞书（不自动点击）
    - 命中目标并完成飞书后结束本任务（单次），不再循环查找

所有滑动通过 device.adb_swipe() 执行（后台线程，主循环不阻塞）。
识别事件及重要操作记录到 logs/alliance_monitor.log。
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.device import Device
from core.vision import Vision
from engine.task_base import BaseTask
from logic.common_steps import HomeNavigator, find_first
from logic.task_config import merge_task_settings

# ── OCR 支持（优先 ddddocr，备选 pytesseract + Tesseract-OCR）────────────────
_ocr_engine = None  # ddddocr 实例
_OCR_AVAILABLE = False

try:
    import ddddocr as _dddd
    _ocr_engine = _dddd.DdddOcr(show_ad=False)
    _OCR_AVAILABLE = True
except Exception:
    try:
        import pytesseract as _tess
        # 验证 tesseract 可执行文件是否真正可用
        _tess.get_tesseract_version()
        _OCR_AVAILABLE = True
    except Exception:
        pass

logger = logging.getLogger(__name__)


# ── 控制台 DEBUG handler（仅对本模块生效，不向上传播避免与根 handler 重复输出）─────────

def _ensure_debug_console() -> None:
    """给模块 logger 添加独立的 DEBUG 控制台 handler。
    
    只执行一次；设置 propagate=False 防止 INFO+ 消息在根 handler 中重复打印。
    飞书（Lark）logger 在 lark_oapi 内部已设为 WARNING，不受影响。
    """
    if logger.handlers:
        return
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


_ensure_debug_console()


# ── 文件日志 ──────────────────────────────────────────────────────────────────

def _get_file_logger() -> logging.Logger:
    fl = logging.getLogger("alliance_monitor.file")
    if fl.handlers:
        return fl
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(logs_dir / "alliance_monitor.log", encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    fl.addHandler(h)
    fl.setLevel(logging.DEBUG)
    fl.propagate = False
    return fl


# ── 常量 ──────────────────────────────────────────────────────────────────────

# 坑位定义：(x1, y1, x2, y2)，中心，积分区域
_SLOTS: List[Dict] = [
    {
        "region":       (51, 708, 418, 1004),
        "center":       (234, 856),
        "score_region": (221, 947, 366, 987),
        "label":        "坑位1",
    },
    {
        "region":       (438, 725, 793, 1001),
        "center":       (615, 863),
        "score_region": (605, 947, 750, 989),
        "label":        "坑位2",
    },
]

_DEFAULT_VALUABLE_TEMPLATES: List[str] = ["拳头", "专精石"]
_DEFAULT_PUB_TEMPLATES: List[str] = ["拳头"]
_DEFAULT_SCORE_THRESHOLD: int = 800
_DEFAULT_CHECK_INTERVAL_SECS: float = 120.0   # 2 分钟
_DEFAULT_REFRESH_COOLDOWN_SECS: float = 300.0  # 5 分钟
_DEFAULT_ACCEPT_WAIT_SECS: float = 0.3
_DEFAULT_CLICK_WAIT_SECS: float = 1.0
_DEFAULT_CONFIRM_REFRESH_WAIT_SECS: float = 0.5
_DEFAULT_POST_SLOT1_WAIT_SECS: float = 1.5  # 坑位1 结束后到坑位2 的间隔
_DEFAULT_POST_ROUND_COUNTDOWN_SECS: float = 2.0
_DEFAULT_VISION_GRAYSCALE: bool = True

# 专属任务滑动（回顶部）
_EXC_SCROLL_START = (420, 1271)
_EXC_SCROLL_END   = (432, 876)
_EXC_SCROLL_MS    = 2000

# 公共任务滑动（到公共池）
_PUB_SCROLL_START = (426, 1032)
_PUB_SCROLL_END   = (432, 753)
_PUB_SCROLL_MS    = 3000

# 导航配置默认值
_DEFAULT_MAX_HOME_RETRIES: int = 5
_DEFAULT_DELAY_AFTER_ENTER_SECS: float = 0.5
_DEFAULT_DELAY_AFTER_TAB_CLICK_SECS: float = 0.5
_DEFAULT_MAX_SWIPE_TIMES: int = 2
_DEFAULT_NAV_SWIPE_START: Tuple[int, int] = (571, 168)
_DEFAULT_NAV_SWIPE_END: Tuple[int, int] = (155, 172)
_DEFAULT_NAV_SWIPE_DURATION_SECS: float = 3.0
_DEFAULT_NAV_SWIPE_POST_WAIT_SECS: float = 5.0
_DEFAULT_SLOT_REFRESH_SECS: Tuple[float, float] = (
    _DEFAULT_CHECK_INTERVAL_SECS,
    _DEFAULT_CHECK_INTERVAL_SECS,
)


# ── OCR 辅助 ─────────────────────────────────────────────────────────────────

def _ocr_score(region: np.ndarray) -> Tuple[Optional[int], str]:
    """
    对积分区域图像执行 OCR，返回 (整数积分值, 原始识别文字)。
    积分识别失败时返回 (None, "")。
    region: BGR numpy 数组（小区域截图）

    优先使用 ddddocr（纯Python，无需外部可执行文件）；
    ddddocr 不可用时回落到 pytesseract。
    """
    if not _OCR_AVAILABLE or region is None or region.size == 0:
        return None, ""

    # ── ddddocr 路径 ──────────────────────────────────────────────────────────
    if _ocr_engine is not None:
        try:
            import PIL.Image as _PILImage
            import io as _io
            # ddddocr 接受 bytes 或 PIL Image；将 numpy BGR 转 PNG bytes
            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            pil_img = _PILImage.fromarray(rgb)
            buf = _io.BytesIO()
            pil_img.save(buf, format="PNG")
            text = _ocr_engine.classification(buf.getvalue())
            digits = "".join(c for c in text if c.isdigit())
            return (int(digits) if digits else None), text
        except Exception:
            return None, ""

    # ── pytesseract 回落路径 ──────────────────────────────────────────────────
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = _tess.image_to_string(
            binary,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        ).strip()
        digits = "".join(c for c in text if c.isdigit())
        return (int(digits) if digits else None), text
    except Exception:
        return None, ""


def _crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """安全裁剪 frame。"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def _coerce_template_name_list(raw: object, default: List[str]) -> List[str]:
    """将 yaml 中的字符串或列表规范为去空后的模板名列表；无效或全空时返回 default 的拷贝。"""
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        names = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, (list, tuple)):
        names = [str(x).strip() for x in raw if str(x).strip()]
    else:
        names = []
    return names if names else list(default)


# ── 状态机 ───────────────────────────────────────────────────────────────────

class _State(Enum):
    # 导航阶段
    CHECK_ACTIVITY_PAGE   = auto()
    NAVIGATE_HOME         = auto()
    FIND_REGULAR_ACTIVITY = auto()
    CHECK_DEFAULT_TAB     = auto()
    SWIPE_AND_FIND_TAB    = auto()
    VERIFY_AFTER_TAB      = auto()

    # 专属任务模式
    EXC_SCROLL_TOP      = auto()
    EXC_CHECK_SLOT1     = auto()
    EXC_FIND_REFRESH1   = auto()
    EXC_CONFIRM_REFRESH1= auto()
    EXC_POST_SLOT1_WAIT = auto()
    EXC_CHECK_SLOT2     = auto()
    EXC_FIND_REFRESH2   = auto()
    EXC_CONFIRM_REFRESH2= auto()
    EXC_COOLDOWN        = auto()

    # 公共任务模式
    PUB_SCROLL       = auto()
    PUB_FIND_FIST    = auto()
    PUB_ACCEPT       = auto()
    PUB_NOTIFY       = auto()


class AllianceMonitorTask(BaseTask):
    """联盟总动员活动页监控任务（专属刷新 / 公共抢任务）。"""

    MODE_EXCLUSIVE = "exclusive"
    MODE_PUBLIC    = "public"

    priority = 29

    def __init__(
        self,
        mode: str,
        config: dict | None = None,
        notify_callback: Optional[Callable[[str], None]] = None,
        skip_initial_pub_scroll: bool = False,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._notify_callback = notify_callback
        self._file_logger = _get_file_logger()

        self.name = "专属任务刷新" if mode == self.MODE_EXCLUSIVE else "公共任务抢"

        # ── 任务配置 ─────────────────────────────────────────
        cfg = merge_task_settings(config, "alliance_monitor_task")
        nav_cfg = merge_task_settings(config, "alliance_mobilization_task")
        raw_am: dict = {}
        _tasks_root = (config or {}).get("tasks")
        if isinstance(_tasks_root, dict):
            _am = _tasks_root.get("alliance_monitor_task")
            if isinstance(_am, dict):
                raw_am = _am

        _raw_tick_ms = cfg.get("tick_ms")
        if _raw_tick_ms is not None:
            self.tick_ms = int(_raw_tick_ms)

        self._score_threshold: int = int(cfg.get("score_threshold", _DEFAULT_SCORE_THRESHOLD))
        _raw_vision_threshold = cfg.get("vision_threshold")
        self._vision_threshold: Optional[float] = (
            float(_raw_vision_threshold) if _raw_vision_threshold is not None else None
        )
        self._vision_grayscale: bool = bool(
            cfg.get("vision_grayscale", _DEFAULT_VISION_GRAYSCALE)
        )
        self._valuable_templates: List[str] = list(
            cfg.get("valuable_templates", _DEFAULT_VALUABLE_TEMPLATES)
        )
        self._check_interval_secs: float = float(
            cfg.get("check_interval_secs", _DEFAULT_CHECK_INTERVAL_SECS)
        )
        raw_slot_refresh = cfg.get("slot_refresh_secs", [])
        if isinstance(raw_slot_refresh, (list, tuple)) and len(raw_slot_refresh) >= 2:
            self._slot_refresh_secs: List[float] = [
                max(1.0, float(raw_slot_refresh[0])),
                max(1.0, float(raw_slot_refresh[1])),
            ]
        else:
            slot1 = float(cfg.get("slot1_refresh_secs", self._check_interval_secs))
            slot2 = float(cfg.get("slot2_refresh_secs", self._check_interval_secs))
            self._slot_refresh_secs = [max(1.0, slot1), max(1.0, slot2)]
        self._refresh_cooldown_secs: float = float(
            cfg.get("refresh_cooldown_secs", _DEFAULT_REFRESH_COOLDOWN_SECS)
        )
        self._accept_wait_secs: float = float(
            cfg.get("accept_wait_secs", _DEFAULT_ACCEPT_WAIT_SECS)
        )
        self._click_wait_secs: float = float(
            cfg.get("click_wait_secs", _DEFAULT_CLICK_WAIT_SECS)
        )
        self._confirm_refresh_wait_secs: float = float(
            cfg.get("confirm_refresh_wait_secs", _DEFAULT_CONFIRM_REFRESH_WAIT_SECS)
        )
        self._post_slot1_wait_secs: float = float(
            cfg.get("post_slot1_wait_secs", _DEFAULT_POST_SLOT1_WAIT_SECS)
        )
        self._post_round_countdown_secs: float = float(
            cfg.get("post_round_countdown_secs", _DEFAULT_POST_ROUND_COUNTDOWN_SECS)
        )

        # 专属任务滑动参数（可配置覆盖）
        exc_scroll = cfg.get("exc_scroll_start", list(_EXC_SCROLL_START))
        self._exc_scroll_start: Tuple[int, int] = (int(exc_scroll[0]), int(exc_scroll[1]))
        exc_end = cfg.get("exc_scroll_end", list(_EXC_SCROLL_END))
        self._exc_scroll_end: Tuple[int, int] = (int(exc_end[0]), int(exc_end[1]))
        self._exc_scroll_ms: int = int(cfg.get("exc_scroll_ms", _EXC_SCROLL_MS))

        # 公共任务滑动参数（可配置覆盖）
        pub_scroll = cfg.get("pub_scroll_start", list(_PUB_SCROLL_START))
        self._pub_scroll_start: Tuple[int, int] = (int(pub_scroll[0]), int(pub_scroll[1]))
        pub_end = cfg.get("pub_scroll_end", list(_PUB_SCROLL_END))
        self._pub_scroll_end: Tuple[int, int] = (int(pub_end[0]), int(pub_end[1]))
        self._pub_scroll_ms: int = int(cfg.get("pub_scroll_ms", _PUB_SCROLL_MS))

        # 公共任务：自动抢 / 匹配中心 Y 下限（并发未显式配置时默认仅匹配中心 y > 1023）
        self._pub_concurrent_cfg: bool = False
        if mode == self.MODE_PUBLIC:
            self._pub_auto_accept: bool = bool(cfg.get("pub_auto_accept", True))
            self._pub_templates: List[str] = _coerce_template_name_list(
                cfg.get("pub_templates"), _DEFAULT_PUB_TEMPLATES
            )
            self._pub_concurrent_cfg = bool(cfg.get("concurrent_exclusive_public", False))
            _concurrent = self._pub_concurrent_cfg
            if "pub_fist_center_y_min" in raw_am:
                _py = raw_am.get("pub_fist_center_y_min")
                self._pub_fist_center_y_min: Optional[int] = None if _py is None else int(_py)
            elif _concurrent:
                self._pub_fist_center_y_min = 1023
            else:
                self._pub_fist_center_y_min = None
        else:
            self._pub_auto_accept = True
            self._pub_fist_center_y_min = None
            self._pub_templates = []

        # ── 导航配置 ─────────────────────────────────────────
        self._delay_after_enter_secs: float = float(
            nav_cfg.get("delay_after_enter_secs", _DEFAULT_DELAY_AFTER_ENTER_SECS)
        )
        self._delay_after_tab_click_secs: float = float(
            nav_cfg.get("delay_after_tab_click_secs", _DEFAULT_DELAY_AFTER_TAB_CLICK_SECS)
        )
        self._max_swipe_times: int = max(
            1, int(nav_cfg.get("max_swipe_times", _DEFAULT_MAX_SWIPE_TIMES))
        )
        nav_swipe_start = nav_cfg.get("swipe_start_point", list(_DEFAULT_NAV_SWIPE_START))
        self._nav_swipe_start: Tuple[int, int] = (int(nav_swipe_start[0]), int(nav_swipe_start[1]))
        nav_swipe_end = nav_cfg.get("swipe_end_point", list(_DEFAULT_NAV_SWIPE_END))
        self._nav_swipe_end: Tuple[int, int] = (int(nav_swipe_end[0]), int(nav_swipe_end[1]))
        self._nav_swipe_duration_secs: float = float(
            nav_cfg.get("swipe_duration_secs", _DEFAULT_NAV_SWIPE_DURATION_SECS)
        )
        self._nav_swipe_post_wait_secs: float = float(
            nav_cfg.get("swipe_post_wait_secs", _DEFAULT_NAV_SWIPE_POST_WAIT_SECS)
        )

        dismiss_blank_pos: Tuple[int, int] = tuple(
            nav_cfg.get("dismiss_blank_pos", [400, 1700])
        )

        self._home_nav = HomeNavigator(
            max_retries=nav_cfg.get("max_home_retries", _DEFAULT_MAX_HOME_RETRIES),
            blank_pos=dismiss_blank_pos,
            tag=self.name,
            vision_match_threshold=self._vision_threshold,
            vision_use_grayscale=self._vision_grayscale,
        )

        # ── 模板列表 ─────────────────────────────────────────
        nav_tpl = list(dict.fromkeys([
            "联盟总动员", "联盟总动员tab", "常规活动",
            "世界地图", "城镇大门",
        ]))
        if mode == self.MODE_EXCLUSIVE:
            work_tpl = list(dict.fromkeys([*self._valuable_templates, "刷新任务", "确认刷新任务", "重新连接"]))
        else:
            work_tpl = list(dict.fromkeys(
                [*self._pub_templates, *([] if not self._pub_auto_accept else ["接受任务"])]
            ))

        self.required_templates: List[str] = list(dict.fromkeys([*nav_tpl, *work_tpl]))

        self._nav_state_templates: Dict[_State, List[str]] = {
            _State.CHECK_ACTIVITY_PAGE:   ["联盟总动员", "世界地图", "城镇大门"],
            _State.NAVIGATE_HOME:         ["世界地图", "城镇大门"],
            _State.FIND_REGULAR_ACTIVITY: ["常规活动"],
            _State.CHECK_DEFAULT_TAB:     ["联盟总动员"],
            _State.SWIPE_AND_FIND_TAB:    ["联盟总动员tab"],
            _State.VERIFY_AFTER_TAB:      ["联盟总动员"],
        }

        # ── 运行时状态 ────────────────────────────────────────
        self._state: _State = _State.CHECK_ACTIVITY_PAGE
        self._wait_until: float = 0.0
        self._swipe_count: int = 0
        self._waiting_post_swipe_check: bool = False
        self._slot1_clicked: bool = False  # 本轮是否点击了坑位1（需要找刷新）
        self._slot2_clicked: bool = False
        self._slot1_found: bool = False   # 坑位1已识别到目标任务，后续不再检测/通知
        self._slot2_found: bool = False   # 坑位2已识别到目标任务，后续不再检测/通知
        self._slot_next_check_at: List[float] = [0.0, 0.0]
        self._slot_checked_in_round: List[bool] = [False, False]
        self._last_pub_task_pos: Tuple[int, int] = (0, 0)  # 公共模式：上次命中的目标坐标
        self._last_pub_task_template: str = ""  # 公共模式：命中的模板名
        self._skip_initial_pub_scroll: bool = bool(
            skip_initial_pub_scroll and mode == self.MODE_PUBLIC
        )
        self._pub_accept_clicked: bool = False  # 本轮是否已成功点击「接受任务」
        # 公共模式：周期性扫描诊断日志（避免每 tick 刷屏）
        self._last_pub_diag_log: float = 0.0
        self._pub_diag_log_interval: float = 2.5
        self._pub_missing_tpl_warned: set[str] = set()

        _vt_log = (
            f"{self._vision_threshold:.3f}"
            if self._vision_threshold is not None
            else "全局(scrcpy_tracker/Tracker)"
        )
        _mode_s = "灰度" if self._vision_grayscale else "BGR"
        if mode == self.MODE_PUBLIC:
            logger.info(
                "[%s] 公共匹配模板: %s，模板匹配: %s，threshold=%s",
                self.name,
                self._pub_templates,
                _mode_s,
                _vt_log,
            )
            self._file_logger.info(
                "公共匹配模板: %s | 模板匹配: %s | threshold=%s",
                self._pub_templates,
                _mode_s,
                _vt_log,
            )
        else:
            logger.info(
                "[%s] 目标任务列表: %s，积分阈值: %d，模板匹配: %s，threshold=%s",
                self.name,
                self._valuable_templates,
                self._score_threshold,
                _mode_s,
                _vt_log,
            )
            self._file_logger.info(
                "目标任务列表: %s | 积分阈值: %d | 模板匹配: %s | threshold=%s",
                self._valuable_templates,
                self._score_threshold,
                _mode_s,
                _vt_log,
            )

        if not _OCR_AVAILABLE:
            logger.warning("[%s] OCR 不可用（ddddocr/pytesseract 均未就绪），将跳过积分 OCR，仅依赖模板识别", self.name)
            self._file_logger.warning("OCR 功能不可用（ddddocr/pytesseract 均未就绪）")
        elif _ocr_engine is not None:
            logger.info("[%s] OCR 引擎：ddddocr", self.name)
        else:
            logger.info("[%s] OCR 引擎：pytesseract", self.name)

        if mode == self.MODE_PUBLIC:
            _ymin = self._pub_fist_center_y_min
            logger.info(
                "[%s] 公共：自动抢=%s，匹配中心 y 过滤=%s",
                self.name,
                self._pub_auto_accept,
                f"y>{_ymin}" if _ymin is not None else "无",
            )
            self._file_logger.info(
                "公共配置 | 自动抢=%s | 匹配中心y过滤=%s",
                self._pub_auto_accept,
                f"y>{_ymin}" if _ymin is not None else "无",
            )
            logger.info(
                "[%s] pub_templates 生效情况（merge_task_settings 后）: 原始字段=%r → 使用列表=%s | "
                "concurrent_exclusive_public=%s | skip_initial_pub_scroll=%s",
                self.name,
                cfg.get("pub_templates"),
                self._pub_templates,
                self._pub_concurrent_cfg,
                self._skip_initial_pub_scroll,
            )
            self._file_logger.info(
                "pub_templates 原始=%r | 使用=%s | 并发=%s | 跳过公共初滑=%s",
                cfg.get("pub_templates"),
                self._pub_templates,
                self._pub_concurrent_cfg,
                self._skip_initial_pub_scroll,
            )

    # ── step_template_names ───────────────────────────────────────────────────

    @property
    def step_template_names(self) -> List[str]:
        if self._state in self._nav_state_templates:
            return self._nav_state_templates[self._state]
        # 工作阶段：按模式返回所需模板
        if self._mode == self.MODE_EXCLUSIVE:
            return list(dict.fromkeys([*self._valuable_templates, "刷新任务", "确认刷新任务", "重新连接"]))
        if self._state == _State.PUB_ACCEPT and self._pub_auto_accept:
            return list(dict.fromkeys([*self._pub_templates, "接受任务"]))
        return list(self._pub_templates)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._state = _State.CHECK_ACTIVITY_PAGE
        self._wait_until = 0.0
        self._swipe_count = 0
        self._waiting_post_swipe_check = False
        self._slot1_clicked = False
        self._slot2_clicked = False
        self._slot1_found = False
        self._slot2_found = False
        self._slot_next_check_at = [0.0, 0.0]
        self._slot_checked_in_round = [False, False]
        self._home_nav.reset()
        self._pub_accept_clicked = False
        self._last_pub_task_pos = (0, 0)
        self._last_pub_task_template = ""
        self._last_pub_diag_log = 0.0
        self._pub_missing_tpl_warned.clear()

    def _vision_match(
        self,
        vision: Vision,
        frame: np.ndarray,
        template_path: str,
    ):
        """本任务配置的模板匹配（threshold / 灰度-BGR）。"""
        kw: Dict[str, object] = {}
        if self._vision_threshold is not None:
            kw["threshold"] = self._vision_threshold
        if not self._vision_grayscale:
            kw["use_grayscale"] = False
        return vision.match(frame, template_path, **kw)

    # ── run_step 入口 ─────────────────────────────────────────────────────────

    def run_step(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: Dict[str, str],
    ) -> None:
        now = time.monotonic()
        if now < self._wait_until:
            return

        logger.debug("[%s] 状态: %s", self.name, self._state.name)

        dispatch = {
            # 导航
            _State.CHECK_ACTIVITY_PAGE:   self._nav_check_activity_page,
            _State.NAVIGATE_HOME:         self._nav_navigate_home,
            _State.FIND_REGULAR_ACTIVITY: self._nav_find_regular_activity,
            _State.CHECK_DEFAULT_TAB:     self._nav_check_default_tab,
            _State.SWIPE_AND_FIND_TAB:    self._nav_swipe_and_find_tab,
            _State.VERIFY_AFTER_TAB:      self._nav_verify_after_tab,
            # 专属任务
            _State.EXC_SCROLL_TOP:         self._exc_scroll_top,
            _State.EXC_CHECK_SLOT1:        self._exc_check_slot1,
            _State.EXC_FIND_REFRESH1:      self._exc_find_refresh1,
            _State.EXC_CONFIRM_REFRESH1:   self._exc_confirm_refresh1,
            _State.EXC_POST_SLOT1_WAIT:    self._exc_post_slot1_wait,
            _State.EXC_CHECK_SLOT2:        self._exc_check_slot2,
            _State.EXC_FIND_REFRESH2:      self._exc_find_refresh2,
            _State.EXC_CONFIRM_REFRESH2:   self._exc_confirm_refresh2,
            _State.EXC_COOLDOWN:           self._exc_cooldown,
            # 公共任务
            _State.PUB_SCROLL:            self._pub_scroll,
            _State.PUB_FIND_FIST:         self._pub_find_fist,
            _State.PUB_ACCEPT:            self._pub_accept,
            _State.PUB_NOTIFY:            self._pub_notify,
        }
        dispatch[self._state](frame, device, vision, template_paths)

    # ─────────────────────────────────────────────────────────────────────────
    # 导航阶段（与 AllianceMobilizationTask 相同逻辑）
    # ─────────────────────────────────────────────────────────────────────────

    def _nav_check_activity_page(self, frame, device, vision, template_paths):
        path = template_paths.get("联盟总动员")
        if path and self._vision_match(vision, frame, path).found:
            logger.info("[%s] 已在联盟总动员页面，进入工作阶段", self.name)
            self._enter_work_mode()
            return
        home_name, _ = find_first(
            frame,
            vision,
            template_paths,
            ["世界地图", "城镇大门"],
            threshold=self._vision_threshold,
            use_grayscale=self._vision_grayscale,
        )
        if home_name:
            logger.info("[%s] 在首页，查找常规活动", self.name)
            self._go_to(_State.FIND_REGULAR_ACTIVITY)
        else:
            logger.info("[%s] 不在目标页，准备回首页", self.name)
            self._home_nav.reset()
            self._go_to(_State.NAVIGATE_HOME)

    def _nav_navigate_home(self, frame, device, vision, template_paths):
        status = self._home_nav.step(frame, device, vision, template_paths)
        if status == HomeNavigator.STATUS_HOME:
            self._go_to(_State.FIND_REGULAR_ACTIVITY)
        elif status == HomeNavigator.STATUS_FAILED:
            logger.error("[%s] 无法回首页，任务退出", self.name)
            self.is_finished = True

    def _nav_find_regular_activity(self, frame, device, vision, template_paths):
        path = template_paths.get("常规活动")
        if path:
            result = self._vision_match(vision, frame, path)
            if result.found:
                logger.info("[%s] 点击常规活动", self.name)
                device.click(result.x, result.y)
                self._wait_until = time.monotonic() + self._delay_after_enter_secs
                self._go_to(_State.CHECK_DEFAULT_TAB)
                return
        self._go_to(_State.CHECK_ACTIVITY_PAGE)

    def _nav_check_default_tab(self, frame, device, vision, template_paths):
        path = template_paths.get("联盟总动员")
        if path and self._vision_match(vision, frame, path).found:
            logger.info("[%s] 进入常规活动后已在联盟总动员，进入工作阶段", self.name)
            self._enter_work_mode()
            return
        self._swipe_count = 0
        self._waiting_post_swipe_check = False
        self._go_to(_State.SWIPE_AND_FIND_TAB)

    def _nav_swipe_and_find_tab(self, frame, device, vision, template_paths):
        tab_path = template_paths.get("联盟总动员tab")
        if self._waiting_post_swipe_check:
            self._waiting_post_swipe_check = False
            if tab_path:
                result = self._vision_match(vision, frame, tab_path)
                if result.found:
                    device.click(result.x, result.y)
                    self._wait_until = time.monotonic() + self._delay_after_tab_click_secs
                    self._go_to(_State.VERIFY_AFTER_TAB)
                    return
            logger.debug("[%s] 滑动后未找到联盟总动员tab，继续下一次滑动", self.name)
        elif tab_path:
            result = self._vision_match(vision, frame, tab_path)
            if result.found:
                device.click(result.x, result.y)
                self._wait_until = time.monotonic() + self._delay_after_tab_click_secs
                self._go_to(_State.VERIFY_AFTER_TAB)
                return

        if self._swipe_count >= self._max_swipe_times:
            logger.warning("[%s] 未找到联盟总动员tab，重新检测", self.name)
            self._go_to(_State.CHECK_ACTIVITY_PAGE)
            return

        x1, y1 = self._nav_swipe_start
        x2, y2 = self._nav_swipe_end
        duration_ms = int(self._nav_swipe_duration_secs * 1000)
        device.adb_swipe(x1, y1, x2, y2, duration_ms=duration_ms)
        self._swipe_count += 1
        self._waiting_post_swipe_check = True
        self._wait_until = time.monotonic() + self._nav_swipe_post_wait_secs

    def _nav_verify_after_tab(self, frame, device, vision, template_paths):
        path = template_paths.get("联盟总动员")
        if path and self._vision_match(vision, frame, path).found:
            logger.info("[%s] 点击 tab 后确认在联盟总动员页面，进入工作阶段", self.name)
            self._enter_work_mode()
            return
        self._go_to(_State.CHECK_ACTIVITY_PAGE)

    def _enter_work_mode(self) -> None:
        """导航完成后，进入对应模式的首个状态。"""
        if self._mode == self.MODE_EXCLUSIVE:
            self._slot1_clicked = False
            self._slot2_clicked = False
            self._slot_checked_in_round = [False, False]
            self._go_to(_State.EXC_SCROLL_TOP)
        elif self._skip_initial_pub_scroll:
            logger.info("[%s] 与专属并行，跳过公共池初始滑动，直接匹配公共模板", self.name)
            self._file_logger.info("与专属并行：跳过公共池初始滑动")
            self._go_to(_State.PUB_FIND_FIST)
        else:
            self._go_to(_State.PUB_SCROLL)

    # ─────────────────────────────────────────────────────────────────────────
    # 专属任务模式
    # ─────────────────────────────────────────────────────────────────────────

    def _exc_scroll_top(self, frame, device, vision, template_paths):
        """任务首次启动时 ADB 滑动回顶部，异步执行，等待完成后检查坑位。"""
        x1, y1 = self._exc_scroll_start
        x2, y2 = self._exc_scroll_end
        logger.info("[%s] 首次启动，ADB 滑动回顶部 (%d,%d)→(%d,%d) %dms", self.name, x1, y1, x2, y2, self._exc_scroll_ms)
        self._file_logger.info("首次启动：ADB 滑动回顶部")
        self._adb_swipe_async(device, x1, y1, x2, y2, self._exc_scroll_ms)
        self._wait_until = time.monotonic() + self._exc_scroll_ms / 1000.0 + 3.0
        self._go_to(_State.EXC_CHECK_SLOT1)

    def _exc_check_slot(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: Dict[str, str],
        slot_idx: int,
        next_state_if_click: _State,
        next_state_if_skip: _State,
        already_found: bool = False,
    ) -> None:
        """通用坑位检查逻辑。already_found=True 时跳过该坑位（已在本次任务中找到目标）。"""
        slot = _SLOTS[slot_idx]
        if already_found:
            logger.debug("[%s] %s 已找到目标任务，本次跳过", self.name, slot["label"])
            self._go_to(next_state_if_skip)
            return
        now = time.monotonic()
        if now < self._slot_next_check_at[slot_idx]:
            remain = self._slot_next_check_at[slot_idx] - now
            logger.debug("[%s] %s 未到刷新时间，剩余 %.1f 秒", self.name, slot["label"], remain)
            self._go_to(next_state_if_skip)
            return
        self._slot_checked_in_round[slot_idx] = True
        label = slot["label"]
        x1, y1, x2, y2 = slot["region"]
        cx, cy = slot["center"]
        sx1, sy1, sx2, sy2 = slot["score_region"]

        # 裁剪坑位区域和积分区域
        slot_img = _crop(frame, x1, y1, x2, y2)
        score_img = _crop(frame, sx1, sy1, sx2, sy2)

        # 模板匹配（在坑位区域内查找）
        matched_name: Optional[str] = None
        for tpl_name in self._valuable_templates:
            path = template_paths.get(tpl_name)
            if path and slot_img.size > 0:
                result = self._vision_match(vision, slot_img, path)
                if result.found:
                    matched_name = tpl_name
                    break

        # OCR 积分
        score, ocr_raw = _ocr_score(score_img)
        score_str = str(score) if score is not None else "未知"

        # OCR 未识别到数字时，先检查是否出现“重新连接”（断线）弹窗
        if score is None and self._check_disconnect_and_abort(frame, vision, template_paths):
            return

        # 保存积分截图到 logs/score_crops/
        crop_path_str = ""
        if score_img is not None and score_img.size > 0:
            crops_dir = Path.cwd() / "logs" / "score_crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            crop_file = crops_dir / f"{label}_{ts}.png"
            try:
                cv2.imwrite(str(crop_file), score_img)
                crop_path_str = str(crop_file)
            except Exception:
                crop_path_str = "(保存失败)"

        logger.info(
            "[%s] %s 匹配结果: %s，积分: %s（阈值 %d，目标任务: %s）",
            self.name, label, matched_name or "无", score_str,
            self._score_threshold, self._valuable_templates,
        )
        logger.debug(
            "[%s] %s OCR原文: %r  截图: %s",
            self.name, label, ocr_raw or "(无)", crop_path_str or "(未保存)",
        )
        self._file_logger.info(
            "%s 检查 | 模板=%s | 积分=%s | OCR原文=%r | 阈值=%d | 截图=%s",
            label, matched_name or "无", score_str, ocr_raw or "(无)",
            self._score_threshold, crop_path_str or "(未保存)",
        )

        # 判断是否是目标任务
        score_ok = (score is not None and score > self._score_threshold) or (
            score is None and not _OCR_AVAILABLE
        )
        if matched_name and score_ok:
            # 标记该坑位已找到，后续轮次不再检测/通知
            if slot_idx == 0:
                self._slot1_found = True
            else:
                self._slot2_found = True
            # 发飞书提醒（仅首次触达，already_found 分支已在入口处短路）
            msg = f"[联盟总动员] {label} 发现目标任务！模板={matched_name}，积分={score_str}"
            logger.info("[%s] %s", self.name, msg)
            self._file_logger.info("目标任务触发飞书提醒 | %s | 积分=%s", matched_name, score_str)
            self._notify(msg)
            # 不刷新，直接进入下一个坑位/结束
            self._go_to(next_state_if_skip)
        else:
            # 点击坑位中心，等 1s 后找刷新按钮
            logger.info("[%s] %s 非目标任务，点击中心 (%d,%d) 后查找刷新", self.name, label, cx, cy)
            self._file_logger.info("%s 点击坑位中心，准备刷新 | 积分=%s", label, score_str)
            device.click(cx, cy)
            if slot_idx == 0:
                self._slot1_clicked = True
            else:
                self._slot2_clicked = True
            self._wait_until = time.monotonic() + self._click_wait_secs
            self._go_to(next_state_if_click)

    def _check_disconnect_and_abort(
        self,
        frame: np.ndarray,
        vision: Vision,
        template_paths: Dict[str, str],
    ) -> bool:
        """检测是否出现“重新连接”弹窗；命中则中断任务。"""
        reconnect_path = template_paths.get("重新连接")
        if not reconnect_path:
            return False
        result = self._vision_match(vision, frame, reconnect_path)
        if not result.found:
            return False

        logger.warning("[%s] 检测到“重新连接”弹窗，判定为断线，任务中断", self.name)
        self._file_logger.warning("检测到重新连接弹窗，判定断线，停止任务")
        self._notify(f"[联盟总动员] 检测到断线（出现“重新连接”），已中断任务：{self.name}")
        self.is_finished = True
        return True

    def _exc_check_slot1(self, frame, device, vision, template_paths):
        self._exc_check_slot(
            frame, device, vision, template_paths,
            slot_idx=0,
            next_state_if_click=_State.EXC_FIND_REFRESH1,
            next_state_if_skip=_State.EXC_POST_SLOT1_WAIT,
            already_found=self._slot1_found,
        )

    def _exc_find_refresh(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: Dict[str, str],
        slot_idx: int,
        next_state: _State,
        confirm_state: _State,
    ) -> None:
        """
        在屏幕上查找"刷新任务"模板并点击。
        找到并点击 → 等 confirm_refresh_wait_secs 后进入 confirm_state 查找确认按钮。
        未找到 → 直接进入 next_state。
        """
        slot_label = _SLOTS[slot_idx]["label"]
        path = template_paths.get("刷新任务")
        if path:
            result = self._vision_match(vision, frame, path)
            if result.found:
                logger.info("[%s] 找到刷新任务按钮 (%d,%d)，点击", self.name, result.x, result.y)
                self._file_logger.info("%s 刷新任务 | 点击刷新按钮 (%d,%d)", slot_label, result.x, result.y)
                device.click(result.x, result.y)
                self._wait_until = time.monotonic() + self._confirm_refresh_wait_secs
                self._go_to(confirm_state)
                return
            else:
                logger.info("[%s] %s 未找到刷新任务按钮，跳过", self.name, slot_label)
                self._file_logger.info("%s 刷新任务 | 未找到刷新按钮，跳过", slot_label)
        else:
            logger.debug("[%s] 刷新任务模板未加载", self.name)
        self._go_to(next_state)

    def _exc_confirm_refresh(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: Dict[str, str],
        slot_idx: int,
        next_state: _State,
    ) -> None:
        """查找"确认刷新任务"模板并点击，然后进入下一状态。"""
        slot_label = _SLOTS[slot_idx]["label"]
        path = template_paths.get("确认刷新任务")
        if path:
            result = self._vision_match(vision, frame, path)
            if result.found:
                logger.info("[%s] 找到确认刷新任务按钮 (%d,%d)，点击", self.name, result.x, result.y)
                self._file_logger.info(
                    "%s 确认刷新 | 点击确认按钮 (%d,%d)，刷新完成",
                    slot_label, result.x, result.y,
                )
                device.click(result.x, result.y)
            else:
                logger.info("[%s] %s 未找到确认刷新任务按钮，跳过", self.name, slot_label)
                self._file_logger.info("%s 确认刷新 | 未找到确认按钮，跳过", slot_label)
        else:
            logger.debug("[%s] 确认刷新任务模板未加载", self.name)
        self._go_to(next_state)

    def _exc_find_refresh1(self, frame, device, vision, template_paths):
        self._exc_find_refresh(
            frame, device, vision, template_paths,
            slot_idx=0,
            next_state=_State.EXC_POST_SLOT1_WAIT,
            confirm_state=_State.EXC_CONFIRM_REFRESH1,
        )

    def _exc_confirm_refresh1(self, frame, device, vision, template_paths):
        self._exc_confirm_refresh(
            frame, device, vision, template_paths,
            slot_idx=0,
            next_state=_State.EXC_POST_SLOT1_WAIT,
        )

    def _exc_post_slot1_wait(self, frame, device, vision, template_paths) -> None:
        """坑位1 流程结束（含跳过/刷新完成）后等待再检查坑位2，避免 UI 未稳定。"""
        self._wait_until = time.monotonic() + self._post_slot1_wait_secs
        self._go_to(_State.EXC_CHECK_SLOT2)

    def _exc_check_slot2(self, frame, device, vision, template_paths):
        self._exc_check_slot(
            frame, device, vision, template_paths,
            slot_idx=1,
            next_state_if_click=_State.EXC_FIND_REFRESH2,
            next_state_if_skip=_State.EXC_COOLDOWN,
            already_found=self._slot2_found,
        )

    def _exc_find_refresh2(self, frame, device, vision, template_paths):
        self._exc_find_refresh(
            frame, device, vision, template_paths,
            slot_idx=1,
            next_state=_State.EXC_COOLDOWN,
            confirm_state=_State.EXC_CONFIRM_REFRESH2,
        )

    def _exc_confirm_refresh2(self, frame, device, vision, template_paths):
        self._exc_confirm_refresh(
            frame, device, vision, template_paths,
            slot_idx=1,
            next_state=_State.EXC_COOLDOWN,
        )

    def _exc_cooldown(self, frame, device, vision, template_paths):
        """本轮检查完毕后先等待 2 秒，再统一开始计时。"""
        now = time.monotonic()
        countdown = max(0.0, self._post_round_countdown_secs)
        for slot_idx in (0, 1):
            if self._slot_checked_in_round[slot_idx]:
                self._slot_next_check_at[slot_idx] = now + countdown + self._slot_refresh_secs[slot_idx]

        pending = []
        if not self._slot1_found:
            pending.append(max(0.0, self._slot_next_check_at[0] - now))
        if not self._slot2_found:
            pending.append(max(0.0, self._slot_next_check_at[1] - now))
        wait_secs = min(pending) if pending else self._check_interval_secs
        wait_secs = max(0.5, wait_secs)
        logger.info(
            "[%s] 本轮检查完毕，先等待 %.1f 秒再计时；下一轮等待 %.1f 秒",
            self.name,
            countdown,
            wait_secs,
        )
        self._file_logger.info(
            "本轮完毕，先等待 %.1f 秒再计时 | 下一轮等待 %.1f 秒",
            countdown,
            wait_secs,
        )
        self._wait_until = time.monotonic() + wait_secs
        self._slot1_clicked = False
        self._slot2_clicked = False
        self._slot_checked_in_round = [False, False]
        self._go_to(_State.EXC_CHECK_SLOT1)

    def update_slot_refresh_interval(self, slot_idx: int, seconds: float) -> None:
        """运行中更新坑位刷新间隔（slot_idx: 0/1）。"""
        if slot_idx not in (0, 1):
            return
        seconds = max(1.0, float(seconds))
        self._slot_refresh_secs[slot_idx] = seconds
        logger.info("[%s] %s 刷新间隔更新为 %.0f 秒", self.name, _SLOTS[slot_idx]["label"], seconds)
        self._file_logger.info("%s 刷新间隔更新为 %.0f 秒", _SLOTS[slot_idx]["label"], seconds)

    # ─────────────────────────────────────────────────────────────────────────
    # 公共任务模式
    # ─────────────────────────────────────────────────────────────────────────

    def _pub_scroll(self, frame, device, vision, template_paths):
        """ADB 滑动到公共任务区域。"""
        x1, y1 = self._pub_scroll_start
        x2, y2 = self._pub_scroll_end
        logger.info(
            "[%s] ADB 滑动到公共任务区域 (%d,%d)→(%d,%d) %dms，随后进入「寻找公共目标任务」",
            self.name, x1, y1, x2, y2, self._pub_scroll_ms,
        )
        self._file_logger.info("滑动到公共任务区域 → PUB_FIND_FIST")
        self._adb_swipe_async(device, x1, y1, x2, y2, self._pub_scroll_ms)
        self._wait_until = time.monotonic() + self._pub_scroll_ms / 1000.0 + 0.5
        self._go_to(_State.PUB_FIND_FIST)

    def _pub_find_fist(self, frame, device, vision, template_paths):
        """按 pub_templates 顺序在全屏（或 y 条带）内匹配；先命中者优先（见 pub_fist_center_y_min / 并发默认）。"""
        now = time.monotonic()
        fh, fw = frame.shape[:2]
        y0 = 0
        search = frame
        ymin = self._pub_fist_center_y_min
        if ymin is not None:
            y0 = min(max(ymin + 1, 0), max(0, frame.shape[0] - 1))
            search = frame[y0:, :]
        if search.size == 0 or search.shape[0] < 8 or search.shape[1] < 8:
            if now - self._last_pub_diag_log >= self._pub_diag_log_interval:
                self._last_pub_diag_log = now
                logger.warning(
                    "[%s] 公共扫描跳过：搜索 ROI 无效 | 全帧=%dx%d y0=%d ymin=%s search.shape=%s",
                    self.name, fw, fh, y0, ymin, getattr(search, "shape", None),
                )
                self._file_logger.warning(
                    "公共 ROI 无效 | frame=%dx%d y0=%s ymin=%s",
                    fw, fh, y0, ymin,
                )
            return

        thresh_used = (
            float(self._vision_threshold)
            if self._vision_threshold is not None
            else float(vision._active_threshold)
        )

        matched_name: Optional[str] = None
        result = None
        diag_parts: List[str] = []
        for tpl_name in self._pub_templates:
            path = template_paths.get(tpl_name)
            if not path:
                if tpl_name not in self._pub_missing_tpl_warned:
                    self._pub_missing_tpl_warned.add(tpl_name)
                    logger.warning(
                        "[%s] 公共模板「%s」无图像路径（未在 Tracker 注册且 assets/templates 下无 "
                        "%s.png/.jpg/.jpeg）。该模板无法参与匹配。",
                        self.name, tpl_name, tpl_name,
                    )
                    self._file_logger.warning("公共模板无路径，跳过: %s", tpl_name)
                diag_parts.append(f"{tpl_name}:无路径")
                continue
            r = self._vision_match(vision, search, path)
            hit = "命中" if r.found else f"未过阈(sim={r.confidence:.3f})"
            diag_parts.append(f"{tpl_name}:{hit}")
            if r.found:
                matched_name = tpl_name
                result = r
                break

        if not matched_name or result is None or not result.found:
            if now - self._last_pub_diag_log >= self._pub_diag_log_interval:
                self._last_pub_diag_log = now
                logger.info(
                    "[%s] 公共扫描中… | 全帧=%dx%d ROI y∈[%d,%d) 宽=%d | 阈值≥%.3f | %s",
                    self.name,
                    fw,
                    fh,
                    y0,
                    y0 + int(search.shape[0]),
                    int(search.shape[1]),
                    thresh_used,
                    "；".join(diag_parts) if diag_parts else "(pub_templates 为空)",
                )
                self._file_logger.info(
                    "公共扫描 | ROI y0=%d h=%d thr=%.3f | %s",
                    y0,
                    int(search.shape[0]),
                    thresh_used,
                    "；".join(diag_parts) if diag_parts else "-",
                )
            return

        gx, gy = int(result.x), int(result.y) + y0
        if ymin is not None and gy <= ymin:
            if now - self._last_pub_diag_log >= self._pub_diag_log_interval:
                self._last_pub_diag_log = now
                logger.info(
                    "[%s] 公共模板「%s」匹配到 sim=%.3f 但中心 y=%d 未满足 y>%d，忽略（避免点到专属区）",
                    self.name, matched_name, float(result.confidence), gy, ymin,
                )
                self._file_logger.info(
                    "公共 y 过滤丢弃 | tpl=%s y=%d need_y>%d sim=%.3f",
                    matched_name, gy, ymin, float(result.confidence),
                )
            return
        logger.info("[%s] 找到公共目标任务「%s」(%d,%d)", self.name, matched_name, gx, gy)
        self._file_logger.info("找到公共目标任务 %s (%d,%d)", matched_name, gx, gy)
        self._last_pub_task_pos = (gx, gy)
        self._last_pub_task_template = matched_name
        self._pub_accept_clicked = False
        if self._pub_auto_accept:
            logger.info(
                "[%s] 自动抢：点击「%s」(%d,%d) sim=%.3f，等待 %.2fs 后查找「接受任务」",
                self.name, matched_name, gx, gy, float(result.confidence), self._accept_wait_secs,
            )
            self._file_logger.info(
                "自动抢点击 | tpl=%s (%d,%d) sim=%.3f",
                matched_name, gx, gy, float(result.confidence),
            )
            device.click(gx, gy)
            self._wait_until = time.monotonic() + self._accept_wait_secs
            self._go_to(_State.PUB_ACCEPT)
        else:
            logger.info(
                "[%s] 未开启自动抢：不点击「%s」(%d,%d)，直接进入通知",
                self.name, matched_name, gx, gy,
            )
            self._file_logger.info("仅通知模式 | tpl=%s (%d,%d)", matched_name, gx, gy)
            self._go_to(_State.PUB_NOTIFY)

    def _pub_accept(self, frame, device, vision, template_paths):
        """查找「接受任务」模板并点击（仅自动抢模式）。"""
        path = template_paths.get("接受任务")
        if path:
            result = self._vision_match(vision, frame, path)
            if result.found:
                logger.info("[%s] 找到接受任务 (%d,%d)，点击", self.name, result.x, result.y)
                self._file_logger.info("找到接受任务按钮，点击 (%d,%d)", result.x, result.y)
                device.click(result.x, result.y)
                self._pub_accept_clicked = True
                self._go_to(_State.PUB_NOTIFY)
                return
            else:
                logger.info(
                    "[%s] 未找到接受任务按钮（当前最高相似度 %.3f，阈值≥%.3f），仍将发送飞书并结束本次公共查找",
                    self.name,
                    float(result.confidence),
                    float(self._vision_threshold)
                    if self._vision_threshold is not None
                    else float(vision._active_threshold),
                )
                self._file_logger.info(
                    "未找到接受任务 | sim=%.3f",
                    float(result.confidence),
                )
        else:
            logger.debug("[%s] 接受任务模板未加载", self.name)
        self._go_to(_State.PUB_NOTIFY)

    def _pub_notify(self, frame, device, vision, template_paths):
        """发送飞书通知，单次命中后结束公共任务。"""
        tx, ty = self._last_pub_task_pos
        tpl = self._last_pub_task_template or "未知模板"
        if self._pub_auto_accept and self._pub_accept_clicked:
            msg = f"[联盟总动员] 公共任务已接受！模板={tpl}，位置=({tx},{ty})"
        elif self._pub_auto_accept:
            msg = (
                f"[联盟总动员] 公共任务已发现「{tpl}」并已点击 ({tx},{ty})，"
                f"但未找到「接受任务」按钮，请手动处理。"
            )
        else:
            msg = (
                f"[联盟总动员] 公共任务发现「{tpl}」（未开启自动抢，未点击），坐标=({tx},{ty})，"
                f"请手动接取。"
            )
        logger.info("[%s] %s", self.name, msg)
        self._file_logger.info("公共任务单次完成 | %s", msg)
        self._notify(msg)
        logger.info("[%s] 公共任务查找已结束（单次命中）", self.name)
        self._file_logger.info("公共任务查找已结束（单次命中）")
        self.is_finished = True

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def _go_to(self, state: _State) -> None:
        logger.debug("[%s] %s → %s", self.name, self._state.name, state.name)
        self._state = state

    def _notify(self, message: str) -> None:
        """调用外部通知回调（通常路由到飞书）。"""
        if self._notify_callback:
            try:
                self._notify_callback(message)
            except Exception:
                logger.exception("[%s] 通知回调异常", self.name)

    @staticmethod
    def _adb_swipe_async(
        device: Device,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int,
    ) -> None:
        """在后台线程执行 ADB 滑动，不阻塞调度器主线程。"""
        def _worker():
            device.adb_swipe(x1, y1, x2, y2, duration_ms=duration_ms)
        threading.Thread(target=_worker, daemon=True, name="adb-swipe").start()

    @property
    def state_label(self) -> str:
        labels: Dict[_State, str] = {
            _State.CHECK_ACTIVITY_PAGE:   "检测活动页面",
            _State.NAVIGATE_HOME:         "返回首页",
            _State.FIND_REGULAR_ACTIVITY: "查找常规活动",
            _State.CHECK_DEFAULT_TAB:     "检查默认Tab",
            _State.SWIPE_AND_FIND_TAB:    "滑动查找Tab",
            _State.VERIFY_AFTER_TAB:      "验证进入页面",
            _State.EXC_SCROLL_TOP:         "滑动回顶部",
            _State.EXC_CHECK_SLOT1:        "检查坑位1",
            _State.EXC_FIND_REFRESH1:      "查找刷新1",
            _State.EXC_CONFIRM_REFRESH1:   "确认刷新1",
            _State.EXC_POST_SLOT1_WAIT:    "坑位1后等待",
            _State.EXC_CHECK_SLOT2:        "检查坑位2",
            _State.EXC_FIND_REFRESH2:      "查找刷新2",
            _State.EXC_CONFIRM_REFRESH2:   "确认刷新2",
            _State.EXC_COOLDOWN:           "冷却等待中",
            _State.PUB_SCROLL:            "滑动公共池",
            _State.PUB_FIND_FIST:         "寻找公共目标任务",
            _State.PUB_ACCEPT:            "点击接受任务",
            _State.PUB_NOTIFY:            "发送飞书并结束",
        }
        return labels.get(self._state, self._state.name)
