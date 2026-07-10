# -*- coding: utf-8 -*-
"""
GiantBeastRallyTask — 自动集结冰原巨兽。

流程：
1. 导航到野外（检测「城镇大门」可见）
2. 裁取 OCR 区域识别当前集结队伍数量
3. 数量 <= ocr_count_threshold → 发起集结；未识别到数字 → 检测是否在野外，是则发起集结，否则 1 分钟后重新导航
4. 每 check_interval_secs 秒检测一次
5. 集结点击序列（每步间隔 click_interval_secs 秒）：
   查找 → 冰原巨兽 → 搜索 → 集结 → 发起集结 → 编队 → 出征
6. 出征后等 verify_wait_secs 秒检测「城镇大门」：
   - 检测到 → 集结成功，发飞书通知，等 rally_cooldown_secs 秒后继续
   - 未检测到 → 可能体力不足，发飞书通知，1 分钟后继续下一轮
7. 每次集结距上次集结成功不得少于 rally_cooldown_secs 秒
"""
from __future__ import annotations

import logging
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.device import Device
from core.vision import Vision
from engine.task_base import BaseTask
from logic.common_steps import WildNavigator
from logic.task_config import merge_task_settings

logger = logging.getLogger(__name__)


# ── OCR 支持（优先 ddddocr，备选 pytesseract）────────────────────────────────
_ocr_engine = None
_OCR_AVAILABLE = False

try:
    import ddddocr as _dddd
    _ocr_engine = _dddd.DdddOcr(show_ad=False)
    try:
        _ocr_engine.set_ranges("0123456789")
    except Exception:
        pass
    _OCR_AVAILABLE = True
except Exception:
    try:
        import pytesseract as _tess
        _tess.get_tesseract_version()
        _OCR_AVAILABLE = True
    except Exception:
        pass


def _preprocess_ocr_region(region: np.ndarray, scale: float = 4.0) -> np.ndarray:
    """放大并二值化 OCR 小图，便于识别多位数字。"""
    h, w = region.shape[:2]
    new_w = max(int(w * scale), 48)
    new_h = max(int(h * scale), 24)
    scaled = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if scaled.ndim == 3 else scaled
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _ocr_number(region: np.ndarray) -> Tuple[Optional[int], str]:
    """对区域图像执行 OCR，返回 (整数, 原始识别文字)。"""
    if not _OCR_AVAILABLE or region is None or region.size == 0:
        return None, ""

    prepared = _preprocess_ocr_region(region)

    if _ocr_engine is not None:
        try:
            import io as _io
            import PIL.Image as _PILImage
            rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
            pil_img = _PILImage.fromarray(rgb)
            buf = _io.BytesIO()
            pil_img.save(buf, format="PNG")
            text = _ocr_engine.classification(buf.getvalue())
            digits = "".join(c for c in text if c.isdigit())
            return (int(digits) if digits else None), text
        except Exception:
            return None, ""

    try:
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        text = _tess.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        ).strip()
        digits = "".join(c for c in text if c.isdigit())
        return (int(digits) if digits else None), text
    except Exception:
        return None, ""


def _crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """安全裁剪帧图像。"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


# ── 默认配置 ──────────────────────────────────────────────────────────────────

_DEFAULT_CHECK_INTERVAL_SECS: float = 2.0
_DEFAULT_RALLY_COOLDOWN_SECS: float = 360.0      # 6 分钟
_DEFAULT_CLICK_INTERVAL_SECS: float = 1.0
_DEFAULT_VERIFY_WAIT_SECS: float = 1.0
_DEFAULT_OCR_COUNT_THRESHOLD: int = 3
# 需覆盖完整数字（含十位），过窄会把「11」裁成「1」
_DEFAULT_OCR_REGION: Tuple[int, int, int, int] = (805, 598, 872, 632)
_DEFAULT_FIND_BTN_POS: Tuple[int, int] = (59, 1431)
_DEFAULT_BEAST_BTN_POS: Tuple[int, int] = (295, 1473)
_DEFAULT_SEARCH_BTN_POS: Tuple[int, int] = (435, 1842)
_DEFAULT_RALLY_BTN_POS: Tuple[int, int] = (435, 798)
_DEFAULT_LAUNCH_RALLY_BTN_POS: Tuple[int, int] = (433, 1194)
_DEFAULT_TEAM_BTN_POS: Tuple[int, int] = (344, 146)
_DEFAULT_EXPEDITION_BTN_POS: Tuple[int, int] = (662, 1836)
_DEFAULT_MAX_HOME_RETRIES: int = 5


# ── 状态机 ────────────────────────────────────────────────────────────────────

class _State(Enum):
    NAVIGATE_TO_WILD = auto()   # 导航到野外
    CHECK_OCR        = auto()   # OCR 识别集结数量
    WAIT_INTERVAL    = auto()   # 等待检测间隔
    CLICK_FIND       = auto()   # 点击「查找」
    CLICK_BEAST      = auto()   # 点击「冰原巨兽」
    CLICK_SEARCH     = auto()   # 点击「搜索」
    CLICK_RALLY      = auto()   # 点击「集结」
    CLICK_LAUNCH     = auto()   # 点击「发起集结」
    CLICK_TEAM       = auto()   # 点击「编队」
    CLICK_EXPEDITION = auto()   # 点击「出征」
    VERIFY_SUCCESS   = auto()   # 验证集结是否成功
    WAIT_COOLDOWN    = auto()   # 成功后冷却等待


class GiantBeastRallyTask(BaseTask):
    """自动集结冰原巨兽任务。"""

    priority = 25
    name = "集结巨兽"

    def __init__(
        self,
        config: dict | None = None,
        notify_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__()
        self._notify_callback = notify_callback

        cfg = merge_task_settings(config, "giant_beast_rally_task")

        # 读取任务专属 tick 间隔（config 中配置了 500ms）
        _raw_tick_ms = cfg.get("tick_ms")
        if _raw_tick_ms is not None:
            self.tick_ms = int(_raw_tick_ms)

        self._check_interval_secs: float = float(
            cfg.get("check_interval_secs", _DEFAULT_CHECK_INTERVAL_SECS)
        )
        self._rally_cooldown_secs: float = float(
            cfg.get("rally_cooldown_secs", _DEFAULT_RALLY_COOLDOWN_SECS)
        )
        self._click_interval_secs: float = float(
            cfg.get("click_interval_secs", _DEFAULT_CLICK_INTERVAL_SECS)
        )
        self._verify_wait_secs: float = float(
            cfg.get("verify_wait_secs", _DEFAULT_VERIFY_WAIT_SECS)
        )
        self._ocr_count_threshold: int = int(
            cfg.get("ocr_count_threshold", _DEFAULT_OCR_COUNT_THRESHOLD)
        )

        ocr_r = cfg.get("ocr_region", list(_DEFAULT_OCR_REGION))
        self._ocr_region: Tuple[int, int, int, int] = (
            int(ocr_r[0]), int(ocr_r[1]), int(ocr_r[2]), int(ocr_r[3])
        )

        def _pos(key: str, default: Tuple[int, int]) -> Tuple[int, int]:
            v = cfg.get(key, list(default))
            return (int(v[0]), int(v[1]))

        self._find_btn_pos        = _pos("find_btn_pos",         _DEFAULT_FIND_BTN_POS)
        self._beast_btn_pos       = _pos("beast_btn_pos",        _DEFAULT_BEAST_BTN_POS)
        self._search_btn_pos      = _pos("search_btn_pos",       _DEFAULT_SEARCH_BTN_POS)
        self._rally_btn_pos       = _pos("rally_btn_pos",        _DEFAULT_RALLY_BTN_POS)
        self._launch_rally_btn_pos= _pos("launch_rally_btn_pos", _DEFAULT_LAUNCH_RALLY_BTN_POS)
        self._team_btn_pos        = _pos("team_btn_pos",         _DEFAULT_TEAM_BTN_POS)
        self._expedition_btn_pos  = _pos("expedition_btn_pos",   _DEFAULT_EXPEDITION_BTN_POS)

        self._wild_nav = WildNavigator(
            max_retries=int(cfg.get("max_home_retries", _DEFAULT_MAX_HOME_RETRIES)),
            tag=self.name,
        )

        self.required_templates: List[str] = ["城镇大门", "世界地图"]

        # 运行时状态
        self._state: _State = _State.NAVIGATE_TO_WILD
        self._wait_until: float = 0.0
        self._last_rally_at: float = 0.0
        self._last_ocr_roi_bytes: Optional[bytes] = None  # OCR ROI 缓存，像素未变化时跳过推理

        if not _OCR_AVAILABLE:
            logger.warning(
                "[%s] OCR 不可用（ddddocr/pytesseract 均未就绪），将视为未识别到数字直接发起集结",
                self.name,
            )

    # ── 属性 ──────────────────────────────────────────────────────────────────

    @property
    def step_template_names(self) -> List[str]:
        return ["城镇大门", "世界地图"]

    @property
    def state_label(self) -> str:
        labels: Dict[_State, str] = {
            _State.NAVIGATE_TO_WILD: "导航到野外",
            _State.CHECK_OCR:        "OCR检测数量",
            _State.WAIT_INTERVAL:    "等待检测间隔",
            _State.CLICK_FIND:       "点击查找",
            _State.CLICK_BEAST:      "点击冰原巨兽",
            _State.CLICK_SEARCH:     "点击搜索",
            _State.CLICK_RALLY:      "点击集结",
            _State.CLICK_LAUNCH:     "点击发起集结",
            _State.CLICK_TEAM:       "点击编队",
            _State.CLICK_EXPEDITION: "点击出征",
            _State.VERIFY_SUCCESS:   "验证集结成功",
            _State.WAIT_COOLDOWN:    "冷却等待中",
        }
        return labels.get(self._state, self._state.name)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._state = _State.NAVIGATE_TO_WILD
        self._wait_until = 0.0
        self._last_ocr_roi_bytes = None
        self._wild_nav.reset()

    # ── run_step ──────────────────────────────────────────────────────────────

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
            _State.NAVIGATE_TO_WILD: self._do_navigate_to_wild,
            _State.CHECK_OCR:        self._do_check_ocr,
            _State.WAIT_INTERVAL:    self._do_wait_interval,
            _State.CLICK_FIND:       self._do_click_find,
            _State.CLICK_BEAST:      self._do_click_beast,
            _State.CLICK_SEARCH:     self._do_click_search,
            _State.CLICK_RALLY:      self._do_click_rally,
            _State.CLICK_LAUNCH:     self._do_click_launch,
            _State.CLICK_TEAM:       self._do_click_team,
            _State.CLICK_EXPEDITION: self._do_click_expedition,
            _State.VERIFY_SUCCESS:   self._do_verify_success,
            _State.WAIT_COOLDOWN:    self._do_wait_cooldown,
        }
        dispatch[self._state](frame, device, vision, template_paths)

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _go_to(self, state: _State, delay: float = 0.0) -> None:
        self._state = state
        if delay > 0:
            self._wait_until = time.monotonic() + delay

    def _notify(self, msg: str) -> None:
        logger.info("[%s] %s", self.name, msg)
        if self._notify_callback:
            try:
                self._notify_callback(msg)
            except Exception:
                logger.exception("[%s] 飞书通知失败", self.name)

    # ── 状态处理 ──────────────────────────────────────────────────────────────

    def _do_navigate_to_wild(self, frame, device, vision, template_paths):
        status = self._wild_nav.step(frame, device, vision, template_paths)
        if status == WildNavigator.STATUS_WILD:
            self._go_to(_State.CHECK_OCR)
        elif status == WildNavigator.STATUS_FAILED:
            logger.error("[%s] 无法导航到野外，任务退出", self.name)
            self._notify("集结巨兽：无法导航到野外，任务退出")
            self.is_finished = True
        # STATUS_ENTERING / STATUS_RETRYING：等待下一 tick

    def _do_check_ocr(self, frame, device, vision, template_paths):
        x1, y1, x2, y2 = self._ocr_region
        roi = _crop(frame, x1, y1, x2, y2)

        # ROI 像素与上次完全相同时跳过 ONNX 推理，直接复用上次决策（进入等待间隔）
        roi_bytes = roi.tobytes()
        if roi_bytes == self._last_ocr_roi_bytes:
            logger.debug("[%s] OCR ROI 未变化，跳过推理，继续等待", self.name)
            self._go_to(_State.WAIT_INTERVAL)
            return
        self._last_ocr_roi_bytes = roi_bytes

        count, ocr_raw = _ocr_number(roi)

        crop_path_str = ""
        crops_dir = Path.cwd() / "logs" / "rally_ocr_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        crop_file = crops_dir / f"rally_{ts}.png"
        try:
            cv2.imwrite(str(crop_file), roi)
            crop_path_str = str(crop_file)
        except Exception:
            crop_path_str = "(保存失败)"

        logger.info(
            "[%s] 集结数量 OCR: 原文=%r 解析=%s（阈值 %d）截图=%s",
            self.name,
            ocr_raw or "(无)",
            count if count is not None else "未知",
            self._ocr_count_threshold,
            crop_path_str,
        )

        if count is None:
            # 未识别到数字，检测是否在野外
            gate_path = template_paths.get("城镇大门")
            in_wild = gate_path and vision.match(frame, gate_path).found
            if in_wild:
                logger.info("[%s] OCR 未识别到数字，但当前在野外，发起集结", self.name)
                self._go_to(_State.CLICK_FIND)
            else:
                logger.info("[%s] OCR 未识别到数字，且当前不在野外，1 分钟后继续", self.name)
                self._notify("[集结巨兽] OCR 未识别到数字且不在野外，1 分钟后继续")
                self._wild_nav.reset()
                self._go_to(_State.NAVIGATE_TO_WILD, delay=60.0)
        elif count <= self._ocr_count_threshold:
            logger.info(
                "[%s] 集结数量 %d（阈值 %d），发起集结",
                self.name, count, self._ocr_count_threshold,
            )
            self._go_to(_State.CLICK_FIND)
        else:
            logger.debug(
                "[%s] 集结数量 %d 已足够（阈值 %d），等待下次检测",
                self.name, count, self._ocr_count_threshold,
            )
            self._go_to(_State.WAIT_INTERVAL)

    def _do_wait_interval(self, frame, device, vision, template_paths):
        self._go_to(_State.CHECK_OCR, delay=self._check_interval_secs)

    def _do_click_find(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击查找按钮 %s", self.name, self._find_btn_pos)
        device.click(*self._find_btn_pos)
        self._go_to(_State.CLICK_BEAST, delay=self._click_interval_secs)

    def _do_click_beast(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击冰原巨兽 %s", self.name, self._beast_btn_pos)
        device.click(*self._beast_btn_pos)
        self._go_to(_State.CLICK_SEARCH, delay=self._click_interval_secs)

    def _do_click_search(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击搜索 %s", self.name, self._search_btn_pos)
        device.click(*self._search_btn_pos)
        self._go_to(_State.CLICK_RALLY, delay=self._click_interval_secs)

    def _do_click_rally(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击集结 %s", self.name, self._rally_btn_pos)
        device.click(*self._rally_btn_pos)
        self._go_to(_State.CLICK_LAUNCH, delay=self._click_interval_secs)

    def _do_click_launch(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击发起集结 %s", self.name, self._launch_rally_btn_pos)
        device.click(*self._launch_rally_btn_pos)
        self._go_to(_State.CLICK_TEAM, delay=self._click_interval_secs)

    def _do_click_team(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击编队 %s", self.name, self._team_btn_pos)
        device.click(*self._team_btn_pos)
        self._go_to(_State.CLICK_EXPEDITION, delay=self._click_interval_secs)

    def _do_click_expedition(self, frame, device, vision, template_paths):
        logger.info("[%s] 点击出征 %s", self.name, self._expedition_btn_pos)
        device.click(*self._expedition_btn_pos)
        self._go_to(_State.VERIFY_SUCCESS, delay=self._verify_wait_secs)

    def _do_verify_success(self, frame, device, vision, template_paths):
        path = template_paths.get("城镇大门")
        if path and vision.match(frame, path).found:
            self._last_rally_at = time.monotonic()
            logger.info("[%s] 集结成功！等待 %.0f 秒后继续", self.name, self._rally_cooldown_secs)
            self._notify(f"[集结巨兽] 集结成功！将在 {int(self._rally_cooldown_secs // 60)} 分钟后继续")
            self._go_to(_State.WAIT_COOLDOWN)
        else:
            logger.info("[%s] 未检测到「城镇大门」，可能体力不足，1 分钟后继续", self.name)
            self._notify("[集结巨兽] 集结失败（可能体力不足），1 分钟后继续")
            self._wild_nav.reset()
            self._go_to(_State.NAVIGATE_TO_WILD, delay=60.0)

    def _do_wait_cooldown(self, frame, device, vision, template_paths):
        logger.info(
            "[%s] 进入冷却等待，%.0f 秒后重新开始",
            self.name, self._rally_cooldown_secs,
        )
        self._wild_nav.reset()
        self._go_to(_State.NAVIGATE_TO_WILD, delay=self._rally_cooldown_secs)
