# -*- coding: utf-8 -*-
"""
AllianceMobilizationTask — 自动进入联盟总动员活动页面。

FSM 状态流：

    CHECK_ACTIVITY_PAGE   检查是否已在联盟总动员页面
      ├─ 找到"联盟总动员"           → DONE
      ├─ 未找到，找到首页标记        → FIND_REGULAR_ACTIVITY
      └─ 未找到，不在首页            → NAVIGATE_HOME

    NAVIGATE_HOME         借助 HomeNavigator 回到首页
      ├─ 成功                        → FIND_REGULAR_ACTIVITY
      └─ 失败                        → is_finished = True（打印错误）

    FIND_REGULAR_ACTIVITY 在首页查找"常规活动"并点击
      ├─ 找到并点击                  → 等待 delay_after_enter_secs → CHECK_DEFAULT_TAB
      └─ 未找到                      → CHECK_ACTIVITY_PAGE

    CHECK_DEFAULT_TAB     进入常规活动页后检查是否默认在联盟总动员
      ├─ 找到"联盟总动员"           → DONE
      └─ 未找到                      → SWIPE_AND_FIND_TAB（swipe_count 归零）

    SWIPE_AND_FIND_TAB    执行横向滑动，每次滑动后查找"联盟总动员tab"（最多 max_swipe_times 次）
      ├─ 滑动后找到"联盟总动员tab"  → 点击 → 等待 delay_after_tab_click_secs → VERIFY_AFTER_TAB
      └─ 达到最大滑动次数仍未找到   → CHECK_ACTIVITY_PAGE

    VERIFY_AFTER_TAB      点击 tab 后等待验证
      ├─ 找到"联盟总动员"           → DONE
      └─ 未找到                      → CHECK_ACTIVITY_PAGE

    DONE                  已成功进入联盟总动员，is_finished = True
"""
from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Tuple

import numpy as np

from core.device import Device
from core.vision import Vision
from engine.task_base import BaseTask
from logic.common_steps import HomeNavigator, find_first
from logic.task_config import merge_task_settings

logger = logging.getLogger(__name__)


# ── 默认配置常量 ──────────────────────────────────────────────────────────────

_DEFAULT_MAX_HOME_RETRIES: int = 5
_DEFAULT_DELAY_AFTER_ENTER_SECS: float = 0.5      # 点击常规活动后等待时长
_DEFAULT_DELAY_AFTER_TAB_CLICK_SECS: float = 0.5  # 点击联盟总动员tab后等待时长
_DEFAULT_MAX_SWIPE_TIMES: int = 2                  # 最大横向滑动次数
_DEFAULT_SWIPE_START_POINT: Tuple[int, int] = (571, 168)
_DEFAULT_SWIPE_END_POINT: Tuple[int, int] = (155, 172)
_DEFAULT_SWIPE_DURATION_SECS: float = 3.0


class State(Enum):
    CHECK_ACTIVITY_PAGE   = auto()  # 检查是否已在联盟总动员页
    NAVIGATE_HOME         = auto()  # 回到首页
    FIND_REGULAR_ACTIVITY = auto()  # 查找并点击"常规活动"
    CHECK_DEFAULT_TAB     = auto()  # 进入常规活动页后检查默认 tab
    SWIPE_AND_FIND_TAB    = auto()  # 滑动并查找联盟总动员 tab
    VERIFY_AFTER_TAB      = auto()  # 点击 tab 后验证
    DONE                  = auto()  # 已进入联盟总动员页面


class AllianceMobilizationTask(BaseTask):
    priority = 30
    name = "联盟总动员"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = merge_task_settings(config, "alliance_mobilization_task")

        _raw_tick_ms = cfg.get("tick_ms")
        if _raw_tick_ms is not None:
            self.tick_ms = int(_raw_tick_ms)

        self._delay_after_enter_secs: float = float(
            cfg.get("delay_after_enter_secs", _DEFAULT_DELAY_AFTER_ENTER_SECS)
        )
        self._delay_after_tab_click_secs: float = float(
            cfg.get("delay_after_tab_click_secs", _DEFAULT_DELAY_AFTER_TAB_CLICK_SECS)
        )
        self._max_swipe_times: int = max(
            1, int(cfg.get("max_swipe_times", _DEFAULT_MAX_SWIPE_TIMES))
        )
        self._swipe_start_point: Tuple[int, int] = tuple(
            cfg.get("swipe_start_point", list(_DEFAULT_SWIPE_START_POINT))
        )
        self._swipe_end_point: Tuple[int, int] = tuple(
            cfg.get("swipe_end_point", list(_DEFAULT_SWIPE_END_POINT))
        )
        self._swipe_duration_secs: float = float(
            cfg.get("swipe_duration_secs", _DEFAULT_SWIPE_DURATION_SECS)
        )

        dismiss_blank_pos: Tuple[int, int] = tuple(cfg.get("dismiss_blank_pos", [400, 1700]))

        self._home_nav = HomeNavigator(
            max_retries=cfg.get("max_home_retries", _DEFAULT_MAX_HOME_RETRIES),
            blank_pos=dismiss_blank_pos,
            tag="联盟总动员",
        )

        self.required_templates: List[str] = list(
            dict.fromkeys([
                "联盟总动员",
                "联盟总动员tab",
                "常规活动",
                "世界地图",
                "城镇大门",
            ])
        )

        self._state_templates: Dict[State, List[str]] = {
            State.CHECK_ACTIVITY_PAGE:   ["联盟总动员", "世界地图", "城镇大门"],
            State.NAVIGATE_HOME:         ["世界地图", "城镇大门"],
            State.FIND_REGULAR_ACTIVITY: ["常规活动"],
            State.CHECK_DEFAULT_TAB:     ["联盟总动员"],
            State.SWIPE_AND_FIND_TAB:    ["联盟总动员tab"],
            State.VERIFY_AFTER_TAB:      ["联盟总动员"],
            State.DONE:                  [],
        }

        # ── 运行时状态
        self._state: State = State.CHECK_ACTIVITY_PAGE
        self._wait_until: float = 0.0
        self._swipe_count: int = 0

    @property
    def step_template_names(self) -> List[str]:
        return self._state_templates.get(self._state, self.required_templates)

    def reset(self) -> None:
        super().reset()
        self._state = State.CHECK_ACTIVITY_PAGE
        self._wait_until = 0.0
        self._swipe_count = 0
        self._home_nav.reset()

    # ── Tick 入口 ─────────────────────────────────────────────────────────────

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

        logger.debug("[联盟总动员] 状态: %s", self._state.name)

        dispatch = {
            State.CHECK_ACTIVITY_PAGE:   self._step_check_activity_page,
            State.NAVIGATE_HOME:         self._step_navigate_home,
            State.FIND_REGULAR_ACTIVITY: self._step_find_regular_activity,
            State.CHECK_DEFAULT_TAB:     self._step_check_default_tab,
            State.SWIPE_AND_FIND_TAB:    self._step_swipe_and_find_tab,
            State.VERIFY_AFTER_TAB:      self._step_verify_after_tab,
            State.DONE:                  self._step_done,
        }
        dispatch[self._state](frame, device, vision, template_paths)

    # ── 各状态处理 ────────────────────────────────────────────────────────────

    def _step_check_activity_page(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """检查是否已在联盟总动员页面；若不在则判断是否处于首页。"""
        activity_path = template_paths.get("联盟总动员")
        if activity_path and vision.match(frame, activity_path).found:
            logger.info("[联盟总动员] 已在联盟总动员页面")
            self._go_to(State.DONE)
            return

        # 检查是否处于首页
        home_name, _ = find_first(frame, vision, template_paths, ["世界地图", "城镇大门"])
        if home_name:
            logger.info("[联盟总动员] 已在首页（%s），直接查找常规活动", home_name)
            self._go_to(State.FIND_REGULAR_ACTIVITY)
        else:
            logger.info("[联盟总动员] 不在联盟总动员页面也不在首页，准备回首页")
            self._home_nav.reset()
            self._go_to(State.NAVIGATE_HOME)

    def _step_navigate_home(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """借助 HomeNavigator 逐步回到首页。"""
        status = self._home_nav.step(frame, device, vision, template_paths)
        if status == HomeNavigator.STATUS_HOME:
            logger.info("[联盟总动员] 已回首页，开始查找常规活动")
            self._go_to(State.FIND_REGULAR_ACTIVITY)
        elif status == HomeNavigator.STATUS_FAILED:
            logger.error("[联盟总动员] 无法回到首页，任务退出")
            self.is_finished = True

    def _step_find_regular_activity(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """在首页查找"常规活动"并点击进入。"""
        path = template_paths.get("常规活动")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[联盟总动员] 点击常规活动 (%d, %d)", result.x, result.y)
                device.click(result.x, result.y)
                self._wait_until = time.monotonic() + self._delay_after_enter_secs
                self._go_to(State.CHECK_DEFAULT_TAB)
                return

        logger.debug("[联盟总动员] 未找到常规活动，重新检测页面状态")
        self._go_to(State.CHECK_ACTIVITY_PAGE)

    def _step_check_default_tab(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """进入常规活动页后，检查是否默认处于联盟总动员 tab。"""
        activity_path = template_paths.get("联盟总动员")
        if activity_path and vision.match(frame, activity_path).found:
            logger.info("[联盟总动员] 进入常规活动页后默认已在联盟总动员")
            self._go_to(State.DONE)
            return

        logger.info("[联盟总动员] 默认不在联盟总动员 tab，开始滑动查找")
        self._swipe_count = 0
        self._go_to(State.SWIPE_AND_FIND_TAB)

    def _step_swipe_and_find_tab(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """执行横向滑动，每次滑动后查找"联盟总动员tab"；最多滑动 max_swipe_times 次。"""
        if self._swipe_count >= self._max_swipe_times:
            logger.warning(
                "[联盟总动员] 已滑动 %d 次仍未找到联盟总动员tab，重新检测页面状态",
                self._swipe_count,
            )
            self._go_to(State.CHECK_ACTIVITY_PAGE)
            return

        x1, y1 = self._swipe_start_point
        x2, y2 = self._swipe_end_point
        logger.info(
            "[联盟总动员] 第 %d 次滑动 (%d,%d)→(%d,%d) 时长=%.1fs",
            self._swipe_count + 1, x1, y1, x2, y2, self._swipe_duration_secs,
        )
        device.swipe(
            int(x1), int(y1), int(x2), int(y2),
            duration=self._swipe_duration_secs,
            blocking=True,
        )
        self._swipe_count += 1

        # 滑动后立即查找联盟总动员tab
        tab_path = template_paths.get("联盟总动员tab")
        if tab_path:
            # 重新获取当前帧后匹配（swipe blocking 期间帧已更新）
            result = vision.match(frame, tab_path)
            if result.found:
                logger.info(
                    "[联盟总动员] 第 %d 次滑动后找到联盟总动员tab (%d, %d)，点击",
                    self._swipe_count, result.x, result.y,
                )
                device.click(result.x, result.y)
                self._wait_until = time.monotonic() + self._delay_after_tab_click_secs
                self._go_to(State.VERIFY_AFTER_TAB)
                return

        logger.debug("[联盟总动员] 第 %d 次滑动后未找到联盟总动员tab", self._swipe_count)
        # 下次 tick 继续（若未超次数则再滑动）

    def _step_verify_after_tab(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """点击联盟总动员 tab 后等待验证，确认已进入该活动页。"""
        activity_path = template_paths.get("联盟总动员")
        if activity_path and vision.match(frame, activity_path).found:
            logger.info("[联盟总动员] 点击 tab 后确认已进入联盟总动员页面")
            self._go_to(State.DONE)
            return

        logger.warning("[联盟总动员] 点击 tab 后未找到联盟总动员，重新检测")
        self._go_to(State.CHECK_ACTIVITY_PAGE)

    def _step_done(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """已成功进入联盟总动员页面，任务完成。"""
        logger.info("[联盟总动员] 已成功进入联盟总动员活动页面，任务完成")
        self.is_finished = True

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _go_to(self, state: State) -> None:
        logger.debug("[联盟总动员] %s → %s", self._state.name, state.name)
        self._state = state

    @property
    def state_label(self) -> str:
        labels = {
            State.CHECK_ACTIVITY_PAGE:   "检测活动页面",
            State.NAVIGATE_HOME:         "返回首页",
            State.FIND_REGULAR_ACTIVITY: "查找常规活动",
            State.CHECK_DEFAULT_TAB:     "检查默认Tab",
            State.SWIPE_AND_FIND_TAB:    "滑动查找Tab",
            State.VERIFY_AFTER_TAB:      "验证进入页面",
            State.DONE:                  "已进入活动页",
        }
        return labels.get(self._state, self._state.name)
