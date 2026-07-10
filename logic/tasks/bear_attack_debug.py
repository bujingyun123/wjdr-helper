# -*- coding: utf-8 -*-
"""
BearAttackDebugTask — 打熊最小等待时间调试任务。

模拟"已找到队伍"之后的完整出征流程：
  1. 立即点击「进入编队」固定点（可在 UI 面板更改）
  2. 等待 delay_before_click_group_secs 秒
  3. 点击「编队」固定点（可在 UI 面板更改）
  4. 等待 delay_before_click_expedition_secs 秒
  5. 点击「出征」固定点（固定，不弹出 UI 修改）
  6. 等待 delay_after_click_expedition_check_secs 秒
  7. 检测「熊集结页」模板，命中则本次模拟成功

每次点击「执行调试」按钮均触发一次完整模拟；任务执行完毕后自动置 is_finished=True。
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
from logic.task_config import merge_task_settings

logger = logging.getLogger(__name__)

_DEFAULT_ENTER_GROUP_POINT: Tuple[int, int] = (766, 1448)
_DEFAULT_GROUP_POINT: Tuple[int, int] = (81, 146)
_DEFAULT_EXPEDITION_POINT: Tuple[int, int] = (662, 1848)
_DEFAULT_DELAY_BEFORE_CLICK_GROUP_SECS: float = 0.1
_DEFAULT_DELAY_BEFORE_CLICK_EXPEDITION_SECS: float = 0.1
_DEFAULT_DELAY_AFTER_CLICK_EXPEDITION_CHECK_SECS: float = 0.3
_DEFAULT_RALLY_TEMPLATE: str = "熊集结页"


class _State(Enum):
    CLICK_ENTER_GROUP = auto()
    CLICK_GROUP = auto()
    CLICK_EXPEDITION = auto()
    CHECK_RALLY = auto()


class BearAttackDebugTask(BaseTask):
    priority = 99
    name = "打熊调试"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self._config_ref: dict = config or {}
        cfg = merge_task_settings(self._config_ref, "bear_attack_debug_task")

        _raw_tick_ms = cfg.get("tick_ms")
        if _raw_tick_ms is not None:
            self.tick_ms = int(_raw_tick_ms)

        self._rally_template: str = cfg.get("rally_template", _DEFAULT_RALLY_TEMPLATE)
        self._enter_group_point: Tuple[int, int] = tuple(
            cfg.get("enter_group_point", list(_DEFAULT_ENTER_GROUP_POINT))
        )
        self._group_point: Tuple[int, int] = tuple(
            cfg.get("group_point", list(_DEFAULT_GROUP_POINT))
        )
        self._expedition_point: Tuple[int, int] = tuple(
            cfg.get("expedition_point", list(_DEFAULT_EXPEDITION_POINT))
        )
        self._delay_before_click_group_secs: float = float(
            cfg.get("delay_before_click_group_secs", _DEFAULT_DELAY_BEFORE_CLICK_GROUP_SECS)
        )
        self._delay_before_click_expedition_secs: float = float(
            cfg.get("delay_before_click_expedition_secs", _DEFAULT_DELAY_BEFORE_CLICK_EXPEDITION_SECS)
        )
        self._delay_after_click_expedition_check_secs: float = float(
            cfg.get(
                "delay_after_click_expedition_check_secs",
                _DEFAULT_DELAY_AFTER_CLICK_EXPEDITION_CHECK_SECS,
            )
        )

        self.required_templates: List[str] = [self._rally_template]
        self._state: _State = _State.CLICK_ENTER_GROUP
        self._wait_until: float = 0.0
        self._sim_index: int = 0

        self._dispatch = {
            _State.CLICK_ENTER_GROUP: self._step_click_enter_group,
            _State.CLICK_GROUP: self._step_click_group,
            _State.CLICK_EXPEDITION: self._step_click_expedition,
            _State.CHECK_RALLY: self._step_check_rally,
        }

    @property
    def step_template_names(self) -> List[str]:
        if self._state == _State.CHECK_RALLY:
            return [self._rally_template]
        return []

    def reset(self) -> None:
        super().reset()
        self._state = _State.CLICK_ENTER_GROUP
        self._wait_until = 0.0

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
        self._dispatch[self._state](frame, device, vision, template_paths)

    # ── 各步骤 ──────────────────────────────────────────────────────────────

    def _step_click_enter_group(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        self._sim_index += 1
        x, y = int(self._enter_group_point[0]), int(self._enter_group_point[1])
        logger.info(
            "[打熊调试] 第%d次模拟 ▶ 步骤1/3 点击「进入编队」固定点 (%d, %d)",
            self._sim_index, x, y,
        )
        device.click(x, y)
        delay = self._delay_before_click_group_secs
        logger.info(
            "[打熊调试] 步骤1完成，等待 %.3f s 后点击编队…", delay
        )
        self._wait_until = time.monotonic() + delay
        self._state = _State.CLICK_GROUP

    def _step_click_group(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        x, y = int(self._group_point[0]), int(self._group_point[1])
        logger.info(
            "[打熊调试] 步骤2/3 点击「编队」固定点 (%d, %d)", x, y
        )
        device.click(x, y)
        delay = self._delay_before_click_expedition_secs
        logger.info(
            "[打熊调试] 步骤2完成，等待 %.3f s 后点击出征…", delay
        )
        self._wait_until = time.monotonic() + delay
        self._state = _State.CLICK_EXPEDITION

    def _step_click_expedition(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        x, y = int(self._expedition_point[0]), int(self._expedition_point[1])
        logger.info(
            "[打熊调试] 步骤3/3 点击「出征」固定点 (%d, %d)", x, y
        )
        device.click(x, y)
        delay = self._delay_after_click_expedition_check_secs
        logger.info(
            "[打熊调试] 步骤3完成，等待 %.3f s 后检测集结页…", delay
        )
        self._wait_until = time.monotonic() + delay
        self._state = _State.CHECK_RALLY

    def _step_check_rally(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        path = template_paths.get(self._rally_template)
        if path and vision.match(frame, path).found:
            logger.info(
                "[打熊调试] ✓ 第%d次模拟成功：检测到「%s」，集结页已出现",
                self._sim_index, self._rally_template,
            )
        else:
            logger.info(
                "[打熊调试] ✗ 第%d次模拟完成：未检测到「%s」，"
                "可延长 delay_after_click_expedition_check_secs（当前 %.3f s）",
                self._sim_index, self._rally_template,
                self._delay_after_click_expedition_check_secs,
            )
        self.is_finished = True

    @property
    def state_label(self) -> str:
        labels = {
            _State.CLICK_ENTER_GROUP: "点击进入编队",
            _State.CLICK_GROUP: "点击编队",
            _State.CLICK_EXPEDITION: "点击出征",
            _State.CHECK_RALLY: "检测集结页",
        }
        return labels.get(self._state, self._state.name)
