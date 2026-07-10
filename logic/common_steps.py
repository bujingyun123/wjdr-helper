# -*- coding: utf-8 -*-
"""
common_steps — 跨任务复用的通用原子操作

提供：
    find_first()      — 从模板列表中找到第一个匹配项
    try_close_popup() — 通用弹窗关闭（统一执行返回上一步）
    try_go_back()     — 通用返回上一步（ADB 返回键）
    HomeNavigator     — 非阻塞「回到首页」帮助器
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.device import Device
from core.shared_state import MatchResult
from core.vision import Vision

logger = logging.getLogger(__name__)


# ── 基础工具 ──────────────────────────────────────────────────────────────────

def find_first(
    frame: np.ndarray,
    vision: Vision,
    template_paths: dict,
    names: List[str],
    threshold: Optional[float] = None,
    use_grayscale: bool = True,
) -> Tuple[Optional[str], Optional[MatchResult]]:
    """
    在给定模板名称列表中，按顺序返回第一个匹配到的 (名称, MatchResult)。
    全部未匹配时返回 (None, None)。

    :param threshold: None 时使用 vision 的默认阈值
    :param use_grayscale: True 为灰度模板匹配（默认），False 为 BGR 匹配
    """
    for name in names:
        path = template_paths.get(name)
        if not path:
            continue
        result = vision.match(
            frame, path, threshold=threshold, use_grayscale=use_grayscale
        )
        if result.found:
            return name, result
    return None, None


# ── 通用弹窗关闭 ───────────────────────────────────────────────────────────────


def try_close_popup(
    frame: np.ndarray,
    device: Device,
    vision: Vision,
    template_paths: dict,
    tag: str = "",
) -> bool:
    """通用弹窗关闭步骤，直接调用 ADB 返回键。"""
    prefix = f"[{tag}] " if tag else ""
    logger.debug("%s执行返回上一步", prefix)
    return try_go_back(frame, device, vision, template_paths, tag=tag)


def try_go_back(
    frame: np.ndarray,
    device: Device,
    vision: Vision,
    template_paths: dict,
    tag: str = "",
) -> bool:
    """通用「返回上一步」步骤，直接调用 ADB 返回键。"""
    prefix = f"[{tag}] " if tag else ""
    ok = device.press_back()
    if ok:
        logger.info("%sADB 返回上一步", prefix)
    else:
        logger.warning("%sADB 返回上一步失败", prefix)
    return ok


# ── 通用状态 tick 重试器 ────────────────────────────────────────────────────────

class StateTickRetry:
    """
    通用“按状态 tick 次数重试”帮助器。

    适用于 FSM 任务里“状态切换后 UI 可能尚未渲染完成”的场景：
    - 在模板未命中时调用 try_retry(...)
    - 状态发生变化时调用 on_state_change(...)
    - 在任务 reset 时调用 reset()
    """

    def __init__(
        self,
        retry_ticks_by_state: Optional[Dict[str, int]] = None,
        skip_retry_on_first_entry_states: Optional[List[str]] = None,
    ) -> None:
        self._retry_ticks_by_state: Dict[str, int] = {}
        self._retry_count: int = 0
        self._state_enter_count: Dict[str, int] = {}
        self._skip_retry_on_first_entry_states: set[str] = set()
        self.configure(retry_ticks_by_state or {})
        self.configure_skip_first_entry(skip_retry_on_first_entry_states or [])

    def configure(self, retry_ticks_by_state: Dict[str, int]) -> None:
        """更新配置：{状态名: 最大重试 tick 次数}。"""
        parsed: Dict[str, int] = {}
        for state_name, retry_ticks in retry_ticks_by_state.items():
            try:
                parsed[str(state_name).upper()] = max(0, int(retry_ticks))
            except (TypeError, ValueError):
                logger.warning(
                    "[StateTickRetry] 无效配置: %r=%r，已忽略",
                    state_name, retry_ticks,
                )
        self._retry_ticks_by_state = parsed
        self._retry_count = 0

    def configure_skip_first_entry(self, states: List[str]) -> None:
        """配置首次进入这些状态时不走 tick 重试。"""
        self._skip_retry_on_first_entry_states = {
            str(state_name).upper() for state_name in states
        }

    def reset(self) -> None:
        self._retry_count = 0
        self._state_enter_count.clear()

    def on_state_change(self, changed: bool, state_name: str = "") -> None:
        """当状态发生切换时重置当前重试计数。"""
        if changed:
            self._retry_count = 0
            if state_name:
                normalized_state = str(state_name).upper()
                self._state_enter_count[normalized_state] = (
                    self._state_enter_count.get(normalized_state, 0) + 1
                )

    def try_retry(self, state_name: str, log_prefix: str = "", scene_desc: str = "") -> bool:
        """
        尝试为当前状态重试 1 个 tick。

        :return: True 表示继续留在当前状态等待下一 tick；False 表示重试次数已耗尽
        """
        normalized_state = str(state_name).upper()
        max_retries = self._retry_ticks_by_state.get(normalized_state, 0)
        if (
            normalized_state in self._skip_retry_on_first_entry_states
            and self._state_enter_count.get(normalized_state, 0) <= 1
        ):
            return False
        if self._retry_count < max_retries:
            self._retry_count += 1
            logger.debug(
                "%s%s未命中，tick 重试 (%d/%d)，状态=%s",
                f"[{log_prefix}] " if log_prefix else "",
                f"{scene_desc} " if scene_desc else "",
                self._retry_count,
                max_retries,
                normalized_state,
            )
            return True
        return False


# ── 回到首页帮助器 ─────────────────────────────────────────────────────────────
# ── 进入野外帮助器 ─────────────────────────────────────────────────────────────

class WildNavigator:
    """
    「进入野外（世界地图）」帮助器，每次调用 step() 只执行一个原子操作。

    逻辑：
        1. 检测「城镇大门」→ 已在野外（世界地图），返回 STATUS_WILD
        2. 检测「世界地图」→ 在城镇主界面，点击进入野外，返回 STATUS_ENTERING
        3. 超过 max_retries → 返回 STATUS_FAILED
        4. 否则执行一次 ADB 返回上一步，retry_count += 1，返回 STATUS_RETRYING
    """

    STATUS_WILD: str = "wild"
    STATUS_ENTERING: str = "entering"
    STATUS_RETRYING: str = "retrying"
    STATUS_FAILED: str = "failed"

    def __init__(
        self,
        max_retries: int = 5,
        tag: str = "进野外",
    ) -> None:
        self._max_retries = max_retries
        self._tag = tag
        self._retry_count: int = 0
        self._wait_until: float = 0.0
        self._wait_status: str = self.STATUS_RETRYING

    def reset(self) -> None:
        """重置重试计数，在每次开始新一轮导航前调用。"""
        self._retry_count = 0
        self._wait_until = 0.0

    def step(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: dict,
    ) -> str:
        """
        执行一步。返回 STATUS_WILD / STATUS_ENTERING / STATUS_RETRYING / STATUS_FAILED。
        """
        now = time.monotonic()

        # 正在等待画面过渡，直接返回等待状态，不触发任何检测
        if now < self._wait_until:
            return self._wait_status

        # 检测「城镇大门」→ 已在野外（世界地图视图）
        path = template_paths.get("城镇大门")
        if path and vision.match(frame, path).found:
            logger.info("[%s] 检测到「城镇大门」，已在野外", self._tag)
            self._retry_count = 0
            return self.STATUS_WILD

        # 检测「世界地图」→ 在城镇主界面，点击进入野外
        path = template_paths.get("世界地图")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[%s] 检测到「世界地图」按钮，点击进入野外", self._tag)
                device.click(result.x, result.y)
                self._wait_until = now + 1.5
                self._wait_status = self.STATUS_ENTERING
                return self.STATUS_ENTERING

        # 超过重试上限
        if self._retry_count >= self._max_retries:
            logger.error(
                "[%s] 尝试 %d 次后仍未进入野外，放弃",
                self._tag, self._max_retries,
            )
            return self.STATUS_FAILED

        # 执行一次返回操作
        self._retry_count += 1
        logger.debug(
            "[%s] 未找到野外标记，执行返回上一步 (%d/%d)",
            self._tag, self._retry_count, self._max_retries,
        )
        try_go_back(frame, device, vision, template_paths, tag=self._tag)
        self._wait_until = now + 1.0
        self._wait_status = self.STATUS_RETRYING
        return self.STATUS_RETRYING




class HomeNavigator:
    """
    非阻塞「回到首页」帮助器，每次调用 step() 只执行一个原子操作。

    使用方式：
        nav = HomeNavigator(max_retries=5)

        # 在 run_step() 中：
        status = nav.step(frame, device, vision, template_paths)
        if status == HomeNavigator.STATUS_HOME:
            # 已回到首页，继续下一阶段
        elif status == HomeNavigator.STATUS_FAILED:
            # 多次尝试均失败，任务应退出
        # STATUS_RETRYING 时什么都不做，等待下一个 tick

    逻辑：
        1. 检测首页模板（如「世界地图」「城镇大门」）→ 找到则返回 STATUS_HOME
        2. 超过 max_retries → 返回 STATUS_FAILED
        3. 否则执行一次 ADB 返回上一步，retry_count += 1，返回 STATUS_RETRYING
    """

    STATUS_HOME: str = "home"
    STATUS_RETRYING: str = "retrying"
    STATUS_FAILED: str = "failed"

    def __init__(
        self,
        max_retries: int = 5,
        home_template_names: Optional[List[str]] = None,
        blank_pos: Tuple[int, int] = (400, 1700),
        tag: str = "回首页",
        vision_match_threshold: Optional[float] = None,
        vision_use_grayscale: bool = True,
    ) -> None:
        self._max_retries = max_retries
        self._home_names: List[str] = home_template_names or ["世界地图", "城镇大门"]
        self._blank_pos = blank_pos
        self._tag = tag
        self._vision_match_threshold = vision_match_threshold
        self._vision_use_grayscale = vision_use_grayscale
        self._retry_count: int = 0
        self._wait_until: float = 0.0

    def reset(self) -> None:
        """重置重试计数，在每次开始新一轮导航前调用"""
        self._retry_count = 0
        self._wait_until = 0.0

    def step(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: dict,
    ) -> str:
        """
        执行一步。返回 STATUS_HOME / STATUS_RETRYING / STATUS_FAILED。
        """
        now = time.monotonic()

        # 正在等待画面过渡，直接返回等待状态
        if now < self._wait_until:
            return self.STATUS_RETRYING

        # 先检测是否已在首页
        home_name, home_result = find_first(
            frame,
            vision,
            template_paths,
            self._home_names,
            threshold=self._vision_match_threshold,
            use_grayscale=self._vision_use_grayscale,
        )
        if home_name and home_result:
            logger.info("[%s] 检测到首页标记「%s」，已回到首页", self._tag, home_name)
            self._retry_count = 0
            return self.STATUS_HOME

        # 超过重试上限
        if self._retry_count >= self._max_retries:
            logger.error(
                "[%s] 尝试 %d 次后仍未回到首页，放弃",
                self._tag, self._max_retries,
            )
            return self.STATUS_FAILED

        # 执行一次返回操作
        self._retry_count += 1
        logger.debug(
            "[%s] 尝试返回上一步回首页 (%d/%d)",
            self._tag, self._retry_count, self._max_retries,
        )
        try_go_back(frame, device, vision, template_paths, tag=self._tag)
        self._wait_until = now + 1.0
        return self.STATUS_RETRYING
