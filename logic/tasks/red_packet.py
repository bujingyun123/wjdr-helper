# -*- coding: utf-8 -*-
"""
RedPacketTask — 自动抢红包任务（重构版）

支持两种模式：
  - 世界模式：在"世界聊天"中抢"世界红包"，入口固定点 [159, 139]，开启固定点 [438, 1500]
  - 联盟模式：在"联盟聊天"中抢"联盟红包"，入口固定点 [435, 142]，开启固定点 [428, 1215]

FSM 状态流：

    NAVIGATE_HOME   首次开启 / 重置时先回首页（兼做断线后兜底）
      ├─ 成功（首次初始化）                → ENTER_HOME_CHAT
      └─ 成功（普通失败重试）              → ENTER_CHAT

    ENTER_HOME_CHAT 从首页点击固定点 [263, 1747] 进入聊天，等待 1s → CHECK_CHAT

    FIND_RED_PACKET 按设定扫描间隔扫描目标红包（阈值 0.9）
      ├─ 找到红包：点击红包 → 等待 click_delay → CLICK_OPEN
      ├─ 未找到：等待 scan_interval 后下次扫描
      └─ 聊天界面消失：→ CHECK_CHAT

    CLICK_OPEN      click_delay 后点击开启固定坐标 → 等待 0.5s → VERIFY_CHAT

    VERIFY_CHAT     验证仍在聊天界面（最多 2 次按返回重试）
      ├─ 检测到聊天：快速滑动到底部 → 发飞书通知 → CHECK_CHAT
      ├─ 未检测到（第 1/2 次）             → 按返回键，等待 0.5s，再次检测
      └─ 未检测到（第 3 次）               → NAVIGATE_HOME

    CHECK_CHAT      抢完红包后 / 首次进入后检测是否在目标聊天界面
      ├─ 检测到目标聊天：滑动到底部        → FIND_RED_PACKET
      └─ 未检测到（连续失败达上限）        → NAVIGATE_HOME
      └─ 未检测到（未达上限）              → ENTER_CHAT

    ENTER_CHAT      点击固定入口坐标进入聊天界面，然后等待后回 CHECK_CHAT
"""
from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.device import Device
from core.vision import Vision
from engine.task_base import BaseTask
from logic.common_steps import HomeNavigator, try_go_back
from logic.task_config import merge_task_settings

logger = logging.getLogger(__name__)


# ── 默认配置常量（作为 config.yaml 未配置时的兜底值）────────────────────────

_DEFAULT_SCAN_INTERVAL_SECS: float = 1.0
_DEFAULT_CLICK_DELAY_SECS: float = 0.3
_DEFAULT_OPEN_AFTER_CLICK_DELAY_SECS: float = 1.0
_DEFAULT_VERIFY_WAIT_SECS: float = 0.5
_DEFAULT_ENTER_CHAT_WAIT_SECS: float = 0.8
_DEFAULT_ENTER_HOME_CHAT_WAIT_SECS: float = 1.0
_DEFAULT_MAX_HOME_RETRIES: int = 5
_DEFAULT_MAX_CHECK_CHAT_RETRIES: int = 3
_DEFAULT_DISMISS_BLANK_POS: Tuple[int, int] = (400, 1700)
_DEFAULT_RED_PACKET_THRESHOLD: float = 0.9
_DEFAULT_CHAT_CHECK_INTERVAL_SECS: float = 60.0

_DEFAULT_WORLD_CHAT_ENTRY_POS: Tuple[int, int] = (159, 139)
_DEFAULT_ALLIANCE_CHAT_ENTRY_POS: Tuple[int, int] = (435, 142)
_DEFAULT_HOME_TO_CHAT_POS: Tuple[int, int] = (263, 1747)
_DEFAULT_WORLD_OPEN_POS: Tuple[int, int] = (438, 1500)
_DEFAULT_ALLIANCE_OPEN_POS: Tuple[int, int] = (428, 1215)

_DEFAULT_SWIPE_BOTTOM_TIMES: int = 3
_DEFAULT_SWIPE_BOTTOM_X: int = 400
_DEFAULT_SWIPE_BOTTOM_Y1: int = 1500
_DEFAULT_SWIPE_BOTTOM_Y2: int = 300
_DEFAULT_SWIPE_BOTTOM_DURATION_MS: int = 150


class ChatMode:
    WORLD = "world"
    ALLIANCE = "alliance"


class State(Enum):
    CHECK_CHAT = auto()       # 检测是否在目标聊天界面
    ENTER_CHAT = auto()       # 点击 Tab 入口进入聊天（普通重试）
    ENTER_HOME_CHAT = auto()  # 从首页固定点 [263, 1747] 进入聊天（首次/重置）
    FIND_RED_PACKET = auto()  # 按间隔扫描红包
    CLICK_OPEN = auto()       # 延迟后点击开启固定点
    VERIFY_CHAT = auto()      # 验证仍在聊天
    NAVIGATE_HOME = auto()    # 回首页


class RedPacketTask(BaseTask):
    priority = 10
    name = "抢红包"

    def __init__(
        self,
        config: dict | None = None,
        notify_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__()
        self._config_ref: dict = config or {}
        cfg = merge_task_settings(self._config_ref, "red_packet_task")

        _raw_tick_ms = cfg.get("tick_ms")
        if _raw_tick_ms is not None:
            self.tick_ms = int(_raw_tick_ms)

        self._chat_mode: str = cfg.get("chat_mode", ChatMode.ALLIANCE)
        self._notify_callback = notify_callback

        self._scan_interval_secs: float = float(
            cfg.get("scan_interval_secs", _DEFAULT_SCAN_INTERVAL_SECS)
        )
        self._click_delay_secs: float = float(
            cfg.get("click_delay_secs", _DEFAULT_CLICK_DELAY_SECS)
        )
        self._open_after_click_delay_secs: float = float(
            cfg.get("open_after_click_delay_secs", _DEFAULT_OPEN_AFTER_CLICK_DELAY_SECS)
        )
        self._verify_wait_secs: float = float(
            cfg.get("verify_wait_secs", _DEFAULT_VERIFY_WAIT_SECS)
        )
        self._enter_chat_wait_secs: float = float(
            cfg.get("enter_chat_wait_secs", _DEFAULT_ENTER_CHAT_WAIT_SECS)
        )
        self._enter_home_chat_wait_secs: float = float(
            cfg.get("enter_home_chat_wait_secs", _DEFAULT_ENTER_HOME_CHAT_WAIT_SECS)
        )
        self._max_check_chat_retries: int = int(
            cfg.get("max_check_chat_retries", _DEFAULT_MAX_CHECK_CHAT_RETRIES)
        )
        self._red_packet_threshold: float = float(
            cfg.get("red_packet_threshold", _DEFAULT_RED_PACKET_THRESHOLD)
        )
        self._chat_check_interval_secs: float = float(
            cfg.get("chat_check_interval_secs", _DEFAULT_CHAT_CHECK_INTERVAL_SECS)
        )

        # 固定坐标（优先从配置读取）
        self._world_chat_entry_pos: Tuple[int, int] = tuple(
            cfg.get("world_chat_entry_pos", list(_DEFAULT_WORLD_CHAT_ENTRY_POS))
        )
        self._alliance_chat_entry_pos: Tuple[int, int] = tuple(
            cfg.get("alliance_chat_entry_pos", list(_DEFAULT_ALLIANCE_CHAT_ENTRY_POS))
        )
        self._home_to_chat_pos: Tuple[int, int] = tuple(
            cfg.get("home_to_chat_pos", list(_DEFAULT_HOME_TO_CHAT_POS))
        )
        self._world_open_pos: Tuple[int, int] = tuple(
            cfg.get("world_open_pos", list(_DEFAULT_WORLD_OPEN_POS))
        )
        self._alliance_open_pos: Tuple[int, int] = tuple(
            cfg.get("alliance_open_pos", list(_DEFAULT_ALLIANCE_OPEN_POS))
        )

        # 滑动到底部参数
        self._swipe_bottom_times: int = int(
            cfg.get("swipe_bottom_times", _DEFAULT_SWIPE_BOTTOM_TIMES)
        )
        self._swipe_bottom_x: int = int(cfg.get("swipe_bottom_x", _DEFAULT_SWIPE_BOTTOM_X))
        self._swipe_bottom_y1: int = int(cfg.get("swipe_bottom_y1", _DEFAULT_SWIPE_BOTTOM_Y1))
        self._swipe_bottom_y2: int = int(cfg.get("swipe_bottom_y2", _DEFAULT_SWIPE_BOTTOM_Y2))
        self._swipe_bottom_duration_ms: int = int(
            cfg.get("swipe_bottom_duration_ms", _DEFAULT_SWIPE_BOTTOM_DURATION_MS)
        )

        dismiss_blank_pos: Tuple[int, int] = tuple(
            cfg.get("dismiss_blank_pos", list(_DEFAULT_DISMISS_BLANK_POS))
        )
        self._home_nav = HomeNavigator(
            max_retries=cfg.get("max_home_retries", _DEFAULT_MAX_HOME_RETRIES),
            blank_pos=dismiss_blank_pos,
            tag="抢红包",
        )

        self.required_templates: List[str] = list(dict.fromkeys([
            "世界聊天",
            "联盟聊天",
            "世界红包",
            "联盟红包",
            "世界地图",
            "城镇大门",
        ]))

        # 运行时状态
        # 首次开启时走完整初始化流程：NAVIGATE_HOME → ENTER_HOME_CHAT → CHECK_CHAT
        self._is_from_init: bool = True
        self._state: State = State.NAVIGATE_HOME
        self._wait_until: float = 0.0
        self._last_action_time: float = 0.0
        self._verify_back_count: int = 0
        self._check_chat_fail_count: int = 0
        self._last_chat_check_time: float = 0.0

        self._dispatch = {
            State.CHECK_CHAT:      self._step_check_chat,
            State.ENTER_CHAT:      self._step_enter_chat,
            State.ENTER_HOME_CHAT: self._step_enter_home_chat,
            State.FIND_RED_PACKET: self._step_find_red_packet,
            State.CLICK_OPEN:      self._step_click_open,
            State.VERIFY_CHAT:     self._step_verify_chat,
            State.NAVIGATE_HOME:   self._step_navigate_home,
        }

    # ── 模式相关属性 ──────────────────────────────────────────────────────────

    @property
    def chat_template_name(self) -> str:
        return "世界聊天" if self._chat_mode == ChatMode.WORLD else "联盟聊天"

    @property
    def red_packet_template_name(self) -> str:
        return "世界红包" if self._chat_mode == ChatMode.WORLD else "联盟红包"

    @property
    def chat_entry_pos(self) -> Tuple[int, int]:
        return self._world_chat_entry_pos if self._chat_mode == ChatMode.WORLD else self._alliance_chat_entry_pos

    @property
    def open_pos(self) -> Tuple[int, int]:
        return self._world_open_pos if self._chat_mode == ChatMode.WORLD else self._alliance_open_pos

    @property
    def step_template_names(self) -> List[str]:
        if self._state in (State.CHECK_CHAT, State.VERIFY_CHAT):
            return [self.chat_template_name]
        if self._state == State.FIND_RED_PACKET:
            return [self.red_packet_template_name, self.chat_template_name]
        if self._state == State.NAVIGATE_HOME:
            return ["世界地图", "城镇大门"]
        return []

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._is_from_init = True
        self._state = State.NAVIGATE_HOME
        self._wait_until = 0.0
        self._last_action_time = 0.0
        self._verify_back_count = 0
        self._check_chat_fail_count = 0
        self._home_nav.reset()

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

        logger.debug("[抢红包] 状态: %s 模式: %s", self._state.name, self._chat_mode)
        self._dispatch[self._state](frame, device, vision, template_paths)

    # ── 状态处理 ──────────────────────────────────────────────────────────────

    def _step_check_chat(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        chat_path = template_paths.get(self.chat_template_name)
        if chat_path and vision.match(frame, chat_path).found:
            logger.info(
                "[抢红包] 检测到「%s」，滑动到底部 ×%d，开始扫描红包",
                self.chat_template_name, self._swipe_bottom_times,
            )
            self._check_chat_fail_count = 0
            self._last_chat_check_time = 0.0
            for _ in range(self._swipe_bottom_times):
                device.adb_swipe(
                    self._swipe_bottom_x, self._swipe_bottom_y1,
                    self._swipe_bottom_x, self._swipe_bottom_y2,
                    duration_ms=self._swipe_bottom_duration_ms,
                )
            self._touch()
            self._go_to(State.FIND_RED_PACKET)
            return

        self._check_chat_fail_count += 1
        logger.debug(
            "[抢红包] 未检测到「%s」（第 %d 次）",
            self.chat_template_name, self._check_chat_fail_count,
        )

        if self._check_chat_fail_count >= self._max_check_chat_retries:
            logger.info(
                "[抢红包] 连续 %d 次未检测到聊天界面，回首页",
                self._check_chat_fail_count,
            )
            self._check_chat_fail_count = 0
            self._home_nav.reset()
            self._go_to(State.NAVIGATE_HOME)
            return

        self._go_to(State.ENTER_CHAT)

    def _step_enter_chat(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        x, y = self.chat_entry_pos
        logger.info("[抢红包] 点击聊天 Tab 入口 (%d, %d)", x, y)
        device.click(x, y)
        self._touch()
        self._wait_until = time.monotonic() + self._enter_chat_wait_secs
        self._go_to(State.CHECK_CHAT)

    def _step_enter_home_chat(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        x, y = self._home_to_chat_pos
        logger.info("[抢红包] 从首页点击固定点 (%d, %d) 进入聊天", x, y)
        device.click(x, y)
        self._touch()
        self._wait_until = time.monotonic() + self._enter_home_chat_wait_secs
        self._go_to(State.CHECK_CHAT)

    def _step_find_red_packet(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        # 确认聊天界面还在（每隔 chat_check_interval_secs 检测一次）
        now = time.monotonic()
        if now - self._last_chat_check_time >= self._chat_check_interval_secs:
            chat_path = template_paths.get(self.chat_template_name)
            if chat_path and not vision.match(frame, chat_path).found:
                logger.info("[抢红包] 聊天界面「%s」不可见，重新检测", self.chat_template_name)
                self._check_chat_fail_count = 0
                self._go_to(State.CHECK_CHAT)
                return
            self._last_chat_check_time = now
            logger.info(
                "[抢红包] 定期滑动到底部 ×%d（每 %.0f 秒）",
                self._swipe_bottom_times, self._chat_check_interval_secs,
            )
            for _ in range(self._swipe_bottom_times):
                device.adb_swipe(
                    self._swipe_bottom_x, self._swipe_bottom_y1,
                    self._swipe_bottom_x, self._swipe_bottom_y2,
                    duration_ms=self._swipe_bottom_duration_ms,
                )
            self._touch()

        # 扫描红包（固定阈值 0.9）
        red_path = template_paths.get(self.red_packet_template_name)
        if not red_path:
            self._wait_until = time.monotonic() + self._scan_interval_secs
            return

        result = vision.match(frame, red_path, threshold=self._red_packet_threshold)
        if result.found:
            logger.info(
                "[抢红包] 发现「%s」(%d, %d) conf=%.3f，点击",
                self.red_packet_template_name, result.x, result.y, result.confidence,
            )
            device.click(result.x, result.y)
            self._touch()
            self._verify_back_count = 0
            self._wait_until = time.monotonic() + self._click_delay_secs
            self._go_to(State.CLICK_OPEN)
        else:
            self._wait_until = time.monotonic() + self._scan_interval_secs

    def _step_click_open(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        x, y = self.open_pos
        logger.info("[抢红包] 点击开启固定点 (%d, %d)，等待 %.1fs", x, y, self._open_after_click_delay_secs)
        device.click(x, y)
        self._touch()
        self._wait_until = time.monotonic() + self._open_after_click_delay_secs
        self._go_to(State.VERIFY_CHAT)

    def _step_verify_chat(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        chat_path = template_paths.get(self.chat_template_name)
        if chat_path and vision.match(frame, chat_path).found:
            logger.info(
                "[抢红包] 验证聊天界面：检测到「%s」，滑动到底部 ×%d",
                self.chat_template_name, self._swipe_bottom_times,
            )
            self._verify_back_count = 0
            for _ in range(self._swipe_bottom_times):
                device.adb_swipe(
                    self._swipe_bottom_x, self._swipe_bottom_y1,
                    self._swipe_bottom_x, self._swipe_bottom_y2,
                    duration_ms=self._swipe_bottom_duration_ms,
                )
            self._touch()
            # 发送飞书通知
            mode_label = "世界" if self._chat_mode == ChatMode.WORLD else "联盟"
            self._send_notify(f"[抢红包] 已抢到一个{mode_label}红包！")
            self._check_chat_fail_count = 0
            self._go_to(State.CHECK_CHAT)
            return

        if self._verify_back_count < 2:
            logger.debug(
                "[抢红包] 验证聊天界面：第 %d 次未找到，执行返回",
                self._verify_back_count + 1,
            )
            try_go_back(
                frame=frame, device=device, vision=vision,
                template_paths=template_paths, tag="抢红包",
            )
            self._verify_back_count += 1
            self._touch()
            self._wait_until = time.monotonic() + self._verify_wait_secs
            return

        # 2 次返回后仍未检测到 → 回首页
        logger.info("[抢红包] 验证聊天界面：2 次返回后仍未找到，回首页")
        self._verify_back_count = 0
        self._check_chat_fail_count = 0
        self._home_nav.reset()
        self._go_to(State.NAVIGATE_HOME)

    def _step_navigate_home(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        status = self._home_nav.step(frame, device, vision, template_paths)
        if status == HomeNavigator.STATUS_HOME:
            self._is_from_init = False
            logger.info("[抢红包] 已回首页，从首页固定点进入聊天")
            self._touch()
            self._go_to(State.ENTER_HOME_CHAT)
        elif status == HomeNavigator.STATUS_FAILED:
            logger.error("[抢红包] 无法回到首页，任务退出")
            self.is_finished = True
        else:
            self._touch()

    # ── 运行时参数调整接口（供 UI 调用）──────────────────────────────────────

    def set_chat_mode(self, mode: str) -> None:
        """切换聊天模式（world / alliance），立即生效并重置状态。"""
        if mode not in (ChatMode.WORLD, ChatMode.ALLIANCE):
            logger.warning("[抢红包] 未知聊天模式：%s", mode)
            return
        if mode != self._chat_mode:
            logger.info("[抢红包] 切换聊天模式：%s → %s", self._chat_mode, mode)
            self._chat_mode = mode
            self._check_chat_fail_count = 0
            self._verify_back_count = 0
            self._wait_until = 0.0
            self._go_to(State.CHECK_CHAT)

    def set_scan_interval(self, interval_secs: float) -> None:
        """设置扫描红包的时间间隔（秒），立即生效。"""
        self._scan_interval_secs = max(0.1, float(interval_secs))
        logger.info("[抢红包] 扫描间隔更新为 %.2f s", self._scan_interval_secs)

    def set_click_delay(self, delay_secs: float) -> None:
        """设置点击红包到点击开启按钮的延迟（秒），立即生效。"""
        self._click_delay_secs = max(0.0, float(delay_secs))
        logger.info("[抢红包] 点击延迟更新为 %.2f s", self._click_delay_secs)

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _go_to(self, state: State) -> None:
        logger.debug("[抢红包] %s → %s", self._state.name, state.name)
        self._state = state

    def _touch(self) -> None:
        self._last_action_time = time.monotonic()

    def _send_notify(self, message: str) -> None:
        if self._notify_callback:
            try:
                self._notify_callback(message)
            except Exception:
                logger.exception("[抢红包] 飞书通知发送异常")

    @property
    def state_label(self) -> str:
        labels = {
            State.CHECK_CHAT:      "检测聊天界面",
            State.ENTER_CHAT:      "进入聊天",
            State.ENTER_HOME_CHAT: "从首页进入聊天",
            State.FIND_RED_PACKET: "扫描红包",
            State.CLICK_OPEN:      "点击开启",
            State.VERIFY_CHAT:     "验证聊天界面",
            State.NAVIGATE_HOME:   "返回首页",
        }
        mode_label = "世界" if self._chat_mode == ChatMode.WORLD else "联盟"
        return f"[{mode_label}] {labels.get(self._state, self._state.name)}"
