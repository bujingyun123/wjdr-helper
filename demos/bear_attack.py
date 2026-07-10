# -*- coding: utf-8 -*-
"""
BearAttackTask — 自动打熊（联盟战争抢队伍）任务

FSM 状态流：

    CHECK_ALLIANCE_WAR  检测是否处于联盟战争
      ├─ 是联盟战争        → FIND_TEAM
      └─ 否               → tick 重试（如出征后界面未就绪）→ 仍否 → NAVIGATE_HOME

    NAVIGATE_HOME        回到首页（借助 HomeNavigator）
      ├─ 成功             → FIND_ALLIANCE_TAB
      └─ 失败             → 任务退出

    FIND_ALLIANCE_TAB    查找联盟 tab 并点击
      ├─ 找到             → FIND_SWORDS
      └─ 未找到           → CHECK_ALLIANCE_WAR

    FIND_SWORDS          查找「两把剑」（联盟战争入口）并点击
      ├─ 找到             → CHECK_ALLIANCE_WAR
      └─ 未找到           → FIND_ALLIANCE_TAB

    FIND_TEAM            按队伍顺序扫描所有模板
      ├─ 找到可加入队伍    → 点击「加入队伍绿色」→ 记录队伍编号 → FIND_GROUP
      ├─ 找到不可加入      → 标记该模板冷却，继续扫描
      └─ 全部未找到/冷却中 → 等待下次 tick

    FIND_GROUP           在编队页查找「编队N」（N = 当前队伍编号）
      ├─ 找到对应编队      → 点击 → FIND_EXPEDITION
      └─ 未找到编队        → 未进入编队页 → CHECK_ALLIANCE_WAR

    FIND_EXPEDITION      查找并点击「出征」
      ├─ 找到             → 点击 → CHECK_ALLIANCE_WAR
      └─ 重试 tick 耗尽未找到 → CHECK_ALLIANCE_WAR
"""
from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.device import Device
from core.vision import Vision
from engine.task_base import BaseTask
from logic.common_steps import HomeNavigator, StateTickRetry, find_first
from logic.task_config import merge_task_settings

logger = logging.getLogger(__name__)


# ── 默认配置常量 ────────────────────────────────────────────────────────────────

_DEFAULT_TEAM_CROP_OFFSET: Tuple[int, int] = (380, 300)
_DEFAULT_TEAM_SKIP_SECS: float = 30.0
_DEFAULT_MAX_HOME_RETRIES: int = 5
_DEFAULT_STATE_RETRY_TICKS: Dict[str, int] = {
    "CHECK_ALLIANCE_WAR": 5,
    "FIND_ALLIANCE_TAB": 3,
    "FIND_SWORDS": 3,
    "FIND_GROUP": 5,
    "FIND_EXPEDITION": 5,
}

# 默认队伍配置：队伍编号 → 模板名称列表
_DEFAULT_TEAMS: Dict[int, List[str]] = {
    1: ["GEN队伍", "小礼物队伍"],
    2: ["幺五七队伍"],
    3: [],
}


class State(Enum):
    CHECK_ALLIANCE_WAR = auto()  # 检测是否处于联盟战争
    NAVIGATE_HOME      = auto()  # 回到首页
    FIND_ALLIANCE_TAB  = auto()  # 查找联盟 tab
    FIND_SWORDS        = auto()  # 查找两把剑
    FIND_TEAM          = auto()  # 扫描队伍模板
    FIND_GROUP         = auto()  # 选择编队
    FIND_EXPEDITION    = auto()  # 出征


class BearAttackTask(BaseTask):
    priority = 20
    name = "打熊"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = merge_task_settings(config, "bear_attack_task")

        # ── 队伍配置 {队伍编号(int): [模板名称, ...]}
        raw_teams = cfg.get("teams", _DEFAULT_TEAMS)
        self._teams: Dict[int, List[str]] = {
            int(k): list(v) for k, v in raw_teams.items()
        }

        # ── 裁剪偏移：从队伍模板坐标向右下裁剪 (dx, dy)
        self._crop_offset: Tuple[int, int] = tuple(
            cfg.get("team_crop_offset", list(_DEFAULT_TEAM_CROP_OFFSET))
        )

        # ── 不可加入时的冷却时间（秒）
        self._team_skip_secs: float = cfg.get("team_skip_secs", _DEFAULT_TEAM_SKIP_SECS)

        # ── 通用配置（从 defaults 或任务级配置继承）
        dismiss_blank_pos: Tuple[int, int] = tuple(cfg.get("dismiss_blank_pos", [400, 1700]))

        self._home_nav = HomeNavigator(
            max_retries=cfg.get("max_home_retries", _DEFAULT_MAX_HOME_RETRIES),
            blank_pos=dismiss_blank_pos,
            tag="打熊",
        )

        # ── 所有队伍模板名（去重后保留顺序）
        _all_team_tmpls: List[str] = list(dict.fromkeys(
            t for tlist in self._teams.values() for t in tlist
        ))

        # ── 所有编队模板名（每个编队支持主模板 + 备用模板）
        self._group_templates: Dict[int, List[str]] = {
            i: [f"编队{i}", f"编队{i}备用"] for i in sorted(self._teams.keys())
        }
        _group_names: List[str] = list(dict.fromkeys(
            name for names in self._group_templates.values() for name in names
        ))

        # ── 任务所需全量模板（调度器 / Tracker 注册用）
        self.required_templates: List[str] = list(dict.fromkeys([
            "联盟战争",
            "联盟tab",
            "两把剑",
            *_all_team_tmpls,
            "加入队伍灰色",
            "加入队伍绿色",
            *_group_names,
            "出征按钮",
            "世界地图",
            "城镇大门",
        ]))

        # ── 按状态声明的最小模板集，减少无效匹配
        self._state_templates: Dict[State, List[str]] = {
            State.CHECK_ALLIANCE_WAR: ["联盟战争"],
            State.NAVIGATE_HOME:      ["世界地图", "城镇大门"],
            State.FIND_ALLIANCE_TAB:  ["联盟tab"],
            State.FIND_SWORDS:        ["两把剑"],
            State.FIND_TEAM:          [*_all_team_tmpls, "加入队伍灰色", "加入队伍绿色"],
            State.FIND_GROUP:         _group_names,
            State.FIND_EXPEDITION:    ["出征按钮"],
        }

        # ── 运行时状态
        self._state: State = State.CHECK_ALLIANCE_WAR
        self._current_team_id: int = 1        # 当前正在尝试加入的队伍编号
        self._skip_until: Dict[str, float] = {}  # 模板名 → 冷却截止时间戳
        self._last_action_time: float = 0.0
        self._action_cooldown: float = 0.6

        # ── 按状态配置 tick 重试次数（状态名 -> 允许重试 tick 数）
        # 例如 CHECK_ALLIANCE_WAR：未识别到「联盟战争」时先重试若干 tick 再回首页
        raw_state_retry_ticks = cfg.get("state_retry_ticks", _DEFAULT_STATE_RETRY_TICKS)
        if not isinstance(raw_state_retry_ticks, dict):
            raw_state_retry_ticks = _DEFAULT_STATE_RETRY_TICKS
        raw_skip_first_entry_states = cfg.get(
            "state_retry_skip_first_entry_states",
            ["CHECK_ALLIANCE_WAR"],
        )
        if not isinstance(raw_skip_first_entry_states, list):
            raw_skip_first_entry_states = ["CHECK_ALLIANCE_WAR"]
        self._state_tick_retry = StateTickRetry(
            raw_state_retry_ticks,
            skip_retry_on_first_entry_states=raw_skip_first_entry_states,
        )

    # ── 接口实现 ───────────────────────────────────────────────────────────────

    @property
    def step_template_names(self) -> List[str]:
        return self._state_templates.get(self._state, self.required_templates)

    def reset(self) -> None:
        super().reset()
        self._state = State.CHECK_ALLIANCE_WAR
        self._current_team_id = 1
        self._skip_until.clear()
        self._last_action_time = 0.0
        self._state_tick_retry.reset()
        self._state_tick_retry.on_state_change(changed=True, state_name=self._state.name)
        self._home_nav.reset()

    def run_step(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: Dict[str, str],
    ) -> None:
        if time.monotonic() - self._last_action_time < self._action_cooldown:
            return

        logger.debug("[打熊] 状态: %s", self._state.name)

        _dispatch = {
            State.CHECK_ALLIANCE_WAR: self._step_check_alliance_war,
            State.NAVIGATE_HOME:      self._step_navigate_home,
            State.FIND_ALLIANCE_TAB:  self._step_find_alliance_tab,
            State.FIND_SWORDS:        self._step_find_swords,
            State.FIND_TEAM:          self._step_find_team,
            State.FIND_GROUP:         self._step_find_group,
            State.FIND_EXPEDITION:    self._step_find_expedition,
        }
        _dispatch[self._state](frame, device, vision, template_paths)

    # ── 各状态处理 ────────────────────────────────────────────────────────────

    def _step_check_alliance_war(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """检测当前是否处于联盟战争界面。"""
        path = template_paths.get("联盟战争")
        if path and vision.match(frame, path).found:
            logger.info("[打熊] 已处于联盟战争，开始查找队伍")
            self._go_to(State.FIND_TEAM)
            return

        if self._retry_current_state_tick(scene_desc="检测联盟战争界面"):
            return
        logger.info("[打熊] 未处于联盟战争，准备回首页导航")
        self._home_nav.reset()
        self._go_to(State.NAVIGATE_HOME)

    def _step_navigate_home(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """借助 HomeNavigator 回到首页，成功后前往查找联盟 tab。"""
        status = self._home_nav.step(frame, device, vision, template_paths)
        if status == HomeNavigator.STATUS_HOME:
            self._touch()
            self._go_to(State.FIND_ALLIANCE_TAB)
        elif status == HomeNavigator.STATUS_FAILED:
            logger.error("[打熊] 无法回到首页，任务退出")
            self.is_finished = True
        else:
            self._touch()

    def _step_find_alliance_tab(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """查找并点击联盟 tab。"""
        path = template_paths.get("联盟tab")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[打熊] 点击联盟tab (%d, %d)", result.x, result.y)
                device.click(result.x, result.y)
                self._touch()
                self._go_to(State.FIND_SWORDS)
                return
        if self._retry_current_state_tick(scene_desc="查找联盟tab"):
            return
        logger.debug("[打熊] 联盟tab 重试结束，重新检测联盟战争")
        self._go_to(State.CHECK_ALLIANCE_WAR)

    def _step_find_swords(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """查找并点击「两把剑」进入联盟战争。"""
        path = template_paths.get("两把剑")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[打熊] 点击两把剑 (%d, %d)", result.x, result.y)
                device.click(result.x, result.y)
                self._touch()
                self._go_to(State.CHECK_ALLIANCE_WAR)
                return
        if self._retry_current_state_tick(scene_desc="查找两把剑"):
            return
        logger.debug("[打熊] 两把剑重试结束，返回查找联盟tab")
        self._go_to(State.FIND_ALLIANCE_TAB)

    def _step_find_team(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """
        按队伍编号从小到大扫描所有模板。
        - 找到可加入（绿色）→ 点击，转 FIND_GROUP
        - 找到不可加入（灰色）→ 标记该模板冷却，继续扫描
        - 本轮全部跳过 → 等待下次 tick
        """
        now = time.monotonic()
        crop_dx, crop_dy = self._crop_offset
        frame_h, frame_w = frame.shape[:2]

        for team_id in sorted(self._teams.keys()):
            for tmpl_name in self._teams[team_id]:
                # 冷却中跳过
                if self._skip_until.get(tmpl_name, 0.0) > now:
                    remaining = self._skip_until[tmpl_name] - now
                    logger.debug(
                        "[打熊] 「%s」冷却中，剩余 %.0fs", tmpl_name, remaining
                    )
                    continue

                team_path = template_paths.get(tmpl_name)
                if not team_path:
                    continue

                team_result = vision.match(frame, team_path)
                if not team_result.found:
                    continue

                tx, ty = team_result.x, team_result.y
                logger.debug(
                    "[打熊] 找到「%s」at (%d, %d)，裁剪区域检查加入状态",
                    tmpl_name, tx, ty,
                )

                # 以队伍模板坐标为起点向右下裁剪
                x1 = tx
                y1 = ty
                x2 = min(tx + crop_dx, frame_w)
                y2 = min(ty + crop_dy, frame_h)
                sub_frame = frame[y1:y2, x1:x2]

                # 优先检测「可加入（绿色）」
                green_path = template_paths.get("加入队伍绿色")
                if green_path:
                    gr = vision.match(sub_frame, green_path)
                    if gr.found:
                        abs_x = x1 + gr.x
                        abs_y = y1 + gr.y
                        logger.info(
                            "[打熊] 「%s」可加入，点击 (%d, %d)，队伍编号=%d",
                            tmpl_name, abs_x, abs_y, team_id,
                        )
                        device.click(abs_x, abs_y)
                        self._current_team_id = team_id
                        self._touch()
                        self._go_to(State.FIND_GROUP)
                        return

                # 检测「不可加入（灰色）」
                gray_path = template_paths.get("加入队伍灰色")
                if gray_path:
                    gr = vision.match(sub_frame, gray_path)
                    if gr.found:
                        logger.info(
                            "[打熊] 「%s」不可加入，冷却 %.0fs",
                            tmpl_name, self._team_skip_secs,
                        )
                        self._skip_until[tmpl_name] = now + self._team_skip_secs
                        continue

        logger.debug("[打熊] 本轮未找到可加入队伍，等待下次 tick")

    def _step_find_group(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """
        在编队页查找对应编队（编队N，N = 当前队伍编号）并点击。
        未找到则说明未成功进入编队页，回到联盟战争检测。
        """
        group_names = self._group_templates.get(
            self._current_team_id,
            [f"编队{self._current_team_id}", f"编队{self._current_team_id}备用"],
        )

        for group_name in group_names:
            path = template_paths.get(group_name)
            if not path:
                continue
            result = vision.match(frame, path)
            if result.found:
                logger.info(
                    "[打熊] 点击「%s」(%d, %d)", group_name, result.x, result.y
                )
                device.click(result.x, result.y)
                self._touch()
                self._go_to(State.FIND_EXPEDITION)
                return

        logger.info(
            "[打熊] 未找到编队模板（%s），未进入编队页，重新检测联盟战争",
            " / ".join(group_names),
        )
        if self._retry_current_state_tick(scene_desc=f"查找编队「{' / '.join(group_names)}」"):
            return
        logger.info(
            "[打熊] 编队模板（%s）重试结束，重新检测联盟战争",
            " / ".join(group_names),
        )
        self._go_to(State.CHECK_ALLIANCE_WAR)

    def _step_find_expedition(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        """查找并点击「出征」按钮；按 state_retry_ticks 重试后仍无则返回检测联盟战争。"""
        path = template_paths.get("出征按钮")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[打熊] 点击出征 (%d, %d)", result.x, result.y)
                device.click(result.x, result.y)
                self._touch()
                self._go_to(State.CHECK_ALLIANCE_WAR)
                return

        if self._retry_current_state_tick(scene_desc="查找出征按钮"):
            return
        logger.info("[打熊] 出征按钮 tick 重试结束，重新检测联盟战争")
        self._go_to(State.CHECK_ALLIANCE_WAR)

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _go_to(self, state: State) -> None:
        self._state_tick_retry.on_state_change(
            changed=(state != self._state),
            state_name=state.name,
        )
        logger.debug("[打熊] %s → %s", self._state.name, state.name)
        self._state = state

    def _touch(self) -> None:
        self._last_action_time = time.monotonic()

    def _retry_current_state_tick(self, scene_desc: str = "") -> bool:
        """
        当前状态模板未命中时，按配置重试若干个 tick。

        :return: True 表示继续留在当前状态等待下一 tick；False 表示重试已耗尽
        """
        return self._state_tick_retry.try_retry(
            state_name=self._state.name,
            log_prefix="打熊",
            scene_desc=scene_desc,
        )

    @property
    def state_label(self) -> str:
        labels = {
            State.CHECK_ALLIANCE_WAR: "检测联盟战争",
            State.NAVIGATE_HOME:      "返回首页",
            State.FIND_ALLIANCE_TAB:  "查找联盟tab",
            State.FIND_SWORDS:        "查找两把剑",
            State.FIND_TEAM:          "查找队伍",
            State.FIND_GROUP:         "选择编队",
            State.FIND_EXPEDITION:    "出征按钮",
        }
        return labels.get(self._state, self._state.name)
