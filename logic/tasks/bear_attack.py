# -*- coding: utf-8 -*-
"""
BearAttackTask — 自动打熊（固定点快速出征）。
"""
from __future__ import annotations

import logging
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.device import Device
from core.vision import Vision
from engine.task_base import BaseTask
from logic.common_steps import HomeNavigator, StateTickRetry, try_go_back
from logic.task_config import merge_task_settings

logger = logging.getLogger(__name__)


_DEFAULT_MAX_HOME_RETRIES: int = 5
_DEFAULT_TEAM_COOLDOWN_SECS: float = 60.0
_DEFAULT_DELAY_BEFORE_CLICK_GROUP_SECS: float = 0.1
_DEFAULT_DELAY_BEFORE_CLICK_EXPEDITION_SECS: float = 0.2
_DEFAULT_DELAY_AFTER_CLICK_EXPEDITION_CHECK_SECS: float = 2.0
_DEFAULT_DELAY_BEFORE_FIND_SWORDS_SECS: float = 1.5
_DEFAULT_DELAY_BEFORE_CLICK_RALLY_TAB_AFTER_SWORDS_SECS: float = 0.5
_DEFAULT_DELAY_BEFORE_CHECK_RALLY_PAGE_AFTER_TAB_CLICK_SECS: float = 0.5
_DEFAULT_RALLY_SCROLL_INTERVAL_SECS: float = 60.0
_DEFAULT_RALLY_SCROLL_TIMES: int = 1
_DEFAULT_RALLY_SCROLL_START_POINT: Tuple[int, int] = (960, 1500)
_DEFAULT_RALLY_SCROLL_END_POINT: Tuple[int, int] = (960, 500)
_DEFAULT_RALLY_SCROLL_DURATION_SECS: float = 0.35
_DEFAULT_RALLY_PAGE_ENTER_SCROLL_TIMES: int = 2
_DEFAULT_RALLY_PAGE_ENTER_SCROLL_START_POINT: Tuple[int, int] = (419, 1504)
_DEFAULT_RALLY_PAGE_ENTER_SCROLL_END_POINT: Tuple[int, int] = (415, 776)
_DEFAULT_RALLY_PAGE_ENTER_SCROLL_DURATION_SECS: float = 0.1
_DEFAULT_DELAY_AFTER_VERIFY_MISS_SECS: float = 0.5
_DEFAULT_DELAY_AFTER_VERIFY_BACK_SECS: float = 0.2
_DEFAULT_RALLY_TAB_POINT: Tuple[int, int] = (163, 154)
_DEFAULT_ENTER_GROUP_BUTTON_Y_VALID_RANGES: List[Tuple[int, int]] = [
    (1448, 1496),
    (997, 1007),
    (508, 518),
]
_DEFAULT_FIRST_TWO_TEAM_Y_EXTRA_VALID_RANGES: List[Tuple[int, int]] = [
    (462, 475),
    (951, 965),
    (1439, 1452),
]
_DEFAULT_STATE_RETRY_TICKS: Dict[str, int] = {
    "CHECK_RALLY_PAGE": 4,
    "FIND_ALLIANCE_TAB": 4,
    "FIND_SWORDS": 4,
}
_DEFAULT_TEAMS: Dict[int, List[str]] = {
    1: ["GEN队伍", "小礼物队伍"],
    2: ["幺五七队伍"],
    3: [],
}


class State(Enum):
    CHECK_RALLY_PAGE = auto()
    NAVIGATE_HOME = auto()
    FIND_ALLIANCE_TAB = auto()
    FIND_SWORDS = auto()
    FIND_TEAM = auto()
    WAIT_CLICK_GROUP = auto()
    CLICK_EXPEDITION = auto()
    VERIFY_RALLY_PAGE = auto()


class BearAttackTask(BaseTask):
    priority = 20
    name = "打熊"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self._config_ref: dict = config or {}
        cfg = merge_task_settings(self._config_ref, "bear_attack_task")

        _raw_tick_ms = cfg.get("tick_ms")
        if _raw_tick_ms is not None:
            self.tick_ms = int(_raw_tick_ms)

        self._teams: Dict[int, List[str]] = self._normalize_teams_cfg(cfg.get("teams", _DEFAULT_TEAMS))

        self._rally_template: str = cfg.get("rally_template", "熊集结页")
        self._team_cooldown_secs: float = float(
            cfg.get("team_cooldown_secs", _DEFAULT_TEAM_COOLDOWN_SECS)
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
        self._delay_after_verify_miss_secs: float = float(
            cfg.get("delay_after_verify_miss_secs", _DEFAULT_DELAY_AFTER_VERIFY_MISS_SECS)
        )
        self._delay_after_verify_back_secs: float = float(
            cfg.get("delay_after_verify_back_secs", _DEFAULT_DELAY_AFTER_VERIFY_BACK_SECS)
        )
        self._delay_before_find_swords_secs: float = float(
            cfg.get("delay_before_find_swords_secs", _DEFAULT_DELAY_BEFORE_FIND_SWORDS_SECS)
        )
        self._delay_before_click_rally_tab_after_swords_secs: float = float(
            cfg.get(
                "delay_before_click_rally_tab_after_swords_secs",
                _DEFAULT_DELAY_BEFORE_CLICK_RALLY_TAB_AFTER_SWORDS_SECS,
            )
        )
        self._delay_before_check_rally_page_after_tab_click_secs: float = float(
            cfg.get(
                "delay_before_check_rally_page_after_tab_click_secs",
                _DEFAULT_DELAY_BEFORE_CHECK_RALLY_PAGE_AFTER_TAB_CLICK_SECS,
            )
        )
        self._rally_scroll_interval_secs: float = float(
            cfg.get("rally_scroll_interval_secs", _DEFAULT_RALLY_SCROLL_INTERVAL_SECS)
        )
        self._rally_scroll_times: int = max(
            0, int(cfg.get("rally_scroll_times", _DEFAULT_RALLY_SCROLL_TIMES))
        )
        self._rally_scroll_start_point: Tuple[int, int] = tuple(
            cfg.get("rally_scroll_start_point", list(_DEFAULT_RALLY_SCROLL_START_POINT))
        )
        self._rally_scroll_end_point: Tuple[int, int] = tuple(
            cfg.get("rally_scroll_end_point", list(_DEFAULT_RALLY_SCROLL_END_POINT))
        )
        self._rally_scroll_duration_secs: float = float(
            cfg.get("rally_scroll_duration_secs", _DEFAULT_RALLY_SCROLL_DURATION_SECS)
        )
        self._rally_page_enter_scroll_times: int = max(
            0, int(cfg.get("rally_page_enter_scroll_times", _DEFAULT_RALLY_PAGE_ENTER_SCROLL_TIMES))
        )
        self._rally_page_enter_scroll_start_point: Tuple[int, int] = tuple(
            cfg.get(
                "rally_page_enter_scroll_start_point",
                list(_DEFAULT_RALLY_PAGE_ENTER_SCROLL_START_POINT),
            )
        )
        self._rally_page_enter_scroll_end_point: Tuple[int, int] = tuple(
            cfg.get(
                "rally_page_enter_scroll_end_point",
                list(_DEFAULT_RALLY_PAGE_ENTER_SCROLL_END_POINT),
            )
        )
        self._rally_page_enter_scroll_duration_secs: float = float(
            cfg.get(
                "rally_page_enter_scroll_duration_secs",
                _DEFAULT_RALLY_PAGE_ENTER_SCROLL_DURATION_SECS,
            )
        )
        self._enter_group_button_y_valid_ranges: List[Tuple[int, int]] = self._parse_y_ranges(
            cfg.get(
                "enter_group_button_y_valid_ranges",
                _DEFAULT_ENTER_GROUP_BUTTON_Y_VALID_RANGES,
            ),
            _DEFAULT_ENTER_GROUP_BUTTON_Y_VALID_RANGES,
        )
        self._first_two_found_enter_y_extra_valid_ranges: List[Tuple[int, int]] = self._parse_y_ranges(
            cfg.get(
                "first_two_team_y_extra_valid_ranges",
                _DEFAULT_FIRST_TWO_TEAM_Y_EXTRA_VALID_RANGES,
            ),
            _DEFAULT_FIRST_TWO_TEAM_Y_EXTRA_VALID_RANGES,
        )

        self._enter_group_button_x: int = int(cfg.get("enter_group_button_x", 770))
        self._enter_group_button_y_offset: int = int(cfg.get("enter_group_button_y_offset", 134))
        self._expedition_button_point: Tuple[int, int] = tuple(
            cfg.get("expedition_button_point", [1665, 995])
        )
        self._rally_tab_point_after_swords: Tuple[int, int] = tuple(
            cfg.get("rally_tab_point_after_swords", list(_DEFAULT_RALLY_TAB_POINT))
        )
        raw_group_points = cfg.get("group_points", {})
        self._group_points: Dict[int, Tuple[int, int]] = {int(k): tuple(v) for k, v in raw_group_points.items()}

        dismiss_blank_pos: Tuple[int, int] = tuple(cfg.get("dismiss_blank_pos", [400, 1700]))

        self._home_nav = HomeNavigator(
            max_retries=cfg.get("max_home_retries", _DEFAULT_MAX_HOME_RETRIES),
            blank_pos=dismiss_blank_pos,
            tag="打熊",
        )

        raw_retry_ticks = cfg.get("state_retry_ticks", _DEFAULT_STATE_RETRY_TICKS)
        if not isinstance(raw_retry_ticks, dict):
            raw_retry_ticks = _DEFAULT_STATE_RETRY_TICKS
        self._state_tick_retry = StateTickRetry(
            raw_retry_ticks,
            skip_retry_on_first_entry_states=["CHECK_RALLY_PAGE"],
        )

        self._all_team_templates: List[str] = list(dict.fromkeys(t for tlist in self._teams.values() for t in tlist))
        self._non_team_required_templates: List[str] = [
            self._rally_template,
            "联盟tab",
            "两把剑",
            "世界地图",
            "城镇大门",
        ]
        self.required_templates: List[str] = list(
            dict.fromkeys(
                [
                    *self._non_team_required_templates,
                    *self._all_team_templates,
                ]
            )
        )
        self._state_templates: Dict[State, List[str]] = {
            State.CHECK_RALLY_PAGE: [self._rally_template],
            State.NAVIGATE_HOME: ["世界地图", "城镇大门"],
            State.FIND_ALLIANCE_TAB: ["联盟tab"],
            State.FIND_SWORDS: ["两把剑"],
            State.FIND_TEAM: self._all_team_templates,
            State.WAIT_CLICK_GROUP: [],
            State.CLICK_EXPEDITION: [],
            State.VERIFY_RALLY_PAGE: [self._rally_template],
        }

        self._state: State = State.CHECK_RALLY_PAGE
        self._current_team_id: int = 1
        self._team_skip_until: Dict[int, float] = {}
        self._last_action_time: float = 0.0
        self._wait_until: float = 0.0
        self._verify_back_count: int = 0
        self._last_rally_scroll_time: float = 0.0
        self._found_team_count: int = 0
        self._last_team_refresh_time: float = 0.0
        self._monster_mode: bool = False
        self._team_search_logger = self._build_team_search_logger()

        self._dispatch = {
            State.CHECK_RALLY_PAGE: self._step_check_rally_page,
            State.NAVIGATE_HOME: self._step_navigate_home,
            State.FIND_ALLIANCE_TAB: self._step_find_alliance_tab,
            State.FIND_SWORDS: self._step_find_swords,
            State.FIND_TEAM: self._step_find_team,
            State.WAIT_CLICK_GROUP: self._step_wait_click_group,
            State.CLICK_EXPEDITION: self._step_click_expedition,
            State.VERIFY_RALLY_PAGE: self._step_verify_rally_page,
        }

    _TEAM_REFRESH_INTERVAL_SECS: float = 5.0

    @property
    def step_template_names(self) -> List[str]:
        now = time.monotonic()
        if now - self._last_team_refresh_time >= self._TEAM_REFRESH_INTERVAL_SECS:
            self._refresh_runtime_teams()
            self._last_team_refresh_time = now
        if self._state == State.FIND_TEAM:
            return self._all_team_templates
        return self._state_templates.get(self._state, self.required_templates)

    def reset(self) -> None:
        super().reset()
        self._state = State.CHECK_RALLY_PAGE
        self._current_team_id = 1
        self._team_skip_until.clear()
        self._last_action_time = 0.0
        self._wait_until = 0.0
        self._verify_back_count = 0
        self._last_rally_scroll_time = 0.0
        self._found_team_count = 0
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
        now = time.monotonic()
        if now < self._wait_until:
            return

        logger.debug("[打熊] 状态: %s", self._state.name)
        self._dispatch[self._state](frame, device, vision, template_paths)

    def _step_check_rally_page(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        path = template_paths.get(self._rally_template)
        if path and vision.match(frame, path).found:
            logger.info("[打熊] 已在集结页，开始查找队伍")
            self._scroll_to_rally_page_bottom_on_enter(device, reason="首次进入集结页")
            self._go_to(State.FIND_TEAM)
            return
        if self._retry_current_state_tick("检测集结页面"):
            return
        logger.info("[打熊] 不在集结页，准备回首页重新进入")
        self._home_nav.reset()
        self._go_to(State.NAVIGATE_HOME)

    def _step_navigate_home(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        status = self._home_nav.step(frame, device, vision, template_paths)
        if status == HomeNavigator.STATUS_HOME:
            logger.info("[打熊] 已回首页，开始找联盟tab")
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
        path = template_paths.get("联盟tab")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[打熊] 点击联盟tab (%d, %d)", result.x, result.y)
                device.click(result.x, result.y)
                self._touch()
                self._wait_until = time.monotonic() + self._delay_before_find_swords_secs
                self._go_to(State.FIND_SWORDS)
                return
        if self._retry_current_state_tick("查找联盟tab"):
            return
        self._go_to(State.CHECK_RALLY_PAGE)

    def _step_find_swords(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        path = template_paths.get("两把剑")
        if path:
            result = vision.match(frame, path)
            if result.found:
                logger.info("[打熊] 点击两把剑 (%d, %d)", result.x, result.y)
                device.click(result.x, result.y)
                self._touch()
                if self._delay_before_click_rally_tab_after_swords_secs > 0:
                    time.sleep(self._delay_before_click_rally_tab_after_swords_secs)
                rally_tab_x, rally_tab_y = self._rally_tab_point_after_swords
                logger.info("[打熊] 两把剑后兼容点击集结tab固定点 (%d, %d)", rally_tab_x, rally_tab_y)
                device.click(int(rally_tab_x), int(rally_tab_y))
                self._touch()
                if self._delay_before_check_rally_page_after_tab_click_secs > 0:
                    time.sleep(self._delay_before_check_rally_page_after_tab_click_secs)
                self._go_to(State.CHECK_RALLY_PAGE)
                return
        if self._retry_current_state_tick("查找两把剑"):
            return
        self._go_to(State.FIND_ALLIANCE_TAB)

    def _step_find_team(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        now = time.monotonic()
        self._maybe_scroll_rally_page(device, now)
        allow_ranges = self._get_enter_y_valid_ranges()
        enter_btn_x = int(self._enter_group_button_x)
        for team_id in sorted(self._teams.keys()):
            if self._team_skip_until.get(team_id, 0.0) > now:
                continue
            for tmpl_name in self._teams[team_id]:
                path = template_paths.get(tmpl_name)
                if not path:
                    continue
                result = vision.match(frame, path)
                if not result.found:
                    continue

                enter_btn_y = int(result.y) + int(self._enter_group_button_y_offset)
                enter_btn_y_ok = self._is_y_in_ranges(enter_btn_y, allow_ranges)
                self._log_team_search(
                    "候选队伍",
                    tmpl_name=tmpl_name,
                    team_id=team_id,
                    team_x=int(result.x),
                    team_y=int(result.y),
                    enter_x=enter_btn_x,
                    enter_y=enter_btn_y,
                    enter_y_ok=enter_btn_y_ok,
                    found_index=self._found_team_count + 1,
                )
                if not enter_btn_y_ok:
                    # 本轮页面位置下坐标已不可信，直接结束本轮查找，等待下一轮重试
                    return

                self._current_team_id = team_id
                logger.info("[打熊] 找到队伍「%s」，队伍编号=%d", tmpl_name, team_id)
                device.click(enter_btn_x, enter_btn_y)
                self._log_team_search(
                    "点击进入编队",
                    tmpl_name=tmpl_name,
                    team_id=team_id,
                    team_x=int(result.x),
                    team_y=int(result.y),
                    enter_x=enter_btn_x,
                    enter_y=enter_btn_y,
                    enter_y_ok=enter_btn_y_ok,
                    found_index=self._found_team_count + 1,
                )
                self._found_team_count += 1
                self._touch()
                self._wait_until = time.monotonic() + self._delay_before_click_group_secs
                self._go_to(State.WAIT_CLICK_GROUP)
                return

    def _maybe_scroll_rally_page(self, device: Device, now: float) -> None:
        if self._rally_scroll_interval_secs <= 0:
            return
        if self._rally_scroll_times <= 0:
            return
        if (now - self._last_rally_scroll_time) < self._rally_scroll_interval_secs:
            return
        x1, y1 = self._rally_scroll_start_point
        x2, y2 = self._rally_scroll_end_point
        duration = max(0.02, float(self._rally_scroll_duration_secs))
        any_ok = False
        for _ in range(self._rally_scroll_times):
            ok = device.swipe(
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                duration=self._rally_scroll_duration_secs,
                blocking=True,
            )
            if ok:
                any_ok = True
        if any_ok:
            self._last_rally_scroll_time = now
            self._wait_until = max(self._wait_until, now + duration)
            self._touch()

    def _scroll_to_rally_page_bottom_on_enter(self, device: Device, reason: str = "") -> None:
        times = int(self._rally_page_enter_scroll_times)
        if times <= 0:
            return
        x1, y1 = self._rally_page_enter_scroll_start_point
        x2, y2 = self._rally_page_enter_scroll_end_point
        duration = max(0.01, float(self._rally_page_enter_scroll_duration_secs))
        logger.info("[打熊] %s，进入集结页后执行到底部滑动 %d 次", reason or "进入集结页", times)
        succeeded = 0
        for _ in range(times):
            ok = device.swipe(int(x1), int(y1), int(x2), int(y2), duration=duration, blocking=True)
            if ok:
                succeeded += 1
        if succeeded > 0:
            now = time.monotonic()
            self._wait_until = max(self._wait_until, now + duration)
            self._touch()

    def _step_wait_click_group(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        pt = self._group_points.get(self._current_team_id)
        if not pt:
            self._team_skip_until[self._current_team_id] = time.monotonic() + self._team_cooldown_secs
            self._go_to(State.VERIFY_RALLY_PAGE)
            return
        device.click(int(pt[0]), int(pt[1]))
        self._touch()
        self._wait_until = time.monotonic() + self._delay_before_click_expedition_secs
        self._go_to(State.CLICK_EXPEDITION)

    def _step_click_expedition(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        x, y = self._expedition_button_point
        device.click(int(x), int(y))
        self._touch()
        self._team_skip_until[self._current_team_id] = time.monotonic() + self._team_cooldown_secs
        self._wait_until = time.monotonic() + self._delay_after_click_expedition_check_secs
        self._verify_back_count = 0
        self._go_to(State.VERIFY_RALLY_PAGE)

    def _step_verify_rally_page(
        self, frame: np.ndarray, device: Device, vision: Vision, template_paths: dict
    ) -> None:
        path = template_paths.get(self._rally_template)
        if path and vision.match(frame, path).found:
            self._scroll_to_rally_page_bottom_on_enter(device, reason="出征后回到集结页")
            self._verify_back_count = 0
            self._go_to(State.FIND_TEAM)
            return

        if self._verify_back_count == 0:
            # 首次未找到集结页，等待后重新检测
            logger.debug("[打熊] 验证集结页：首次未找到，等待 %.1f s 后重试", self._delay_after_verify_miss_secs)
            self._wait_until = time.monotonic() + self._delay_after_verify_miss_secs
            self._verify_back_count = 1
            self._touch()
            return

        if self._verify_back_count <= 2:
            # 第1、2次：尝试返回上一步，等待后重新检测
            logger.debug("[打熊] 验证集结页：第%d次尝试返回上一步", self._verify_back_count)
            try_go_back(
                frame=frame,
                device=device,
                vision=vision,
                template_paths=template_paths,
                tag="打熊",
            )
            self._wait_until = time.monotonic() + self._delay_after_verify_back_secs
            self._verify_back_count += 1
            self._touch()
            return

        # 返回两次后仍未找到集结页，回首页
        logger.info("[打熊] 验证集结页：返回两次后仍未找到，回首页")
        self._verify_back_count = 0
        self._home_nav.reset()
        self._go_to(State.NAVIGATE_HOME)

    def _go_to(self, state: State) -> None:
        self._state_tick_retry.on_state_change(changed=(state != self._state), state_name=state.name)
        self._state = state

    def _touch(self) -> None:
        self._last_action_time = time.monotonic()

    def _retry_current_state_tick(self, scene_desc: str = "") -> bool:
        return self._state_tick_retry.try_retry(state_name=self._state.name, log_prefix="打熊", scene_desc=scene_desc)

    def get_team_cooldowns(self) -> Dict[int, float]:
        """返回每只队伍的剩余冷却秒数，0.0 表示已就绪。"""
        now = time.monotonic()
        return {
            team_id: max(0.0, self._team_skip_until.get(team_id, 0.0) - now)
            for team_id in self._teams
        }

    def clear_team_cooldown(self, team_id: int) -> bool:
        self._refresh_runtime_teams()
        if team_id not in self._teams:
            logger.warning("[打熊] 清空冷却失败：未知队伍ID=%s", team_id)
            return False
        had_cooldown = team_id in self._team_skip_until
        self._team_skip_until.pop(team_id, None)
        return had_cooldown

    def force_refresh_teams(self) -> Dict[int, List[str]]:
        """供 UI 在配置变更后主动触发一次热刷新。"""
        self._refresh_runtime_teams()
        self._last_team_refresh_time = time.monotonic()
        return {k: list(v) for k, v in self._teams.items()}

    def set_monster_mode(self, monster_mode: bool) -> None:
        """切换打怪模式。打怪模式下 enter_y 校验范围始终合并两组，无前两次查找限制。"""
        self._monster_mode = monster_mode

    def update_click_delays(self, delay_group: float, delay_expedition: float) -> None:
        """实时更新「进入编队→编队」和「编队→出征」的等待时长。"""
        self._delay_before_click_group_secs = max(0.0, float(delay_group))
        self._delay_before_click_expedition_secs = max(0.0, float(delay_expedition))

    def _get_enter_y_valid_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = list(self._enter_group_button_y_valid_ranges)
        if self._monster_mode or self._found_team_count < 2:
            ranges.extend(self._first_two_found_enter_y_extra_valid_ranges)
        return ranges

    @staticmethod
    def _parse_y_ranges(raw_ranges: object, default_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        parsed: List[Tuple[int, int]] = []
        if isinstance(raw_ranges, list):
            for item in raw_ranges:
                if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, (int, float)) for v in item):
                    a, b = int(item[0]), int(item[1])
                    parsed.append((min(a, b), max(a, b)))
        return parsed or list(default_ranges)

    @staticmethod
    def _is_y_in_ranges(y: int, ranges: List[Tuple[int, int]]) -> bool:
        return any(start <= y <= end for start, end in ranges)

    @staticmethod
    def _normalize_teams_cfg(raw_teams: object) -> Dict[int, List[str]]:
        normalized: Dict[int, List[str]] = {}
        src = raw_teams if isinstance(raw_teams, dict) else _DEFAULT_TEAMS
        # 统一将所有键转为 int，支持任意队伍编号（不再硬编码 1/2/3）
        all_int_keys: List[int] = []
        for k in src:
            try:
                all_int_keys.append(int(k))
            except (ValueError, TypeError):
                pass
        for team_id in sorted(set(all_int_keys)):
            values = src.get(team_id, src.get(str(team_id), []))
            if not isinstance(values, list):
                values = []
            normalized[team_id] = [str(v) for v in values if isinstance(v, str) and str(v).strip()]
        return normalized

    def _refresh_runtime_teams(self) -> None:
        cfg = merge_task_settings(self._config_ref, "bear_attack_task")
        latest_teams = self._normalize_teams_cfg(cfg.get("teams", _DEFAULT_TEAMS))
        if latest_teams == self._teams:
            return
        self._teams = latest_teams
        self._all_team_templates = list(
            dict.fromkeys(t for tlist in self._teams.values() for t in tlist)
        )
        self.required_templates = list(
            dict.fromkeys([*self._non_team_required_templates, *self._all_team_templates])
        )
        self._state_templates[State.FIND_TEAM] = self._all_team_templates
        logger.info("[打熊] 队伍配置已热更新: %s", self._teams)

    def _build_team_search_logger(self) -> logging.Logger:
        logger_name = f"{__name__}.team_search"
        task_logger = logging.getLogger(logger_name)
        task_logger.setLevel(logging.INFO)
        task_logger.propagate = False
        if task_logger.handlers:
            return task_logger
        logs_dir = Path.cwd() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "bear_attack_team_search.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        task_logger.addHandler(file_handler)
        return task_logger

    def _log_team_search(
        self,
        stage: str,
        tmpl_name: str,
        team_id: int,
        team_x: int,
        team_y: int,
        enter_x: int,
        enter_y: int,
        enter_y_ok: bool,
        found_index: int,
    ) -> None:
        self._team_search_logger.info(
            "[%s] 第%d次查找到队伍 模板=%s 队伍ID=%d 队伍点=(%d,%d) 进入编队点=(%d,%d) 进入编队Y范围校验=%s",
            stage,
            found_index,
            tmpl_name,
            team_id,
            team_x,
            team_y,
            enter_x,
            enter_y,
            enter_y_ok,
        )

    @property
    def state_label(self) -> str:
        labels = {
            State.CHECK_RALLY_PAGE: "检测集结页",
            State.NAVIGATE_HOME: "返回首页",
            State.FIND_ALLIANCE_TAB: "查找联盟tab",
            State.FIND_SWORDS: "查找两把剑",
            State.FIND_TEAM: "查找队伍",
            State.WAIT_CLICK_GROUP: "点击编队",
            State.CLICK_EXPEDITION: "点击出征",
            State.VERIFY_RALLY_PAGE: "回集结页检查",
        }
        return labels.get(self._state, self._state.name)
