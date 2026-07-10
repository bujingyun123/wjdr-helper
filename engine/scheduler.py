"""
Scheduler：Tick-based 任务调度器。
- 维护优先级任务队列
- 通过 root.after() 驱动，每 TICK_MS 执行一次
- 每 tick 在「当前最高优先级」的已启用未完成任务中选出其一，调用其 run_step()
- 若多个任务优先级相同且同为当前全局最高，则按队列顺序轮询（公平调度），避免后加任务永远得不到 tick
  （例如「专属任务刷新」与「公共任务抢」均为 priority=29 时）
- 更新 SharedContext.active_template_names
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk

from core.device import Device
from core.shared_state import MatchResult, SharedContext
from core.vision import Vision
from engine.task_base import BaseTask

logger = logging.getLogger(__name__)

TICK_MS = 300           # 全局默认帧间隔毫秒数
TICK_COUNT_MAX = 100_000  # tick_count 归零阈值（达到后从 0 重新计数）


class Scheduler:
    """
    单线程优先级任务调度器，运行在 Tkinter 主线程中。
    """

    def __init__(
        self,
        ctx: SharedContext,
        device: Device,
        vision: Vision,
        root: tk.Tk,
        template_dir: str = "",
    ) -> None:
        self._ctx = ctx
        self._device = device
        self._vision = vision
        self._root = root
        self._template_dir = Path(template_dir) if template_dir else Path("assets/templates")
        self._tasks: List[BaseTask] = []
        self._running: bool = False
        self._tick_id: Optional[str] = None
        self._last_task_name: str = ""
        self._current_task: Optional[BaseTask] = None  # 上一次 tick 执行的任务（供 _tick_loop 读 tick_ms）
        # 最高优先级档内多个任务时轮询用（同 priority 公平分时）
        self._same_max_priority_rr: int = 0

        # 供 UI 观察
        self.tick_count: int = 0
        self.current_task_name: str = "无"

    # ── 任务管理 ──────────────────────────────────────────

    def add_task(self, task: BaseTask) -> None:
        if not any(t is task for t in self._tasks):
            self._tasks.append(task)
            self._tasks.sort(key=lambda t: t.priority, reverse=True)
            logger.info("任务已加入队列: %s (priority=%d)", task.name, task.priority)

    def remove_task(self, task: BaseTask) -> None:
        self._tasks = [t for t in self._tasks if t is not task]
        logger.info("任务已移除: %s", task.name)

    def remove_task_by_name(self, name: str) -> None:
        self._tasks = [t for t in self._tasks if t.name != name]

    def get_task_by_name(self, name: str) -> Optional[BaseTask]:
        for t in self._tasks:
            if t.name == name:
                return t
        return None

    # ── 调度控制 ──────────────────────────────────────────

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._tick_loop()

    def stop(self) -> None:
        self._running = False
        if self._tick_id:
            try:
                self._root.after_cancel(self._tick_id)
            except Exception:
                pass

    # ── Tick 主循环 ────────────────────────────────────────

    def _tick_loop(self) -> None:
        if self._running:
            self._tick()
            # 优先使用当前任务的专属间隔，其次全局 TICK_MS
            interval = (
                self._current_task.tick_ms
                if self._current_task is not None and self._current_task.tick_ms is not None
                else TICK_MS
            )
            self._tick_id = self._root.after(interval, self._tick_loop)

    def _tick(self) -> None:
        self.tick_count += 1
        if self.tick_count >= TICK_COUNT_MAX:
            self.tick_count = 0

        if getattr(self._ctx, "scheduler_paused_for_game_autolaunch", False):
            return

        frame = self._ctx.get_frame()
        if frame is None:
            return

        # 选出当前最高优先级的已启用任务；同优先级多个时轮询，避免永远只跑列表中的第一个
        current_task: Optional[BaseTask] = None
        enabled_alive: List[BaseTask] = [
            t for t in self._tasks if t.is_enabled and not t.is_finished
        ]
        if enabled_alive:
            max_pri = max(t.priority for t in enabled_alive)
            band = [t for t in enabled_alive if t.priority == max_pri]
            if len(band) == 1:
                current_task = band[0]
            else:
                i = self._same_max_priority_rr % len(band)
                current_task = band[i]
                self._same_max_priority_rr += 1
                logger.debug(
                    "调度: 最高优先级=%d 共 %d 个任务，本轮 #%d → %s",
                    max_pri,
                    len(band),
                    i,
                    current_task.name,
                )

        # 移除已完成的一次性任务，并同步置为未启用，避免上层状态判断为“仍在运行”
        for t in self._tasks:
            if t.is_finished and t.is_enabled:
                t.is_enabled = False
                logger.info("任务已完成并停用: %s", t.name)
        self._tasks = [t for t in self._tasks if not t.is_finished]
        # 重新排序（移除后顺序不变，但保险起见）
        self._tasks.sort(key=lambda t: t.priority, reverse=True)

        # 记录本次执行的任务，供 _tick_loop 计算下一次间隔
        self._current_task = current_task

        if current_task is None:
            if self.current_task_name != "无":
                self.current_task_name = "无"
                self._ctx.active_template_names = set()
            return

        # 更新激活模板（只激活当前步骤需要的模板子集）
        self.current_task_name = current_task.name
        step_names = current_task.step_template_names
        self._ctx.active_template_names = set(step_names)
        self._ctx.sync_enabled_from_active()

        # 构建本次用到的模板路径表（仅当前步骤所需）
        template_paths = self._build_template_paths(step_names)

        # 执行一步
        try:
            t0 = time.monotonic()
            current_task.run_step(frame, self._device, self._vision, template_paths)
            elapsed = time.monotonic() - t0
            if elapsed > 0.2:
                logger.warning("run_step 耗时过长: %.3fs (%s)", elapsed, current_task.name)
        except Exception as e:
            logger.exception("run_step 异常 (%s): %s", current_task.name, e)

        # TrackerUI 在集成模式下由自己的后台线程（_run_matching_on_frame）更新展示结果，
        # 此处不再重复对同一批模板做二次 matchTemplate，避免每 tick 双倍 CPU 消耗。

    def _build_template_paths(self, names: List[str]) -> Dict[str, str]:
        """根据模板名称，从已注册模板列表或默认目录中查找路径"""
        result: Dict[str, str] = {}
        # 先从 SharedContext.templates 中查找（与 tracker 共用）
        registered = {t.name: t.path for t in self._ctx.get_templates()}
        for name in names:
            if name in registered:
                path = registered[name]
                # 注册模板路径可能已不存在（文件被移动/删除），做存在性检查
                if Path(path).exists():
                    result[name] = path
                else:
                    logger.warning("注册模板文件不存在，跳过: %s → %s", name, path)
            if name not in result:
                # 从 assets/templates/ 目录按名称匹配
                found = False
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = self._template_dir / f"{name}{ext}"
                    if candidate.exists():
                        result[name] = str(candidate)
                        found = True
                        break
                if not found:
                    logger.debug(
                        "模板未找到: '%s'  (已注册: %s, 查找目录: %s)",
                        name,
                        list(registered.keys()),
                        self._template_dir,
                    )
        return result

