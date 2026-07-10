# -*- coding: utf-8 -*-
"""
任务配置合并：tasks.defaults（全局）与 tasks.<task_key>（任务）合并，任务键覆盖全局。
优先级：任务 > 全局 defaults > 代码内默认值（由各任务自行 .get）。
"""
from __future__ import annotations

from typing import Any, Dict


def merge_task_settings(config: dict | None, task_key: str) -> Dict[str, Any]:
    tasks = (config or {}).get("tasks")
    if not isinstance(tasks, dict):
        tasks = {}
    global_cfg = tasks.get("defaults")
    if not isinstance(global_cfg, dict):
        global_cfg = {}
    task_cfg = tasks.get(task_key)
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    return {**global_cfg, **task_cfg}
