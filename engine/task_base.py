"""
BaseTask：所有业务任务的抽象基类。

规则：
- run_step() 每次只做「一小步」，立即返回，不阻塞主线程
- 绝对禁止在 run_step() 中使用 while True 或长时间 time.sleep()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from core.device import Device
from core.vision import Vision


class BaseTask(ABC):
    """
    任务基类。子类需声明：
        priority          : int  — 数值越大优先级越高
        name              : str  — 任务展示名称
        required_templates: List[str]  — 本任务用到的模板名称列表
    """

    priority: int = 0
    name: str = "未命名任务"
    required_templates: List[str] = []

    def __init__(self) -> None:
        self.is_finished: bool = False
        self.is_enabled: bool = True
        # 任务专属 tick 间隔（毫秒）；None 表示沿用调度器全局 TICK_MS
        self.tick_ms: Optional[int] = None

    @property
    def step_template_names(self) -> List[str]:
        """
        当前步骤实际需要匹配的模板名称列表。

        默认返回 required_templates（全量），子类可按当前状态覆写，
        让调度器只激活并传入本步骤真正用到的模板，避免无效匹配。
        """
        return self.required_templates

    @abstractmethod
    def run_step(
        self,
        frame: np.ndarray,
        device: Device,
        vision: Vision,
        template_paths: dict,
    ) -> None:
        """
        执行任务的一个步骤。
        :param frame:          当前帧（BGR）
        :param device:         设备控制对象
        :param vision:         视觉识别对象
        :param template_paths: {模板名称: 文件路径} 映射
        """

    def reset(self) -> None:
        """重置任务到初始状态（可选重写）"""
        self.is_finished = False
