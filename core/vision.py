"""
Vision：封装 OpenCV 模板匹配。
入参：大图(BGR)、模板路径或图像、阈值
出参：MatchResult
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np

from core.shared_state import MatchResult

CV_METHODS = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED":  cv2.TM_CCORR_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
}

# 模板图像内存缓存，避免重复从磁盘读取（存储灰度图）
_template_cache: Dict[str, np.ndarray] = {}

# frame 灰度图缓存，用 id(frame) 做 key 避免重复转换同一帧
_gray_frame_cache: Dict[int, np.ndarray] = {}


def _load_template(path: str, scale: float = 1.0, grayscale: bool = True) -> Optional[np.ndarray]:
    """加载并缓存模板图像（灰度或 BGR），scale 变化时单独缓存。"""
    cache_key = f"{path}@{scale:.4f}@{'g' if grayscale else 'c'}"
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    raw = np.frombuffer(open(path, "rb").read(), np.uint8)
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(raw, flags)
    if img is None:
        return None

    if abs(scale - 1.0) > 1e-4:
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h))

    _template_cache[cache_key] = img
    return img


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """将 BGR frame 转换为灰度图，同一帧对象只转换一次。"""
    fid = id(frame)
    gray = _gray_frame_cache.get(fid)
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # 只保留最近一帧，避免内存泄漏
        _gray_frame_cache.clear()
        _gray_frame_cache[fid] = gray
    return gray


def clear_cache() -> None:
    _template_cache.clear()
    _gray_frame_cache.clear()


class Vision:
    """模板匹配工具类"""

    def __init__(self, config: dict | None = None, ctx=None) -> None:
        """
        :param config: 全局配置字典
        :param ctx:    SharedContext，若传入则优先读取其 vision_threshold/vision_scale/vision_method
        """
        cfg = config or {}
        tracker_cfg = cfg.get("scrcpy_tracker", {})
        method_name = tracker_cfg.get("method", "TM_CCOEFF_NORMED")
        self._default_method = CV_METHODS.get(method_name, cv2.TM_CCOEFF_NORMED)
        self._default_threshold: float = float(tracker_cfg.get("threshold", 0.82))
        self._default_scale: float = float(tracker_cfg.get("scale", 1.0))
        self._ctx = ctx  # 若不为 None，运行时从 ctx 读取参数

        # 兼容旧属性名
        self._method = self._default_method
        self._scale = self._default_scale

    @property
    def _active_threshold(self) -> float:
        if self._ctx is not None:
            return self._ctx.vision_threshold
        return self._default_threshold

    @property
    def _active_scale(self) -> float:
        if self._ctx is not None:
            return self._ctx.vision_scale
        return self._default_scale

    @property
    def _active_method(self) -> int:
        if self._ctx is not None:
            return CV_METHODS.get(self._ctx.vision_method, cv2.TM_CCOEFF_NORMED)
        return self._default_method

    def match(
        self,
        frame: np.ndarray,
        template_path: str,
        threshold: Optional[float] = None,
        scale: Optional[float] = None,
        use_grayscale: bool = True,
    ) -> MatchResult:
        """
        在 frame 中匹配 template_path 指定的模板。
        threshold/scale 不传时优先读取 SharedContext 中的值（由 Tracker UI 实时调整）。
        use_grayscale=True（默认）时大图与模板均按灰度匹配；False 时按 BGR 三通道匹配。
        返回 MatchResult（若匹配失败返回 found=False 的结果）。
        """
        thresh = threshold if threshold is not None else self._active_threshold
        sc = scale if scale is not None else self._active_scale

        try:
            tmpl = _load_template(template_path, sc, grayscale=use_grayscale)
            if tmpl is None:
                return MatchResult(0.0, 0, 0, 0, 0, False)

            th, tw = tmpl.shape[:2]
            if th > frame.shape[0] or tw > frame.shape[1]:
                return MatchResult(0.0, 0, 0, th, tw, False)

            if use_grayscale:
                work_frame = _to_gray(frame)
            else:
                if frame.ndim == 2:
                    work_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    work_frame = frame
            method = self._active_method
            res_map = cv2.matchTemplate(work_frame, tmpl, method)
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res_map)

            if method == cv2.TM_SQDIFF_NORMED:
                conf, loc = 1.0 - min_v, min_loc
            else:
                conf, loc = max_v, max_loc

            cx = loc[0] + tw // 2
            cy = loc[1] + th // 2
            found = conf >= thresh

            return MatchResult(confidence=conf, x=cx, y=cy, w=tw, h=th, found=found)

        except Exception:
            return MatchResult(0.0, 0, 0, 0, 0, False)

    def match_all(
        self,
        frame: np.ndarray,
        template_paths: Dict[str, str],
        threshold: Optional[float] = None,
        use_grayscale: bool = True,
    ) -> Dict[str, MatchResult]:
        """批量匹配，返回 {名称: MatchResult} 字典"""
        return {
            name: self.match(frame, path, threshold, use_grayscale=use_grayscale)
            for name, path in template_paths.items()
        }
