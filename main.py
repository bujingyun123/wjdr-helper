#!/usr/bin/env python3
"""
main.py — 游戏自动化脚本唯一启动入口

启动后会打开两个窗口：
  1. 控制面板（主窗口）：管理设备连接、任务开关、查看状态日志
  2. Tracker 工具（子窗口）：实时画面 + 模板识别可视化（调试用）

用法：
    python main.py
"""
from __future__ import annotations

import logging
import os
import sys
import tkinter as tk
from pathlib import Path

# ── 确保项目根目录在 sys.path ─────────────────────────────
_ROOT = Path(__file__).resolve().parent   # resolve() 保证绝对路径
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Windows 控制台强制 UTF-8，避免中文日志显示为 ??? ────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 无 reconfigure，忽略

# ── 可选：PyAV 兼容补丁（scrcpy-client 依赖）────────────────
try:
    import av as _av
    if not hasattr(_av, 'AVError') and hasattr(_av, 'FFmpegError'):
        _av.AVError = _av.FFmpegError
except ImportError:
    pass


def _load_config() -> dict:
    config_path = _ROOT / "config.yaml"
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logging.warning("未安装 pyyaml，将使用默认配置。执行: pip install pyyaml")
        return {}
    except FileNotFoundError:
        logging.warning("config.yaml 未找到，将使用默认配置")
        return {}


def _setup_logging(config: dict) -> None:
    level_name = config.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 压低第三方网络库日志，避免心跳包刷屏
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> None:
    config = _load_config()
    _setup_logging(config)

    # ── 更新调度器全局参数（从配置读取）──────────────────
    import engine.scheduler as _sched_mod
    sched_cfg = config.get("scheduler", {})
    _sched_mod.TICK_MS = sched_cfg.get("tick_ms", 300)
    _sched_mod.TICK_COUNT_MAX = sched_cfg.get("tick_count_max", 100_000)

    from core.shared_state import SharedContext
    from core.device import Device
    from core.vision import Vision
    from engine.scheduler import Scheduler
    from integrations.feishu_bot import build_from_config
    from ui.control_panel import ControlPanel

    # ── 核心对象初始化 ────────────────────────────────────
    ctx = SharedContext()
    # 视觉参数初始值从 scrcpy_tracker 节读取，Tracker 启动后会以相同配置再次同步到 ctx
    tracker_cfg = config.get("scrcpy_tracker", {})
    ctx.vision_threshold = float(tracker_cfg.get("threshold", 0.82))
    ctx.vision_scale     = float(tracker_cfg.get("scale", 1.0))
    ctx.vision_method    = tracker_cfg.get("method", "TM_CCOEFF_NORMED")

    device = Device(ctx, config)
    vision = Vision(config, ctx=ctx)  # Vision 持有 ctx，运行时动态读取参数

    # ── 创建主 Tk 窗口（控制面板）──────────────────────────
    root = tk.Tk()

    scheduler = Scheduler(
        ctx=ctx,
        device=device,
        vision=vision,
        root=root,
        template_dir=str(_ROOT / "assets" / "templates"),
    )

    panel = ControlPanel(
        root=root,
        scheduler=scheduler,
        device=device,
        ctx=ctx,
        config=config,
    )

    # ── 飞书机器人（接收命令 + 主动通知）─────────────────────
    feishu_bot = build_from_config(config)
    if feishu_bot:
        feishu_bot.start()
        panel.set_feishu_notify(feishu_bot.notify_last_chat)

        def _on_panel_event(message: str) -> None:
            feishu_bot.notify_last_chat(f"[任务通知] {message}")

        panel.add_event_listener(_on_panel_event)

        def _poll_feishu_commands() -> None:
            cmd = feishu_bot.poll_command()
            while cmd is not None:
                try:
                    logging.info(
                        "收到飞书命令: sender=%s chat_id=%s text=%s",
                        cmd.sender_id,
                        cmd.chat_id,
                        cmd.command_text,
                    )
                    result_text = panel.handle_remote_command(cmd.command_text)
                    logging.info("飞书命令执行结果: %s", result_text)
                    feishu_bot.reply_to(
                        chat_id=cmd.chat_id,
                        chat_type=cmd.chat_type,
                        text=result_text,
                    )
                except Exception as exc:
                    logging.exception("处理飞书命令异常: %s", exc)
                    feishu_bot.reply_to(
                        chat_id=cmd.chat_id,
                        chat_type=cmd.chat_type,
                        text=f"命令处理失败: {exc}",
                    )
                cmd = feishu_bot.poll_command()
            root.after(500, _poll_feishu_commands)

        root.after(500, _poll_feishu_commands)

    # ── 打开 Tracker 工具窗口（集成模式）────────────────────
    tracker_win = tk.Toplevel(root)
    tracker_win.title("实时屏幕查看与图标跟踪  [集成模式]")

    # 延迟导入，避免在 sys.path 未配置好前导入
    from tools.scrcpy_tracker import ScrcpyTrackerApp
    tracker_app = ScrcpyTrackerApp(tracker_win, shared_ctx=ctx)
    # 让 Tracker 点击事件复用主流程 Device 通道
    tracker_app.external_click_handler = device.click

    # ── 调度器自动启动 ────────────────────────────────────
    scheduler.start()

    # ── 窗口关闭处理 ──────────────────────────────────────
    def on_close():
        panel.on_close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # 将 Tracker 窗口置于控制面板旁边
    root.update_idletasks()
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_w = root.winfo_width()
    tracker_win.geometry(f"+{root_x + root_w + 10}+{root_y}")

    root.mainloop()


if __name__ == "__main__":
    main()
