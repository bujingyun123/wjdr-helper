from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

logger = logging.getLogger(__name__)
_msg_logger = logging.getLogger(f"{__name__}.message")


def _setup_message_logger() -> None:
    if _msg_logger.handlers:
        return
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(logs_dir / "feishu_bot.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    _msg_logger.addHandler(handler)
    _msg_logger.setLevel(logging.INFO)
    _msg_logger.propagate = False


@dataclass
class IncomingCommand:
    command_text: str
    chat_type: str
    chat_id: str
    sender_id: str


class FeishuBotService:
    """
    飞书机器人服务：
    - 长连接接收消息
    - 将文本命令放入线程安全队列（由 Tk 主线程消费）
    - 支持主动发送消息到最后一次会话
    """

    def __init__(self, app_id: str, app_secret: str, default_chat_id: str = "") -> None:
        _setup_message_logger()
        self._app_id = (app_id or "").strip()
        self._app_secret = (app_secret or "").strip()
        self._default_chat_id = (default_chat_id or "").strip()
        self._incoming_queue: "queue.Queue[IncomingCommand]" = queue.Queue()
        self._lock = threading.Lock()
        self._last_chat_id: str = ""
        self._last_chat_type: str = "p2p"
        self._ws_client = None
        self._client = None
        self._started = False

        if self.enabled:
            self._client = (
                lark.Client.builder()
                .app_id(self._app_id)
                .app_secret(self._app_secret)
                .build()
            )
            _msg_logger.info("飞书机器人初始化完成")

    @property
    def enabled(self) -> bool:
        return bool(self._app_id and self._app_secret)

    def start(self) -> None:
        if not self.enabled:
            logger.info("飞书机器人未启用：缺少 app_id 或 app_secret。")
            _msg_logger.info("飞书机器人未启用：缺少 app_id 或 app_secret")
            return
        if self._started:
            _msg_logger.info("飞书机器人重复启动请求已忽略")
            return
        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message_receive)
            .build()
        )
        self._ws_client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        self._started = True
        thread = threading.Thread(target=self._ws_client.start, daemon=True)
        thread.start()
        logger.info("飞书机器人长连接已启动。")
        _msg_logger.info("飞书机器人长连接已启动")

    def _on_message_receive(self, data: P2ImMessageReceiveV1) -> None:
        text = ""
        msg_type = data.event.message.message_type
        if msg_type == "text":
            try:
                text = json.loads(data.event.message.content).get("text", "")
            except Exception:
                text = ""
        if not text:
            _msg_logger.info(
                "收到非文本消息，chat_id=%s chat_type=%s",
                data.event.message.chat_id,
                data.event.message.chat_type,
            )
            self._safe_reply(
                chat_id=data.event.message.chat_id,
                chat_type=data.event.message.chat_type,
                text="仅支持文本命令，发送“帮助”查看可用命令。",
            )
            return

        cmd = IncomingCommand(
            command_text=text.strip(),
            chat_type=data.event.message.chat_type,
            chat_id=data.event.message.chat_id,
            sender_id=data.event.sender.sender_id.open_id,
        )
        with self._lock:
            self._last_chat_id = cmd.chat_id
            self._last_chat_type = cmd.chat_type
        self._incoming_queue.put(cmd)
        _msg_logger.info(
            "收到命令: chat_id=%s sender=%s text=%s",
            cmd.chat_id,
            cmd.sender_id,
            self._brief(cmd.command_text),
        )

    def poll_command(self) -> Optional[IncomingCommand]:
        try:
            return self._incoming_queue.get_nowait()
        except queue.Empty:
            return None

    def reply_to(self, chat_id: str, chat_type: str, text: str) -> bool:
        if not self.enabled or not self._client:
            _msg_logger.warning("reply_to 失败：机器人未启用")
            return False
        _msg_logger.info(
            "回复消息: chat_id=%s chat_type=%s text=%s",
            chat_id,
            chat_type,
            self._brief(text),
        )
        return self._safe_reply(chat_id=chat_id, chat_type=chat_type, text=text)

    def notify_last_chat(self, text: str) -> bool:
        with self._lock:
            chat_id = self._last_chat_id
            chat_type = self._last_chat_type
        if not chat_id:
            if self._default_chat_id:
                chat_id = self._default_chat_id
                chat_type = "group"
                _msg_logger.info("主动通知使用默认会话: chat_id=%s text=%s", chat_id, self._brief(text))
            else:
                _msg_logger.info("主动通知跳过：尚无最近会话，且未配置 default_chat_id")
                return False
        _msg_logger.info("主动通知: chat_id=%s text=%s", chat_id, self._brief(text))
        return self._safe_reply(chat_id=chat_id, chat_type=chat_type, text=text)

    def _safe_reply(self, chat_id: str, chat_type: str, text: str) -> bool:
        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(json.dumps({"text": text}, ensure_ascii=False))
                    .build()
                )
                .build()
            )
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "飞书发消息失败: code=%s msg=%s log_id=%s",
                    response.code,
                    response.msg,
                    response.get_log_id(),
                )
                _msg_logger.error(
                    "发送失败: chat_id=%s code=%s msg=%s",
                    chat_id,
                    response.code,
                    response.msg,
                )
                return False
            _msg_logger.info("发送成功: chat_id=%s", chat_id)
            return True
        except Exception:
            logger.exception("飞书发消息异常")
            _msg_logger.exception("发送异常: chat_id=%s", chat_id)
            return False

    @staticmethod
    def _brief(text: str, max_len: int = 200) -> str:
        s = (text or "").replace("\n", "\\n")
        if len(s) <= max_len:
            return s
        return s[:max_len] + "...(truncated)"


def build_from_config(config: dict) -> Optional[FeishuBotService]:
    bot_cfg = (config or {}).get("feishu_bot") or {}
    if not isinstance(bot_cfg, dict):
        bot_cfg = {}
    if not bool(bot_cfg.get("enabled", False)):
        return None
    service = FeishuBotService(
        app_id=str(bot_cfg.get("app_id", "")).strip(),
        app_secret=str(bot_cfg.get("app_secret", "")).strip(),
        default_chat_id=str(bot_cfg.get("default_chat_id", "")).strip(),
    )
    if not service.enabled:
        logger.warning("feishu_bot.enabled=true 但 app_id/app_secret 不完整，已跳过。")
        return None
    return service
