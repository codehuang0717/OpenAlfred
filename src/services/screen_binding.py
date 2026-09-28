"""Persist desktop ownership only after an interactive Windows confirmation."""

import ctypes
import ipaddress
import os
import sqlite3
import threading
from contextlib import closing
from urllib.parse import urlparse

from core.config import config
from utils.auth_utils import require_explicit_user_id

_confirmation_lock = threading.Lock()


def binding_path():
    return config.PROJECT_ROOT / "data/screenpipe/device-binding.sqlite3"


def configured_owner() -> str | None:
    """Read on every access so FastAPI, LangGraph and Supervisor need no restart."""
    saved = None
    path = binding_path()
    if path.exists():
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            row = db.execute("SELECT user_id FROM device_binding WHERE id = 1").fetchone()
            if row:
                saved = require_explicit_user_id(row[0])
    explicit = config.SCREEN_MONITOR_USER_ID
    if explicit:
        explicit = require_explicit_user_id(explicit)
        if saved and saved != explicit:
            raise ValueError("本机绑定与 SCREEN_MONITOR_USER_ID 冲突，请由本机管理员处理")
        return explicit
    return saved


def require_local_binding_request(peer: str, headers) -> None:
    """Reject known remote requests; headers alone never authorize a binding."""
    def loopback(value):
        try:
            address = ipaddress.ip_address(value)
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                address = address.ipv4_mapped
            return address.is_loopback
        except ValueError:
            return False

    if not loopback(peer):
        raise PermissionError("请在这台电脑上打开 http://localhost:3000 完成本机绑定")
    for key in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip"):
        if key in headers and not all(loopback(item.strip()) for item in headers[key].split(",")):
            raise PermissionError("不允许通过远程代理认领本机屏幕")
    if "forwarded" in headers:
        raise PermissionError("绑定请求不支持 Forwarded 代理，请直接访问 localhost")
    origin = urlparse(headers.get("origin", ""))
    if origin.scheme not in {"http", "https"} or origin.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise PermissionError("请从本机 localhost 页面发起绑定")


def confirm_on_desktop(username: str, user_id: str) -> bool:
    if os.name != "nt":
        raise RuntimeError("本机绑定确认目前仅支持 Windows 桌面")
    # This OS dialog, not a spoofable HTTP header, proves local approval.
    # Default button is No; remote login or browser JavaScript cannot accept it.
    text = (f"是否将这台电脑的屏幕绑定到以下 OpenAlfred 账号？\n\n"
            f"用户名：{username}\n账号 ID：{user_id}\n\n"
            "此账号之后可请求查看本机屏幕。绑定不会开启连续录屏。\n"
            "只有你刚刚在本机页面点击了绑定按钮，才应选择“是”。")
    message_box = ctypes.windll.user32.MessageBoxW
    message_box.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    message_box.restype = ctypes.c_int
    result = message_box(None, text, "OpenAlfred — 本机屏幕账号绑定", 0x4 | 0x30 | 0x100 | 0x10000)
    if result == 0:
        raise RuntimeError("无法显示 Windows 本机确认框，请使用交互式桌面运行服务")
    return result == 6


def bind_on_desktop(user_id: str, username: str) -> None:
    user_id = require_explicit_user_id(user_id)
    if not _confirmation_lock.acquire(blocking=False):
        raise ValueError("已有绑定确认正在进行，请先处理 Windows 确认框")
    try:
        owner = configured_owner()
        if owner:
            if owner != user_id:
                raise PermissionError("本机已绑定其他账号，不能覆盖绑定")
            return  # Idempotent; no second dialog or change to preferences.
        if not confirm_on_desktop(username, user_id):
            raise PermissionError("已取消本机绑定，未修改设置")
        path = binding_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as db:
            # The singleton primary key + transaction also protect multiple API workers.
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS device_binding (id INTEGER PRIMARY KEY CHECK(id = 1), user_id TEXT NOT NULL)")
                db.execute("INSERT OR IGNORE INTO device_binding VALUES (1, ?)", (user_id,))
                actual = db.execute("SELECT user_id FROM device_binding WHERE id = 1").fetchone()[0]
                if actual != user_id:
                    raise PermissionError("本机已被另一个账号绑定，未覆盖原绑定")
    finally:
        _confirmation_lock.release()
