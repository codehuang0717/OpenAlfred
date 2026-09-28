"""Explicit local-device ownership and managed Screenpipe lifecycle.

Never adopts a server merely because port 3030 answers. Legacy unowned data
is not migrated. The desktop owner must be explicitly approved on the machine.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from contextlib import contextmanager
from urllib.parse import urlparse

import httpx
import psutil

from core.config import config
from db.settings import get_setting, set_setting
from utils.auth_utils import require_explicit_user_id
from services.screen_binding import configured_owner
from services.screenpipe_models import verify_models


def require_screen_owner(user_id: str) -> str:
    owner = configured_owner()
    if owner is None:
        raise PermissionError("本机未绑定账号，请在主管设置中绑定当前账号")
    user_id = require_explicit_user_id(user_id)
    if user_id != owner:
        raise PermissionError("当前账号未绑定这台电脑，不能控制或读取屏幕")
    return owner


@contextmanager
def device_lock():
    """One supervisor per desktop, including when launched manually twice."""
    import msvcrt
    directory = config.PROJECT_ROOT / "data/screenpipe"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "supervisor.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("本机已有 Supervisor，请勿重复启动") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def data_directory(user_id: str) -> Path:
    owner = require_screen_owner(user_id)
    return config.PROJECT_ROOT / "data/screenpipe/users" / hashlib.sha256(owner.encode()).hexdigest()


def server_address() -> tuple[str, int]:
    url = urlparse(config.SCREENPIPE_URL)
    if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"}
            or url.username or url.password or url.path not in {"", "/"}
            or url.query or url.fragment):
        raise ValueError("SCREENPIPE_URL 必须是本机 http://127.0.0.1:<port>")
    if not url.port:
        raise ValueError("SCREENPIPE_URL 必须明确指定端口")
    return "127.0.0.1", url.port


def require_free_port(host: str, port: int) -> None:
    """Reject an occupied local endpoint without waiting for a TCP handshake."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError(
            f"端口 {port} 无法绑定到 {host}：{exc}；拒绝接管未知 Screenpipe"
        ) from exc


async def read_preferences(user_id: str) -> dict:
    owner = require_screen_owner(user_id)
    raw = await get_setting(f"screen_monitor:v1:{owner}")
    if raw is None:
        return {"recording_enabled": False, "smart_supervision_enabled": False}
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {"recording_enabled", "smart_supervision_enabled"}
            or any(type(v) is not bool for v in value.values())
            or (value["smart_supervision_enabled"] and not value["recording_enabled"])):
        raise ValueError("屏幕监控配置损坏，请重新保存配置")
    return value


async def write_preferences(user_id: str, recording: bool, smart: bool) -> None:
    owner = require_screen_owner(user_id)
    if type(recording) is not bool or type(smart) is not bool or (smart and not recording):
        raise ValueError("智能监督需要先开启屏幕记录")
    await set_setting(f"screen_monitor:v1:{owner}", json.dumps({
        "recording_enabled": recording, "smart_supervision_enabled": smart,
    }))


def process_status(pid: int, user_id: str) -> str:
    """Separate startup from a foreign process or an exited owned process."""
    try:
        proc = psutil.Process(pid)
        args = proc.cmdline()
        expected = data_directory(user_id).resolve()
        if (Path(proc.exe()).resolve() != Path(config.SCREENPIPE_EXE).resolve()
                or "--data-dir" not in args
                or Path(args[args.index("--data-dir") + 1]).resolve() != expected):
            return "mismatch"
        if any(c.status == psutil.CONN_LISTEN and c.laddr.port == server_address()[1]
               for c in proc.net_connections(kind="tcp")):
            return "ready"
        return "starting"
    except psutil.NoSuchProcess:
        return "exited"
    except (psutil.Error, OSError, ValueError, IndexError):
        return "mismatch"


def verified_process(pid: int, user_id: str) -> bool:
    return process_status(pid, user_id) == "ready"


async def healthy() -> bool:
    host, port = server_address()
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.get(f"http://{host}:{port}/health", timeout=2)
        response.raise_for_status()
        return response.json().get("frame_status") == "ok"


async def monitor_status(user_id: str) -> dict:
    owner = require_screen_owner(user_id)
    prefs = await read_preferences(owner)
    raw = await get_setting(f"screen_monitor:runtime:{owner}")
    state = json.loads(raw) if raw else {}
    alive = 0 <= time.time() - state.get("heartbeat", 0) < 20
    running = False
    error = state.get("error") if alive else "Supervisor 未运行或心跳已过期"
    if alive and state.get("screenpipe_pid"):
        process = await asyncio.to_thread(process_status, state["screenpipe_pid"], owner)
        if process == "ready":
            try:
                running = await healthy()
                if not running:
                    error = "Screenpipe OCR 尚未就绪，请检查 screenpipe.log"
            except (httpx.HTTPError, ValueError) as exc:
                error = f"Screenpipe 健康检查失败：{type(exc).__name__}"
        elif process == "starting":
            error = "Screenpipe 正在启动，监听端口尚未就绪"
        elif process == "exited":
            error = "Screenpipe 进程已退出，请检查 screenpipe.log"
        else:
            error = "Screenpipe 进程或数据目录与绑定账号不匹配"
    return {**prefs, "supervisor_running": alive,
            "screenpipe_running": running, "analysis_running": alive and running and state.get("analysis_running", False),
            "error": error}


async def require_screen_access(user_id: str) -> None:
    state = await monitor_status(user_id)
    if not state["recording_enabled"] or not state["screenpipe_running"]:
        raise PermissionError(state["error"] or "屏幕记录已关闭，拒绝读取屏幕数据")


class ScreenRecorder:
    def __init__(self, user_id: str):
        self.user_id = require_screen_owner(user_id)
        self.process = None
        self.log_file = None

    async def start(self) -> None:
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError(f"Screenpipe 退出（{self.process.returncode}），请检查 screenpipe.log")
            return
        host, port = server_address()
        await asyncio.to_thread(require_free_port, host, port)
        exe = Path(config.SCREENPIPE_EXE).resolve()
        if not exe.is_file():
            raise FileNotFoundError(f"Screenpipe 未安装：{exe}")
        bundled = config.PROJECT_ROOT / "src/body/windows_system/eye/screenpipe-0.3.6-x86_64-pc-windows-msvc/bin/screenpipe.exe"
        if exe == bundled.resolve():
            await asyncio.to_thread(verify_models)
        directory = data_directory(self.user_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.log_file = (directory / "screenpipe.log").open("ab")
        try:
            self.process = subprocess.Popen(
                [str(exe), "--disable-audio", "--fps", "1", "--data-dir", str(directory),
                 "--port", str(port), "--ocr-engine", "windows-native", "--disable-telemetry",
                 "--auto-destruct-pid", str(os.getpid())],
                cwd=exe.parent, stdout=self.log_file, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.log_file.close()
            self.log_file = None
            raise

    async def stop(self) -> None:
        if self.process is not None:
            # Only stop the process started by this supervisor, never all screenpipe.exe.
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    await asyncio.to_thread(self.process.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait, timeout=5)
            self.process = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    async def heartbeat(self, *, error: str | None = None, analysis: bool = False, stopped: bool = False) -> None:
        await set_setting(f"screen_monitor:runtime:{self.user_id}", json.dumps({
            "heartbeat": 0 if stopped else time.time(), "screenpipe_pid": self.process.pid if self.process else None,
            "analysis_running": analysis, "error": error,
        }))
