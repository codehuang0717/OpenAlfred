"""One owned GPU worker, serialized transitions and verified readiness."""

import asyncio
import json
import os
import secrets
import subprocess
import time
from pathlib import Path

import httpx
import psutil
from filelock import FileLock, Timeout

from core.config import config
from schemas.voice import VoiceEngineInfo, VoiceRuntime, VoiceSettings
from services.voice_store import VOICE_DATA
from utils.logger import get_logger

logger = get_logger("voice-runtime")
WORKER = Path(__file__).with_name("voice_engine.py").resolve()


def model_path(settings: VoiceSettings) -> Path:
    if settings.engine == "voxcpm":
        path = config.VOXCPM_PROJECT / "pretrained_models" / "VoxCPM2"
    else:
        cache = Path(os.getenv("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
        repo = cache / f"models--Qwen--Qwen3-TTS-12Hz-{settings.qwen_model}-Base"
        ref = repo / "refs" / "main"
        path = repo / "snapshots" / ref.read_text().strip() if ref.is_file() else next(iter(sorted((repo / "snapshots").glob("*"))), repo)
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise ValueError("本地模型文件不完整，请先下载对应模型；启动不会自动下载")
    return path.resolve()


def launch_signature(settings: VoiceSettings) -> tuple:
    return settings.engine, settings.qwen_model if settings.engine == "fasterqwentts" else "VoxCPM2", settings.qwen.max_seq_len if settings.engine == "fasterqwentts" else 0


def engine_info() -> list[VoiceEngineInfo]:
    infos = []
    for engine, name, project in [("voxcpm", "VoxCPM2", config.VOXCPM_PROJECT), ("fasterqwentts", "Faster-Qwen3-TTS", config.QWEN_TTS_PROJECT)]:
        available = (project / ".venv/Scripts/python.exe").is_file()
        infos.append(VoiceEngineInfo(id=engine, name=name, available=available, reason=None if available else "未找到该项目的 Python 环境"))
    return infos


def _terminate(pid: int, created: float) -> None:
    """Terminate only a validated owned process and its descendants."""
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - created) > 0.01 or str(WORKER) not in process.cmdline():
            return
        children = process.children(recursive=True)
        for item in [*children, process]:
            try:
                item.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([*children, process], timeout=5)
        for item in alive:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    except psutil.NoSuchProcess:
        pass


class VoiceManager:
    def __init__(self) -> None:
        self.state = "stopped"
        self.error = None
        self.signature = None
        self.process = None
        self.created = None
        self.token = None
        self.transition = None
        self.guard = asyncio.Lock()
        self.file_lock = None
        self.log = None

    async def initialize(self) -> None:
        if self.file_lock is not None:
            return
        VOICE_DATA.mkdir(parents=True, exist_ok=True)
        lock = FileLock(VOICE_DATA / "runtime.lock")
        try:
            lock.acquire(timeout=0)
        except Timeout as error:
            raise RuntimeError("语音管理器已在其他 API 进程运行，请使用单个业务 API worker") from error
        self.file_lock = lock
        # Recover only the exact process left by an interrupted API lifetime.
        record = VOICE_DATA / "process.json"
        try:
            if record.is_file():
                previous = json.loads(record.read_text())
                await asyncio.to_thread(_terminate, previous["pid"], previous["created"])
                record.unlink(missing_ok=True)
        except BaseException:
            lock.release()
            self.file_lock = None
            raise

    def status(self, settings: VoiceSettings) -> VoiceRuntime:
        if self.process is not None and self.process.poll() is not None and self.state == "running":
            self.state, self.error = "failed", "语音引擎意外退出，请查看语音运行日志后重新启动"
        return VoiceRuntime(
            state=self.state, engine=self.signature[0] if self.signature else None,
            model=self.signature[1] if self.signature else None, error=self.error,
            restart_required=self.signature is not None and self.signature != launch_signature(settings),
        )

    async def _stop(self) -> None:
        if self.process is not None:
            await asyncio.to_thread(_terminate, self.process.pid, self.created)
            await asyncio.to_thread(self.process.wait)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        (VOICE_DATA / "process.json").unlink(missing_ok=True)
        (VOICE_DATA / "launch.json").unlink(missing_ok=True)
        self.token = None

    async def _start(self, settings: VoiceSettings, path: Path) -> None:
        try:
            await self._stop()
            self.signature = launch_signature(settings)
            project = config.VOXCPM_PROJECT if settings.engine == "voxcpm" else config.QWEN_TTS_PROJECT
            self.token = secrets.token_urlsafe(32)
            manifest = {"engine": settings.engine, "model": self.signature[1], "model_path": str(path), "max_seq_len": settings.qwen.max_seq_len, "port": config.VOICE_ENGINE_PORT, "token": self.token}
            launch_file = VOICE_DATA / "launch.json"
            launch_file.write_text(json.dumps(manifest), encoding="utf-8")
            self.log = (VOICE_DATA / "engine.log").open("wb")
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
            self.process = subprocess.Popen(
                [str(project / ".venv/Scripts/python.exe"), str(WORKER), str(launch_file)],
                cwd=project, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.created = psutil.Process(self.process.pid).create_time()
            (VOICE_DATA / "process.json").write_text(json.dumps({"pid": self.process.pid, "created": self.created}))
            deadline = time.monotonic() + config.VOICE_START_TIMEOUT
            async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError("模型加载失败，请检查 data/voice/engine.log 后重新启动")
                    try:
                        response = await client.get(f"http://127.0.0.1:{config.VOICE_ENGINE_PORT}/health", headers={"X-Engine-Token": self.token})
                        if response.is_success and response.json().get("ready"):
                            self.state, self.error = "running", None
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.5)
            raise RuntimeError("模型启动超时，请检查显存和语音运行日志")
        except asyncio.CancelledError:
            await self._stop()
            self.state = "stopped"
            raise
        except Exception as error:
            logger.exception("Voice engine start failed")
            await self._stop()
            self.state, self.error = "failed", str(error)

    async def control(self, action: str, settings: VoiceSettings) -> VoiceRuntime:
        async with self.guard:
            await self.initialize()
            if action == "start" and self.status(settings).state == "running" and self.signature == launch_signature(settings):
                return self.status(settings)
            # Validate a requested replacement before stopping a healthy engine.
            path = await asyncio.to_thread(model_path, settings) if action != "stop" else None
            if action != "stop" and not settings.tts_enabled:
                raise ValueError("请先开启语音合成")
            if self.transition is not None and not self.transition.done():
                self.transition.cancel()
                await asyncio.gather(self.transition, return_exceptions=True)
            self.error = None
            if action == "stop":
                self.state = "stopping"
                await self._stop()
                self.state, self.signature = "stopped", None
            else:
                project = config.VOXCPM_PROJECT if settings.engine == "voxcpm" else config.QWEN_TTS_PROJECT
                if not (project / ".venv/Scripts/python.exe").is_file():
                    raise ValueError("未找到引擎 Python 环境，请先使用 uv sync 安装")
                self.state = "starting"
                self.transition = asyncio.create_task(self._start(settings.model_copy(deep=True), path), name="voice-start")
            return self.status(settings)

    async def close(self) -> None:
        if self.file_lock is None:
            return
        async with self.guard:
            if self.transition is not None and not self.transition.done():
                self.transition.cancel()
                await asyncio.gather(self.transition, return_exceptions=True)
            await self._stop()
            self.state = "stopped"
            self.file_lock.release()
            self.file_lock = None


voice_manager = VoiceManager()
