"""Verified model cache required by the bundled Screenpipe 0.3.6 binary."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODELS = {
    "wespeaker_en_voxceleb_CAM++.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
        29_292_684,
        "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef",
    ),
    "segmentation-3.0.onnx": (
        "https://huggingface.co/altunenes/modelsfolder/resolve/main/segmentation-3.0.onnx",
        5_983_836,
        "b78fc48113bb46fd247ae6a9aea737079550c647638db961df7e0e1e9f4ba62e",
    ),
}


def model_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA 未设置，无法定位 Screenpipe 模型目录")
    return Path(local_app_data) / "screenpipe" / "models"


def _valid_model(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def verify_models(directory: Path | None = None) -> None:
    """Fail before capture if the legacy binary would load missing or corrupt models."""
    directory = directory or model_directory()
    invalid = [name for name, (_, size, digest) in MODELS.items()
               if not _valid_model(directory / name, size, digest)]
    if invalid:
        raise RuntimeError(
            "Screenpipe 模型缺失或校验失败：" + ", ".join(invalid)
            + "；先运行 ./src/body/windows_system/eye/setup_eye.ps1 -InstallModels"
        )


def install_models(directory: Path | None = None) -> list[str]:
    """Explicit setup only: download pinned files, verify, then replace bad cache files."""
    directory = directory or model_directory()
    directory.mkdir(parents=True, exist_ok=True)
    installed = []
    for name, (url, size, digest) in MODELS.items():
        target = directory / name
        if _valid_model(target, size, digest):
            continue
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".download", dir=directory)
        backup = None
        try:
            with os.fdopen(fd, "wb") as output, urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    if output.tell() > size:
                        raise RuntimeError(f"{name} 下载大小超过预期，拒绝安装")
            if not _valid_model(Path(temporary), size, digest):
                raise RuntimeError(f"{name} 下载校验失败，原缓存未修改")
            if target.exists():
                backup = directory / f"{name}.invalid-{os.urandom(4).hex()}"
                target.replace(backup)
            try:
                Path(temporary).replace(target)
            except OSError:
                if backup is not None:
                    backup.replace(target)
                raise
            installed.append(name)
        finally:
            Path(temporary).unlink(missing_ok=True)
    verify_models(directory)
    return installed
