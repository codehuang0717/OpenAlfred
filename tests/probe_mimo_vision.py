"""Live probe: mimo-v2.6 vision describe_image. Does not print secrets.

Run: uv run python tests/probe_mimo_vision.py
"""

from __future__ import annotations

import sys
import tempfile
import zlib
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.image_describer import describe_image  # noqa: E402
from core.config import config  # noqa: E402


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _tiny_red_png() -> bytes:
    """8x8 solid red PNG."""
    w = h = 8
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def main() -> int:
    print(f"vision model: {config.MIMO_VISION_MODEL}")
    print(f"chat model:   {config.MIMO_CHAT_MODEL}")
    if config.MIMO_VISION_MODEL == config.MIMO_CHAT_MODEL:
        print("FAIL: vision model must not be the Pro chat model")
        return 1
    if "pro" in config.MIMO_VISION_MODEL.lower():
        print("FAIL: vision model must not contain 'pro'")
        return 1
    if not config.MIMO_API_KEY:
        print("FAIL: MIMO_API_KEY is empty")
        return 1

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_tiny_red_png())
        path = f.name

    try:
        desc = describe_image(path, force=True)
    finally:
        Path(path).unlink(missing_ok=True)

    if not desc:
        print("FAIL: empty description")
        return 1
    print(f"OK ({len(desc)} chars): {desc[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
