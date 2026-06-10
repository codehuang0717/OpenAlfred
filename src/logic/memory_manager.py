"""
L1 Local Memory Manager — manages per-user markdown files under memory/{user_id}/.

Files: profile.md, preferences.md, relationship.md, learned_patterns.md
Only profile.md and preferences.md are injected into the system prompt every
turn. Relationship and behavioral pattern memories stay available through
tools and are loaded on demand.
"""

import re
import shutil
import logging
from pathlib import Path
from typing import Optional

from core.config import config

logger = logging.getLogger("memory-manager")

ALL_L1_FILES = ["profile.md", "preferences.md", "relationship.md", "learned_patterns.md"]
DEFAULT_INJECTED_FILES = ["profile.md", "preferences.md"]
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class MemoryManager:
    """Manages L1 local markdown memory files."""

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or config.MEMORY_DIR
        self._templates_dir = self.memory_dir / "_templates"

    # ── Internal helpers ──────────────────────────────────────────────

    def _user_dir(self, user_id: str) -> Path:
        return self.memory_dir / user_id

    def _ensure_user_dir(self, user_id: str):
        """Create user memory directory from templates if it doesn't exist."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
            for fname in ALL_L1_FILES:
                src = self._templates_dir / fname
                dst = user_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            logger.info(f"Initialized L1 memory directory for user '{user_id}'.")
        return user_dir

    def _read_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _strip_comments(self, text: str) -> str:
        return COMMENT_RE.sub("", text).strip()

    # ── Public API ────────────────────────────────────────────────────

    def load_memories(self, user_id: str, filenames: Optional[list[str]] = None) -> str:
        """Load and format selected L1 files. Returns '' if empty."""
        self._ensure_user_dir(user_id)
        user_dir = self._user_dir(user_id)
        parts: list[str] = []
        for fname in filenames or ALL_L1_FILES:
            if fname not in ALL_L1_FILES:
                continue
            content = self._read_file(user_dir / fname)
            content = self._strip_comments(content).strip()
            if content:
                lines = content.split("\n")
                if lines and lines[0].startswith("# "):
                    title = lines[0][2:].strip()
                    body = "\n".join(lines[1:]).strip()
                else:
                    title = fname.replace(".md", "").replace("_", " ").title()
                    body = content
                if body:
                    parts.append(f"## {title}\n{body}")
        return "\n\n".join(parts) if parts else ""

    def load_all_memories(self, user_id: str) -> str:
        """Load and format all L1 files. Returns '' if empty."""
        return self.load_memories(user_id, ALL_L1_FILES)

    def append_to_memory_file(self, user_id: str, filename: str, text: str):
        """Append a timestamped line to a memory file."""
        self._ensure_user_dir(user_id)
        path = self._user_dir(user_id) / filename
        existing = self._read_file(path)
        new_content = existing.rstrip() + "\n" + text.strip() + "\n"
        path.write_text(new_content, encoding="utf-8")

    def build_injection_text(self, user_id: str) -> str:
        """Format only core profile/preferences memories for prompt injection."""
        memories = self.load_memories(user_id, DEFAULT_INJECTED_FILES)
        return f"[用户长期记忆]\n---\n{memories}\n---" if memories else ""


# Singleton
memory_manager = MemoryManager()
