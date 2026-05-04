"""
Two-Level Memory Manager — L1 (local .md files, hot path) + L2 (Mem0 semantic, warm path).

L1 files live under memory/{user_id}/ and are injected into every system prompt.
L2 (Mem0) is searched on-demand via the search_memory tool.
Periodic consolidation promotes valuable L2 memories to L1.
"""

import re
import json
import shutil
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from core.config import config

logger = logging.getLogger("memory-manager")

# ── File mapping ──────────────────────────────────────────────────────────
CATEGORY_FILES = {
    "profile": "profile.md",
    "preferences": "preferences.md",
    "relationship": "relationship.md",
    "patterns": "learned_patterns.md",
}

ALL_L1_FILES = ["profile.md", "preferences.md", "relationship.md", "learned_patterns.md"]

# Regex to strip HTML-like comments from .md files when injecting
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class MemoryManager:
    """Manages L1 local markdown memory files with automatic compression and dedup."""

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        max_tokens_per_file: int = config.L1_MAX_TOKENS_PER_FILE,
        max_total_tokens: int = config.L1_MAX_TOTAL_TOKENS,
    ):
        self.memory_dir = memory_dir or config.MEMORY_DIR
        self.max_tokens_per_file = max_tokens_per_file
        self.max_total_tokens = max_total_tokens
        self._templates_dir = self.memory_dir / "_templates"

    # ── Helpers ────────────────────────────────────────────────────────

    def _user_dir(self, user_id: str) -> Path:
        return self.memory_dir / user_id

    def _ensure_user_dir(self, user_id: str):
        """Create user memory directory from templates if it doesn't exist."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
            # Copy templates
            for fname in ["MEMORY.md"] + ALL_L1_FILES:
                src = self._templates_dir / fname
                dst = user_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            logger.info(f"Initialized L1 memory directory for user '{user_id}' from templates.")
        return user_dir

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token for mixed CN/EN text."""
        return max(1, len(text) // 4)

    def _read_file(self, path: Path) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _strip_comments(self, text: str) -> str:
        """Remove HTML comments (template instructions) from content for injection."""
        return COMMENT_RE.sub("", text).strip()

    # ── Core API ───────────────────────────────────────────────────────

    def load_all_memories(self, user_id: str) -> str:
        """Load and format all L1 files for system prompt injection. Returns '' if none."""
        self._ensure_user_dir(user_id)
        user_dir = self._user_dir(user_id)
        parts: list[str] = []
        for fname in ALL_L1_FILES:
            content = self._read_file(user_dir / fname)
            content = self._strip_comments(content).strip()
            if content:
                # Remove the leading "# Title" line for compact injection
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

    def get_memory_file(self, user_id: str, filename: str) -> str:
        """Read a single L1 file. Returns '' if not found."""
        self._ensure_user_dir(user_id)
        return self._read_file(self._user_dir(user_id) / filename)

    def update_memory_file(self, user_id: str, filename: str, content: str):
        """Overwrite a single L1 file."""
        self._ensure_user_dir(user_id)
        path = self._user_dir(user_id) / filename
        path.write_text(content, encoding="utf-8")
        self._check_and_compress(user_id, filename)
        self._update_index(user_id)

    def append_to_memory_file(self, user_id: str, filename: str, text: str):
        """Append a line/entry to a memory file. Triggers compression if over limit."""
        self._ensure_user_dir(user_id)
        path = self._user_dir(user_id) / filename
        existing = self._read_file(path)
        new_content = existing.rstrip() + "\n" + text.strip() + "\n"
        path.write_text(new_content, encoding="utf-8")
        self._check_and_compress(user_id, filename)
        self._update_index(user_id)

    # ── Compression ────────────────────────────────────────────────────

    def _check_and_compress(self, user_id: str, filename: str):
        """Check if a file exceeds the token limit and compress if needed."""
        path = self._user_dir(user_id) / filename
        content = self._read_file(path)
        if self._estimate_tokens(content) > self.max_tokens_per_file:
            logger.warning(
                f"L1 file {filename} for user '{user_id}' exceeds "
                f"{self.max_tokens_per_file} token limit. Compression needed (async)."
            )
            # Compression requires LLM — we mark it but don't block here.
            # The consolidation cycle or next extract_and_merge call will handle it.

    async def compress_file(self, user_id: str, filename: str, llm) -> str:
        """Use LLM to compress an over-limit memory file. Returns compressed content."""
        path = self._user_dir(user_id) / filename
        content = self._read_file(path)
        if self._estimate_tokens(content) <= self.max_tokens_per_file:
            return content

        prompt = (
            f"以下是用户的长期记忆文件 '{filename}'，内容过长需要压缩。"
            f"请保留所有关键事实和偏好，删除冗余和重复信息，"
            f"将输出控制在 {self.max_tokens_per_file} tokens 以内。"
            f"保持 Markdown 格式，只输出压缩后的内容。\n\n"
            f"原始内容：\n{content}"
        )
        from langchain_core.messages import HumanMessage
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        compressed = result.content.strip()
        path.write_text(compressed, encoding="utf-8")
        logger.info(f"Compressed {filename} for user '{user_id}': "
                     f"{len(content)} → {len(compressed)} chars")
        self._update_index(user_id)
        return compressed

    # ── Extraction & Merging ───────────────────────────────────────────

    async def extract_and_merge(self, user_id: str, facts: list[dict], llm) -> dict[str, int]:
        """Receive extracted facts, deduplicate, and merge into corresponding L1 files.

        Args:
            user_id: User identifier.
            facts: List of {"category": str, "fact": str, "importance": str}.
            llm: LLM instance for dedup decisions.

        Returns:
            Dict mapping filename → number of new facts added.
        """
        self._ensure_user_dir(user_id)
        added_counts: dict[str, int] = {}

        # Group facts by filename
        by_file: dict[str, list[str]] = {}
        for f in facts:
            cat = f.get("category", "profile")
            fname = CATEGORY_FILES.get(cat, "profile.md")
            if fname not in by_file:
                by_file[fname] = []
            by_file[fname].append(f.get("fact", ""))

        for fname, new_facts in by_file.items():
            path = self._user_dir(user_id) / fname
            existing = self._read_file(path)

            # Dedup: ask LLM which facts are genuinely new
            deduped = await self._deduplicate(existing, new_facts, llm)
            if not deduped:
                continue

            # Append new facts
            timestamp = datetime.now().strftime("%Y-%m-%d")
            new_lines = "\n".join(f"- [{timestamp}] {fact}" for fact in deduped)
            updated = existing.rstrip() + "\n" + new_lines + "\n"
            path.write_text(updated, encoding="utf-8")
            added_counts[fname] = len(deduped)

            # Compress if needed
            if self._estimate_tokens(updated) > self.max_tokens_per_file:
                await self.compress_file(user_id, fname, llm)

        if added_counts:
            logger.info(f"L1 memory updated for user '{user_id}': {added_counts}")
            self._update_index(user_id)

        return added_counts

    async def _deduplicate(
        self, existing_content: str, new_facts: list[str], llm
    ) -> list[str]:
        """Filter out facts already present in existing content.

        Uses a fast string-match pre-filter first to avoid unnecessary LLM calls.
        Only invokes the LLM when there's content overlap that needs semantic judgment.
        Always returns a list of strings.
        """
        existing_stripped = self._strip_comments(existing_content).strip()
        if not existing_stripped:
            return new_facts  # Empty file, all facts are new

        # ── Fast pre-filter: exact/substring match ──
        existing_lower = existing_stripped.lower()
        definitely_new = []
        needs_llm_check = []
        for fact in new_facts:
            fact_lower = fact.lower().strip()
            # Direct substring check (fast, no API call)
            if fact_lower in existing_lower:
                continue  # definite duplicate
            # Check for substantial word overlap (simple heuristic)
            fact_words = set(fact_lower.split())
            if fact_words:
                # Count how many words appear in existing content
                overlap = sum(1 for w in fact_words if w in existing_lower)
                overlap_ratio = overlap / len(fact_words)
                if overlap_ratio > 0.7:
                    needs_llm_check.append(fact)  # ambiguous, needs LLM
                else:
                    definitely_new.append(fact)  # likely new
            else:
                definitely_new.append(fact)

        # All facts were filtered by fast path
        if not needs_llm_check:
            return definitely_new

        # ── LLM semantic dedup for ambiguous cases only ──
        facts_json = json.dumps(needs_llm_check, ensure_ascii=False)
        prompt = (
            f"现有记忆内容：\n{existing_stripped}\n\n"
            f"待判断的新事实（JSON数组）：\n{facts_json}\n\n"
            f"请判断每个新事实是否与现有内容重复或高度相似。"
            f"返回一个 JSON 字符串数组，只包含那些 NOT 重复的新事实（保留原始字符串）。"
            f"如果所有事实都已存在，返回空数组 []。\n"
            f"只输出 JSON 字符串数组。"
        )
        from langchain_core.messages import HumanMessage
        try:
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content
            if not raw:
                return definitely_new + needs_llm_check
            if not isinstance(raw, str):
                raw = str(raw)
            text = raw.strip()
            if "```" in text:
                text = re.sub(r"```\w*", "", text).replace("```", "").strip()
            deduped = json.loads(text)
            if isinstance(deduped, list):
                normalized = []
                for item in deduped:
                    if isinstance(item, str):
                        normalized.append(item)
                    elif isinstance(item, dict):
                        normalized.append(item.get("fact", str(item)))
                    else:
                        normalized.append(str(item))
                return definitely_new + normalized
        except json.JSONDecodeError:
            logger.debug(f"Dedup LLM returned non-JSON, keeping all ambiguous facts. "
                         f"Raw: {text[:200]}")
        except Exception as e:
            logger.warning(f"Dedup LLM call failed, keeping all ambiguous facts: {e}")
        return definitely_new + needs_llm_check

    # ── Injection ──────────────────────────────────────────────────────

    def build_injection_text(self, user_id: str) -> str:
        """Generate formatted text for injection into the system prompt."""
        memories = self.load_all_memories(user_id)
        if not memories:
            return ""
        return f"[用户长期记忆]\n---\n{memories}\n---"

    # ── Index Maintenance ──────────────────────────────────────────────

    def _update_index(self, user_id: str):
        """Regenerate MEMORY.md index with file summaries."""
        user_dir = self._user_dir(user_id)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# 用户长期记忆索引",
            "",
            "> 此文件为总索引，列出所有子文件和摘要。硬上限 200 行。",
            "> 由系统自动维护，请勿手动编辑。",
            "",
            "## 文件清单",
            "",
            "| 文件 | 用途 | 最后更新 |",
            "|------|------|----------|",
        ]
        descriptions = {
            "profile.md": "基本信息、身份、重要日期",
            "preferences.md": "偏好、喜欢/讨厌的事物",
            "relationship.md": "关系状态、互动历史",
            "learned_patterns.md": "行为模式、习惯",
        }
        for fname in ALL_L1_FILES:
            desc = descriptions.get(fname, "")
            path = user_dir / fname
            mtime = now if path.exists() else "-"
            lines.append(f"| {fname} | {desc} | {mtime} |")

        lines.append("")
        lines.append("## 摘要")
        lines.append("")
        for fname in ALL_L1_FILES:
            content = self._read_file(user_dir / fname)
            content = self._strip_comments(content).strip()
            if content:
                # Take first meaningful line as summary
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("<!--"):
                        lines.append(f"- **{fname}**: {line[:120]}")
                        break

        index_path = user_dir / "MEMORY.md"
        index_path.write_text("\n".join(lines), encoding="utf-8")

    # ── Consolidation (L2 → L1) ────────────────────────────────────────

    async def consolidate_l2_to_l1(
        self, user_id: str, mem0_results: list[str], llm
    ) -> int:
        """Evaluate L2 (Mem0) memories and promote valuable ones to L1.

        Args:
            user_id: User identifier.
            mem0_results: List of memory strings from Mem0 search.
            llm: LLM instance for evaluation.

        Returns:
            Number of facts promoted to L1.
        """
        if not mem0_results:
            return 0

        self._ensure_user_dir(user_id)
        # Load existing L1 for context
        existing_l1 = self.load_all_memories(user_id)

        l2_text = "\n".join(f"- {m}" for m in mem0_results)
        prompt = (
            f"现有 L1 本地记忆（长期保留）：\n{existing_l1 or '(空)'}\n\n"
            f"L2 Mem0 记忆（待评估）：\n{l2_text}\n\n"
            f"从 L2 中挑选值得提升到 L1 的长期事实（用户画像、偏好、习惯、关系）。"
            f"只挑选重要的、可长期保留的信息。忽略临时性、会话级别的信息。\n"
            f"输出 JSON 数组："
            f'[{{"category": "profile|preferences|relationship|patterns", '
            f'"fact": "...", "importance": "high|medium"}}]\n'
            f"只输出 JSON 数组。"
        )
        from langchain_core.messages import HumanMessage
        try:
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            if "```" in text:
                text = re.sub(r"```\w*", "", text).replace("```", "").strip()
            facts = json.loads(text)
            if isinstance(facts, list) and facts:
                return sum(
                    (await self.extract_and_merge(user_id, facts, llm)).values()
                )
        except Exception as e:
            logger.error(f"L2→L1 consolidation failed: {e}")
        return 0


# Singleton
memory_manager = MemoryManager()
