"""Paired synthetic regression checks; no timing or provider-cache assertions."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from context_benchmark import baseline, ContextManager, connection, replay, scenarios


class TestContextBenchmark(unittest.IsolatedAsyncioTestCase):
    async def compare(self, name, requests):
        rows = []
        for cls, tuned in [(baseline.ContextManager, False), (ContextManager, True)]:
            with tempfile.TemporaryDirectory() as tmp, patch.object(connection, "DATABASE_PATH", str(Path(tmp) / "bench.db")):
                await connection.init_db()
                rows.append(await replay(cls, name, requests, tuned))
        return rows

    async def test_short_chat_avoids_summary_churn_and_keeps_larger_prefix(self):
        old, new = await self.compare("short_chat", scenarios()["short_chat"])
        self.assertGreater(old["summary_calls"], 0)
        self.assertEqual(new["summary_calls"], 0)
        self.assertGreater(new["prefix_overlap_pct_proxy"], old["prefix_overlap_pct_proxy"])

    async def test_medium_results_trade_tokens_for_visible_evidence(self):
        old, new = await self.compare("medium_results", scenarios()["medium_results"][:4])
        self.assertEqual(old["latest_tool_middle_visible"], "0/3")
        self.assertEqual(new["latest_tool_middle_visible"], "3/3")
        self.assertGreater(new["main_input_tokens_est"], old["main_input_tokens_est"])
