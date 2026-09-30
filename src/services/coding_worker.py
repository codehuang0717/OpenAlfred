"""One bounded worker pool, durable checkpoints and cancellation-aware jobs."""

import asyncio
import time
import uuid
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import APIConnectionError, APITimeoutError

from core.config import config
from core.event_bus import EventType, event_bus
from db.coding_jobs import claim_job, expire_jobs, get_job, keep_lease, progress, stop_job
from db.user_apps import finish_code_app_job
from logic.coding_agent import build_coding_agent, initial_coding_state, source_from_files, source_hash
from services.code_apps import create_codegen_model
from services.user_time import validate_timezone
from utils.logger import get_logger

logger = get_logger("coding-worker")


def checkpoint_config(job: dict) -> dict:
    context = job["context"]
    return {"configurable": {
        "thread_id": f"coding:{job['user_id']}:{job['app_id']}:{job['id']}:{context.get('checkpoint_generation', 0)}",
        "owner": job["user_id"], "timezone": validate_timezone(context["timezone"]),
    }, "callbacks": [], "recursion_limit": config.CODING_MAX_MODEL_CALLS * 4 + 10}


async def notify(job: dict) -> None:
    await event_bus.publish(EventType.USER_APP_UPDATED, {
        "id": job["app_id"], "job_id": job["id"], "user_id": job["user_id"],
        "source_thread_id": job["context"].get("source_thread_id"),
    })


class CodingWorker:
    def __init__(self, checkpoint_path: Path | None = None):
        self.path = checkpoint_path or config.CODING_CHECKPOINT_PATH
        self.tasks: list[asyncio.Task] = []
        self.running: dict[str, asyncio.Task] = {}
        self.conn = None
        self.saver = None
        self.runner_id = str(uuid.uuid4())

    async def start(self) -> None:
        if self.tasks:
            raise RuntimeError("Coding worker already started")
        if config.CODING_CONCURRENCY < 1 or config.CODING_TIMEOUT_SECONDS <= 0:
            raise ValueError("Coding concurrency and deadline must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(str(self.path))
        try:
            await self.conn.execute("PRAGMA busy_timeout=5000")
            await self.conn.execute("PRAGMA journal_mode=WAL")
            self.saver = AsyncSqliteSaver(self.conn)
            await self.saver.setup()
            await self._purge_orphans()
        except BaseException:
            await self.conn.close()
            self.conn = None
            raise
        self.tasks = [asyncio.create_task(self._loop(), name=f"coding-worker-{i}") for i in range(config.CODING_CONCURRENCY)]

    async def _purge_orphans(self) -> None:
        """Recover a crash between app deletion and checkpoint cleanup."""
        after = ""
        while True:
            async with self.conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'coding:%' AND thread_id > ? ORDER BY thread_id LIMIT 100",
                (after,),
            ) as cursor:
                identities = [row[0] for row in await cursor.fetchall()]
            if not identities:
                return
            for identity in identities:
                parts = identity.rsplit(":", 3)
                if len(parts) != 4 or not parts[0].startswith("coding:"):
                    raise ValueError("Invalid coding checkpoint identity")
                owner, app_id, job_id = parts[0][len("coding:"):], parts[1], parts[2]
                job = await get_job(owner, job_id)
                if job is None or job["app_id"] != app_id:
                    await self.saver.adelete_thread(identity)
            after = identities[-1]

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def purge_jobs(self, jobs: list[dict]) -> None:
        """After DB deletion fences the jobs, stop local runs and erase private checkpoints."""
        tasks = [self.running[job["id"]] for job in jobs if job["id"] in self.running]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for job in jobs:
            if "timezone" not in job["context"]:
                continue  # Legacy one-shot jobs never had coding checkpoints.
            for generation in range(job["context"].get("checkpoint_generation", 0) + 1):
                identity = {**job, "context": {**job["context"], "checkpoint_generation": generation}}
                await self.saver.adelete_thread(checkpoint_config(identity)["configurable"]["thread_id"])

    async def _loop(self) -> None:
        while True:
            try:
                await expire_jobs()
                job = await claim_job(self.runner_id, config.CODING_CONCURRENCY)
                if job is None:
                    await asyncio.sleep(1)
                    continue
                task = asyncio.create_task(self.run_job(job))
                self.running[job["id"]] = task
                try:
                    await task
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    # User cancellation stops the job, not its queue worker.
                finally:
                    self.running.pop(job["id"], None)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Coding queue iteration failed")
                await asyncio.sleep(2)

    async def _heartbeat(self, job: dict, task: asyncio.Task) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                owned = await keep_lease(job)
            except Exception:
                logger.exception("Cannot renew coding lease")
                task.cancel()
                return
            if not owned:
                task.cancel()
                return

    async def run_job(self, job: dict, *, model=None) -> None:
        start = time.monotonic()
        harness = None
        owns_model = model is None
        heartbeat = asyncio.create_task(self._heartbeat(job, asyncio.current_task()))
        try:
            if model is None:
                model = create_codegen_model(job["model"])
            actual_name = getattr(model, "model_name", None) or getattr(model, "model", None)
            if actual_name != job["context"]["model_name"]:
                raise ValueError("模型配置在排队期间变化；请明确重试，未自动切换模型")

            async def on_progress(stage, message, metrics):
                await progress(job, stage, message, metrics=metrics)
                await notify(job)

            graph, harness = build_coding_agent(model, job["model"], self.saver, on_progress)
            harness.restore_metrics(job["metrics"])
            run_config = checkpoint_config(job)
            if job["context"].get("resume"):
                checkpoint = await graph.aget_state(run_config)
                if not checkpoint.values:
                    raise ValueError("没有可恢复的检查点，请选择重新生成")
                if not checkpoint.next and not checkpoint.values.get("summary"):
                    raise ValueError("检查点已经结束且没有完成报告，请选择重新生成")
                harness.restore_metrics({**checkpoint.values,
                    "token_upper_bound": checkpoint.values.get("token_estimate", 0)})
                initial = None
            else:
                initial = initial_coding_state(job)
            async with asyncio.timeout(config.CODING_TIMEOUT_SECONDS):
                state = await graph.ainvoke(initial, config=run_config)
            source = source_from_files(state.get("files", {}))
            report = state.get("validation", {})
            if not state.get("summary") or report.get("source_hash") != source_hash(state["files"]):
                raise ValueError("编码 Agent 没有交付经验证且未修改的代码")
            metrics = {**harness.metrics(), "elapsed_seconds": round(time.monotonic() - start, 2)}
            await finish_code_app_job(job["user_id"], job["id"], epoch=job["epoch"],
                                      source=source.model_dump(), validation=report,
                                      report=state["summary"], metrics=metrics)
        except asyncio.CancelledError:
            await stop_job(job["user_id"], job["id"], "interrupted", "编码执行中断；可恢复检查点，未发布代码",
                           expected_epoch=job["epoch"])
            await notify(job)
            raise
        except Exception as exc:
            logger.exception("Coding job %s failed (%s)", job["id"], job["model"])
            if isinstance(exc, (APIConnectionError, APITimeoutError)):
                reason = "无法连接所选模型服务，请检查网络或代理"
            elif isinstance(exc, (ValueError, RuntimeError, TimeoutError)):
                reason = str(exc)[:400] or "编码总等待超过配置时限"
            else:
                reason = "内部编码服务错误，请检查 coding-worker 日志"
            error = f"{type(exc).__name__}: {reason}（{job['model']}）；未切换模型，也未发布代码。"
            current = await get_job(job["user_id"], job["id"])
            if current and current["status"] == "generating" and current["epoch"] == job["epoch"]:
                await finish_code_app_job(job["user_id"], job["id"], epoch=job["epoch"], error=error,
                                          metrics=harness.metrics() if harness else {})
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if owns_model and model is not None:
                if getattr(model, "root_async_client", None) is not None:
                    await model.root_async_client.close()
                if getattr(model, "root_client", None) is not None:
                    model.root_client.close()
            # Deletion must also erase checkpointed private read results.
            if self.saver and await get_job(job["user_id"], job["id"]) is None:
                await self.saver.adelete_thread(checkpoint_config(job)["configurable"]["thread_id"])
        await notify(job)
