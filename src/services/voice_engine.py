"""Isolated inference worker; launched with the selected project's Python.

Never import GPU frameworks into the business API process. Only listen on
loopback and require a per-launch token for both health and synthesis.
"""

import sys
from pathlib import Path

# Executing a file inside services/ would shadow stdlib email with email.py.
# Remove that script directory before importing the HTTP server dependencies.
sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json
import logging
import queue
import secrets
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from schemas.voice import VoiceModel, VoxOptions, QwenOptions

logger = logging.getLogger("voice-engine")


class EngineSpeech(VoiceModel):
    text: str = Field(min_length=1, max_length=4000)
    reference_path: str | None = None
    reference_text: str = ""
    vox: VoxOptions = Field(default_factory=VoxOptions)
    qwen: QwenOptions = Field(default_factory=QwenOptions)


def create_app(manifest: dict) -> FastAPI:
    model = None
    gate = asyncio.Lock()

    def load():
        if manifest["engine"] == "voxcpm":
            from voxcpm import VoxCPM
            return VoxCPM.from_pretrained(manifest["model_path"], device="cuda", load_denoiser=False, optimize=False, local_files_only=True)
        import torch
        from faster_qwen3_tts import FasterQwen3TTS
        return FasterQwen3TTS.from_pretrained(manifest["model_path"], device="cuda", dtype=torch.bfloat16, max_seq_len=manifest["max_seq_len"], attn_implementation="sdpa")

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal model
        model = await asyncio.to_thread(load)
        logger.info("Voice model loaded: %s", manifest["engine"])
        yield
        model = None

    app = FastAPI(lifespan=lifespan)

    def authorized(x_engine_token: str = Header("")) -> None:
        if not secrets.compare_digest(x_engine_token, manifest["token"]):
            raise HTTPException(401, "Invalid engine token")

    @app.get("/health", dependencies=[Depends(authorized)])
    def health():
        return {"ready": model is not None, "engine": manifest["engine"], "model": manifest["model"]}

    def generate(request: EngineSpeech):
        if manifest["engine"] == "voxcpm":
            kwargs = request.vox.model_dump()
            kwargs.update(text=request.text.strip(), retry_badcase=False)
            if request.reference_path:
                kwargs.update(reference_wav_path=request.reference_path, prompt_wav_path=request.reference_path, prompt_text=request.reference_text)
            for chunk in model.generate_streaming(**kwargs):
                yield np.asarray(chunk).reshape(-1), model.tts_model.sample_rate
        else:
            if not request.reference_path:
                raise ValueError("Qwen Base requires a reference recording")
            kwargs = request.qwen.model_dump(exclude={"max_seq_len"})
            for chunk, rate, _timing in model.generate_voice_clone_streaming(
                text=request.text.strip(), ref_audio=request.reference_path,
                ref_text=request.reference_text, non_streaming_mode=False, **kwargs,
            ):
                yield np.asarray(chunk).reshape(-1), rate

    async def pcm_stream(request: EngineSpeech):
        pending: queue.Queue = queue.Queue(maxsize=4)
        cancelled = threading.Event()
        done = object()
        loop = asyncio.get_running_loop()

        def put(value):
            while not cancelled.is_set():
                try:
                    pending.put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def producer():
            resampler = None
            try:
                for audio, rate in generate(request):
                    if cancelled.is_set():
                        break
                    if rate != 24000:
                        # VoxCPM's runtime includes soxr via librosa. Stateful
                        # resampling keeps filter history across audio chunks.
                        import soxr
                        if resampler is None:
                            resampler = soxr.ResampleStream(rate, 24000, 1, dtype="float32")
                        audio = resampler.resample_chunk(audio.astype(np.float32))
                    put(np.clip(audio * 32767, -32768, 32767).astype("<i2").tobytes())
                if resampler is not None and not cancelled.is_set():
                    tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                    put(np.clip(tail * 32767, -32768, 32767).astype("<i2").tobytes())
            except Exception as error:
                logger.exception("Speech generation failed")
                put(error)
            finally:
                try:
                    put(done)
                finally:
                    # The producer owns the inference lock. A disconnected
                    # client cannot release it while CUDA is still running.
                    loop.call_soon_threadsafe(gate.release)

        thread = threading.Thread(target=producer, daemon=True, name="voice-inference")
        thread.start()
        try:
            while True:
                try:
                    item = await asyncio.to_thread(pending.get, True, 0.5)
                except queue.Empty:
                    continue
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                if item:
                    yield item
        finally:
            cancelled.set()

    @app.post("/speech", dependencies=[Depends(authorized)])
    async def speech(request: EngineSpeech):
        try:
            await asyncio.wait_for(gate.acquire(), timeout=10)
        except TimeoutError as error:
            raise HTTPException(429, "语音引擎繁忙，请稍后重试") from error
        stream = pcm_stream(request)
        try:
            first = await anext(stream)
        except BaseException:
            await stream.aclose()
            raise

        async def response_stream():
            try:
                yield first
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()

        return StreamingResponse(response_stream(), media_type="audio/pcm", headers={"X-Audio-Sample-Rate": "24000"})

    return app


if __name__ == "__main__":
    import uvicorn
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    uvicorn.run(create_app(manifest), host="127.0.0.1", port=manifest["port"], log_level="info")
