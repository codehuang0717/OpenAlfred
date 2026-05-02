
import os
import sys
import asyncio
import wave

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

async def main():
    # Use the same TTS logic as the worker
    from livekit_worker import get_tts_stream
    
    # Absolute path
    base_dir = r"E:\PythonProjects\OpenAlfred\agent"
    output_dir = os.path.join(base_dir, "assets", "sounds")
    output_path = os.path.join(output_dir, "wake_up.wav")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating voice for: 我在，请讲。")
    frames = []
    async for chunk in get_tts_stream("我在，请讲。", target_sample_rate=24000):
        frames.append(chunk)
    
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"".join(frames))
    
    print(f"Voice saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
