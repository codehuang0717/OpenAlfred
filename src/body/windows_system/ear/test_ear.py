import asyncio
import logging
import sys
import os

# Add the src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from body.windows_system.ear.ear_service import EarService

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*50)
    print("OpenAlfred Wake Word Test")
    print("Keyword: 'Hey Jarvis' (default)")
    print("Duration: 30 seconds")
    print("="*50 + "\n")
    
    def on_wake(name, score):
        print(f"\n[!] WAKE WORD DETECTED: {name} (Confidence: {score:.2f})")
        print("Listening for more...")

    ear = EarService()
    try:
        ear.start(on_wake)
        print("Service started. Please say 'Hey Jarvis' to your microphone...")
        await asyncio.sleep(30)
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        ear.stop()
        print("\nTest finished.")

if __name__ == "__main__":
    asyncio.run(main())
