#This file is used to enroll voice with alibaba cloud dashscope
# But I'm using the local TTS model right now

# import os
# import time
# import dashscope
# from dotenv import load_dotenv
# from dashscope.audio.tts_v2 import VoiceEnrollmentService

# # Load environment variables explicitly
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# # Configuration
# dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
# TARGET_MODEL = "cosyvoice-v3.5-flash"
# VOICE_PREFIX = "hhx"
# AUDIO_URL = "https://pub-027dd4d8fcaf434db1dec64d35aab4ca.r2.dev/cosyvoice_shot_hhx.m4a"  # Replace with user's URL


# def enroll_voice():
#     service = VoiceEnrollmentService()
#     try:
#         voice_id = service.create_voice(
#             target_model=TARGET_MODEL, prefix=VOICE_PREFIX, url=AUDIO_URL
#         )
#         print(f"Voice enrollment submitted. Voice ID: {voice_id}")

#         # Poll for status
#         for _ in range(30):
#             voice_info = service.query_voice(voice_id=voice_id)
#             status = voice_info.get("status")
#             print(f"Status: {status}")
#             if status == "OK":
#                 print("Voice ready!")
#                 return voice_id
#             time.sleep(5)
#     except Exception as e:
#         print(f"Error: {e}")
#     return None


# if __name__ == "__main__":
#     enroll_voice()
