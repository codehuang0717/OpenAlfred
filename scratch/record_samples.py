import sounddevice as sd
from scipy.io import wavfile
import os
import time

# 配置
WAKEWORD_NAME = "alfred"
SAMPLE_RATE = 16000  # 必须是 16000Hz
DURATION = 2  # 每次录制 2 秒
# 获取当前文件所在目录的父目录的父目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "custom_samples", WAKEWORD_NAME)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_sample(index):
    print(f"\n[{index}] 准备录音... 请在看到 '开始' 后说出 '{WAKEWORD_NAME}'")
    time.sleep(0.5)
    print(">>> 开始录音！")
    
    # 录制单声道音频
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    
    filename = os.path.join(OUTPUT_DIR, f"{WAKEWORD_NAME}_{index}_{int(time.time())}.wav")
    wavfile.write(filename, SAMPLE_RATE, recording)
    print(f"已保存: {filename}")

if __name__ == "__main__":
    print(f"--- 唤醒词录制工具 ---")
    print(f"建议：录制时尝试不同的距离、音量和语速。")
    print(f"保存路径: {OUTPUT_DIR}")
    
    count = 0
    try:
        while True:
            cmd = input(f"\n按 [Enter] 开始录制第 {count+1} 个样本 (输入 'q' 退出)...")
            if cmd.lower() == 'q':
                break
            record_sample(count)
            count += 1
    except KeyboardInterrupt:
        pass
    
    print(f"\n录制结束！总共录制了 {count} 个样本。")
