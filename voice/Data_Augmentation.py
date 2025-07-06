import librosa
import numpy as np
import soundfile as sf
import os
from glob import glob

# 增強方法
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def change_volume(data, gain=1.05):
    return data * gain

def my_pitch_shift(data, sr, n_steps=1):
    try:
        return librosa.effects.pitch_shift(data, sr, n_steps)
    except Exception as e:
        print(f"音調處理錯誤: {e}")
        return data  # 如果出錯，返回原始資料

# 資料夾路徑
input_folder = 'voice/voice_data/split/sick'         # 你的原始呼吸中止音檔資料夾
output_folder = 'voice/voice_data/split/sick_data'   # 增強後的資料要存這裡

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 讀取所有wav檔
file_list = glob(os.path.join(input_folder, '*.wav'))
print(f"檔案數量: {len(file_list)}")  # 確認讀取到的檔案數量

# 每個檔案做增強
for file_path in file_list:
    try:
        # 讀取音檔
        y, sr = librosa.load(file_path, sr=22050)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"處理檔案: {file_path}，取樣率: {sr}，資料長度: {len(y)}")
        
        # 保存原始音檔
        output_file = os.path.join(output_folder, f'{base_name}_original.wav')
        sf.write(output_file, y, sr)
        
        # 加噪音
        output_file = os.path.join(output_folder, f'{base_name}_noise.wav')
        sf.write(output_file, add_noise(y), sr)
        
        # 音量增大
        output_file = os.path.join(output_folder, f'{base_name}_volume_up.wav')
        sf.write(output_file, change_volume(y, 1.1), sr)
        
        # 音量減小
        output_file = os.path.join(output_folder, f'{base_name}_volume_down.wav')
        sf.write(output_file, change_volume(y, 0.9), sr)
        
        # 音調升高
        output_file = os.path.join(output_folder, f'{base_name}_pitch_up.wav')
        sf.write(output_file, my_pitch_shift(y, sr, n_steps=1), sr)
        
        # 音調降低
        output_file = os.path.join(output_folder, f'{base_name}_pitch_down.wav')
        sf.write(output_file, my_pitch_shift(y, sr, n_steps=-1), sr)

    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {e}")
        continue  # 繼續處理下一個檔案

print("資料增強完成")
