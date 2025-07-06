import torch
import torchaudio
from model import AudioCNN
from dataset import AudioDataset

# 設定計算設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載訓練好的模型
model = AudioCNN(num_classes=len(AudioDataset().cat_strs))  # 類別數
model.load_state_dict(torch.load('voice/model/audio_cnn_model6.pth'))  # 加載模型權重
model = model.to(device)
model.eval()  # 設置模型為推理模式

# 讀取測試音檔
def load_and_preprocess_audio(file_path, duration=3, sample_rate_0_2=44100):
    waveform, sample_rate = torchaudio.load(file_path)  # 讀取音檔
    waveform = waveform.mean(0)  # 將多聲道取平均

    # 填充或截取音檔
    if waveform.size(0) < duration * sample_rate_0_2:
        padding = torch.zeros(duration * sample_rate_0_2 - waveform.size(0))
        waveform = torch.cat([waveform, padding], 0)
    else:
        waveform = waveform[:duration * sample_rate_0_2]

    # 構建梅爾頻譜圖
    mel_spectrogram = AudioDataset().mel_spectrogram_44100(waveform.unsqueeze(0))  # 梅爾頻譜圖的形狀應該是 (1, n_mels, time_steps)
    mel_spectrogram = mel_spectrogram.squeeze(0)  # 去掉批次維度

    # 統一時間步長
    target_time_steps = 64
    if mel_spectrogram.size(1) < target_time_steps:
        padding = torch.zeros(mel_spectrogram.size(0), target_time_steps - mel_spectrogram.size(1))
        mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)
    else:
        mel_spectrogram = mel_spectrogram[:, :target_time_steps]
    
    return mel_spectrogram

# 測試音檔
audio_file = 'voice/voice_data/split/snore/snore1_4.wav'  # 替換為你的測試音檔路徑
mel_spectrogram = load_and_preprocess_audio(audio_file).unsqueeze(0).unsqueeze(0).to(device)  # 加載並處理音檔，並將其轉移到GPU

# 使用模型進行推斷
with torch.no_grad():
    outputs = model(mel_spectrogram)
    _, predicted_class = torch.max(outputs, 1)


label=str(predicted_class.item())
print(f"Predicted Class: {label}")

if label=="0":
    print("無打呼")
elif label=="1":
    print("有呼吸中止症")
else:
    print("正常打呼")
# 顯示預測結果

