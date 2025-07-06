import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms as T
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    # 初始設定
    def __init__(
        self,
        data_dir="voice/voice_data/split",
        sample_rate_0_2=44100,  # 類別 0 和 2 的音檔頻率
        sample_rate_1=22050,  # 類別 1 的音檔頻率
        duration=3,  # 音檔秒數
        n_fft=1024,  # 樣本數 1024
        win_length=None,
        hop_length_44100=2085,  # 計算出來的 hop_length，這樣可以確保有 64 個時間步
        hop_length_22050=1034,  # 計算出來的 hop_length，這樣可以確保有 64 個時間步
        n_mels=24
    ):
        self.data_dir = data_dir
        self.duration = duration
        self.sample_rate_0_2 = sample_rate_0_2
        self.sample_rate_1 = sample_rate_1
        self.hop_length_44100 = hop_length_44100
        self.hop_length_22050 = hop_length_22050


        self.mel_spectrogram_44100 = T.MelSpectrogram(
            sample_rate=sample_rate_0_2,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length_44100,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )

        self.mel_spectrogram_22050 = T.MelSpectrogram(
            sample_rate=sample_rate_1,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length_22050,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )

        self.files = []  # 每個音檔路徑
        self.cats = []   # 音檔的資料夾是哪個編號
        self.cat_strs = []  # 分類(各個資料夾的名稱)

        for idx, cat_dir in enumerate(os.listdir(data_dir)):
            cat_path = os.path.join(data_dir, cat_dir)
            cat_files = os.listdir(cat_path)
            self.cat_strs.append(cat_dir)
            self.files.extend([os.path.join(cat_path, file) for file in cat_files])
            self.cats.extend([idx] * len(cat_files))

    # 定義資料集長度
    def __len__(self):
        return len(self.files)

    # 資料集索引訪問邏輯
    def __getitem__(self, item):
        file, category = self.files[item], self.cats[item]  # 獲取音檔路徑和分類
        waveform, sample_rate = torchaudio.load(file)  # 讀取音檔
        #print(f"Length of audio at index {item}: {waveform.size(1) / sample_rate} seconds")
        waveform = waveform.mean(0)  # 將多聲道取平均（如果有多聲道）

        # 根據類別選擇合適的 sample_rate 並進行重採樣
        if category == 1:  # 類別1的音檔需要是22050
            #print("use22050")
            resample_transform = T.Resample(sample_rate, self.sample_rate_1)
            waveform = resample_transform(waveform)
            mel_spectrogram = self.mel_spectrogram_22050(waveform)
        else:  # 類別0和2的音檔需要是44100
            #print("use44100")
            resample_transform = T.Resample(sample_rate, self.sample_rate_0_2)
            waveform = resample_transform(waveform)
            mel_spectrogram = self.mel_spectrogram_44100(waveform)

        # 音檔填充或是自動切割
        if waveform.size(0) < self.duration * self.sample_rate_0_2:
            padding = torch.zeros(self.duration * self.sample_rate_0_2 - waveform.size(0))
            waveform = torch.cat([waveform, padding], 0)  # 拼接填充的數值
        else:
            waveform = waveform[:self.duration * self.sample_rate_0_2]  # 擷取到設定的長度

        # 添加批次維度，變成 (1, channels, samples)
        waveform = waveform.unsqueeze(0)  # 新的形狀: (1, 1, 132300)

        # 梅爾頻譜圖的形狀應該是 (n_mels, time_steps)，現在去掉批次維度
        # mel_spectrogram = mel_spectrogram.squeeze(0)  # 形狀變成 (n_mels, time_steps)

        # 統一梅爾頻譜圖的時間步長，這裡你可以根據最大時間步長進行填充或裁剪
        target_time_steps = 64  # 根據需要設置時間步長
        if mel_spectrogram.size(1) < target_time_steps:
            padding = torch.zeros(mel_spectrogram.size(0), target_time_steps - mel_spectrogram.size(1))
            mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)
        else:
            mel_spectrogram = mel_spectrogram[:, :target_time_steps]

        one_hot_category = torch.zeros(len(self.cat_strs))  # 建立一個資料夾長度的0 list
        one_hot_category[category] = 1  # 在對應分類的位置填上1

        #print(mel_spectrogram.shape)
        return mel_spectrogram, one_hot_category


if __name__ == "__main__":
    dataset = AudioDataset()

    spectrum, cat = dataset[0]
    print(f"Class of dataset[0]: {cat}")
    #print(f"Spectrum shape at th1 {spectrum.shape}")
    plt.imshow(spectrum.log2().detach().numpy(), aspect="auto", origin="lower")
    plt.show()

    spectrum, cat = dataset[203]
    print(f"Class of dataset[203]: {cat}")
    plt.imshow(spectrum.log2().detach().numpy(), aspect="auto", origin="lower")
    plt.show()

    spectrum, cat = dataset[400]
    print(f"Class of dataset[400]: {cat}")
   # print(f"Spectrum shape at th3: {spectrum.shape}")
    plt.imshow(spectrum.log2().detach().numpy(), aspect="auto", origin="lower")
    plt.show()

    print()
