import cv2
import mediapipe as mp
import torch
import numpy as np
import torchaudio
import sounddevice as sd
from voice.model import AudioCNN
from voice.dataset import AudioDataset
from sleep_pose.model import SleepPostureModel
import time


# 姿勢模型載入
pose_model = SleepPostureModel()
pose_model.load_state_dict(torch.load("sleep_posture_model_100.pth"))
pose_model.eval()

# 聲音模型載入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = AudioCNN(num_classes=len(AudioDataset().cat_strs))
audio_model.load_state_dict(torch.load('voice/model/audio_cnn_model6.pth'))
audio_model = audio_model.to(device)
audio_model.eval()

# Mediapipe 姿勢設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 姿勢標籤
pose_label_map = {0: "left_side", 1: "right_side", 2: "supine", 3: "lie"}

# 打呼標籤
def predict_audio():
    duration = 3  # 3秒
    sample_rate = 44100
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    waveform = torch.tensor(recording.squeeze(), dtype=torch.float32)

    if waveform.size(0) < duration * sample_rate:
        padding = torch.zeros(duration * sample_rate - waveform.size(0))
        waveform = torch.cat([waveform, padding], 0)
    else:
        waveform = waveform[:duration * sample_rate]

    mel_spectrogram = AudioDataset().mel_spectrogram_44100(waveform.unsqueeze(0))
    mel_spectrogram = mel_spectrogram.squeeze(0)

    target_time_steps = 64
    if mel_spectrogram.size(1) < target_time_steps:
        padding = torch.zeros(mel_spectrogram.size(0), target_time_steps - mel_spectrogram.size(1))
        mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)
    else:
        mel_spectrogram = mel_spectrogram[:, :target_time_steps]

    mel_spectrogram = mel_spectrogram.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = audio_model(mel_spectrogram)
        _, predicted_class = torch.max(outputs, 1)

    label = str(predicted_class.item())
    if label == "0":
        return "non-snore"
    elif label == "1":
        return "sleep_apnea"
    else:
        return "normal_snore"

# 攝影機設定

# rtsp_url = 0
rtsp_url ="rtsp://Miswork:4080520@172.20.10.2:554/live/profile0/stream2"
cap = cv2.VideoCapture(rtsp_url)

# 時間與狀態追蹤變數
last_audio_time = time.time()
audio_interval = 60  # 每60秒做一次音訊辨識
current_audio_label = ""
last_pose_label = None
turn_count = 0

# 音訊偵測計時 
import time
last_audio_time = time.time()
audio_interval = 5  # 每5秒偵測一次聲音
current_audio_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        keypoints = results.pose_landmarks.landmark
        input_tensor = []

        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            input_tensor.extend([keypoints[i].x, keypoints[i].y])

            h, w, _ = frame.shape
            x, y = int(keypoints[i].x * w), int(keypoints[i].y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = pose_model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        posture_label = pose_label_map[predicted_label]
        if last_pose_label is None:
            last_pose_label = posture_label
            print(f"Posture: {posture_label}")
        elif posture_label != last_pose_label:
            turn_count += 1
            print(f"Posture: {posture_label} (Turn_over: {turn_count})")
            last_pose_label = posture_label

    # 每隔一段時間做一次音訊辨識
    if time.time() - last_audio_time > audio_interval:
        current_audio_label = predict_audio()
        print(f"Snore: {current_audio_label}")
        last_audio_time = time.time()
    # 顯示音訊結果
    #cv2.putText(frame, f"Snore: {current_audio_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Sleep Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
