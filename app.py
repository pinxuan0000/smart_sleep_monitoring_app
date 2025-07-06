from flask import Flask, request, jsonify
import torch
import mediapipe as mp
import numpy as np
import sounddevice as sd
from voice.model import AudioCNN
from voice.dataset import AudioDataset
from sleep_pose.model import SleepPostureModel
import io
from PIL import Image
from db import SessionLocal,init_db,  PredictionRecord
from datetime import datetime

app = Flask(__name__)
init_db()

# 載入姿勢模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_model = SleepPostureModel().to(device)
pose_model.load_state_dict(torch.load("sleep+pose/model/sleep_posture_model_100.pth"))
pose_model.eval()

# 載入音訊模型

audio_model = AudioCNN(num_classes=len(AudioDataset().cat_strs))
audio_model.load_state_dict(torch.load('voice/model/audio_cnn_model6.pth'))
audio_model = audio_model.to(device)
audio_model.eval()

# Mediapipe 姿勢設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

pose_label_map = {0: "left_side", 1: "right_side", 2: "supine", 3: "lie"}

# Helper function: 姿勢推論
def predict_pose_from_image(image: Image.Image):
    # 將 PIL Image 轉成 mediapipe 格式
    image_rgb = np.array(image.convert('RGB'))
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"error": "No pose landmarks found."}

    keypoints = results.pose_landmarks.landmark
    input_tensor = []

    # 取指定的關鍵點 index
    indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for i in indices:
        input_tensor.extend([keypoints[i].x, keypoints[i].y])

    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = pose_model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    posture_label = pose_label_map[predicted_label]
    return {"posture": posture_label}

# Helper function: 音訊推論
label_map = {0: "non-snore", 1: "sleep_apnea", 2: "normal_snore"}

def record_and_predict(duration=3, sample_rate=44100):
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

    return label_map[predicted_class.item()]



# API 路由：查詢歷史推論結果
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    db = SessionLocal()
    records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(20).all()
    db.close()
    return jsonify([
        {"id": r.id, "type": r.api_type, "result": r.result, "timestamp": r.timestamp.isoformat()}
        for r in records
    ])

# API 路由：接收圖片，回傳姿勢判斷結果
@app.route("/api/predict_pose", methods=["POST"])
def api_predict_pose():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Invalid image file."}), 400

    result = predict_pose_from_image(image)

    # 儲存到資料庫
    db = SessionLocal()
    record = PredictionRecord(api_type='pose', result=result)
    db.add(record)
    db.commit()
    db.close()


    return jsonify({"result": result})

# API 路由：接收音訊檔案（WAV格式），回傳打鼾判斷結果
@app.route("/api/start-recording", methods=["POST"])
def start_recording():
    try:
        result = record_and_predict()
        # 儲存到資料庫
        db = SessionLocal()
        record = PredictionRecord(api_type='audio', result=result)
        db.add(record)
        db.commit()
        db.close()

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
