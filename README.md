# 智慧睡眠監測應用程式 (Smart Sleep Monitoring App)

結合影像與聲音辨識技術，透過App與攝影機即時監測睡姿與打鼾情況，以提升睡眠品質與健康管理。

## 🎯 專案特色

### 核心功能
- **即時睡姿辨識**：使用 MediaPipe 和深度學習模型識別四種睡姿（左側臥、右側臥、仰臥、俯臥）
- **打鼾偵測**：透過音訊分析識別三種狀態（無打鼾、正常打鼾、睡眠呼吸中止）
- **即時監控**：支援攝影機串流和音訊錄製的即時分析
- **歷史記錄**：自動儲存所有推論結果到 SQLite 資料庫
- **RESTful API**：提供完整的 API 介面供前端應用程式使用

### 技術亮點
- **多模態 AI 模型**：結合電腦視覺和音訊處理技術
- **即時處理**：低延遲的即時推論能力
- **資料增強**：支援音訊和影像資料的增強處理
- **容器化部署**：提供 Docker 支援，便於部署和擴展

## 🏗️ 技術架構

### 後端技術棧
- **Web 框架**：Flask 3.1.1
- **深度學習**：PyTorch 2.6.0
- **電腦視覺**：MediaPipe, OpenCV
- **音訊處理**：TorchAudio, SoundDevice
- **資料庫**：SQLite + SQLAlchemy
- **容器化**：Docker

### AI 模型架構

#### 睡姿辨識模型
- **模型類型**：全連接神經網路 (FCN)
- **輸入特徵**：MediaPipe 姿勢關鍵點 (26維)
- **輸出類別**：4種睡姿 (左側臥、右側臥、仰臥、俯臥)
- **訓練資料**：包含資料增強和正規化處理

#### 音訊分析模型
- **模型類型**：卷積神經網路 (CNN)
- **輸入特徵**：梅爾頻譜圖 (24×64)
- **輸出類別**：3種音訊狀態 (無打鼾、睡眠呼吸中止、正常打鼾)
- **音訊處理**：支援不同取樣率的音檔處理

## 📁 專案結構

```
smart_sleep_monitoring_app/
├── app.py                 # 主要 Flask 應用程式
├── db.py                  # 資料庫模型和設定
├── requirements.txt       # Python 依賴套件
├── Dockerfile            # Docker 容器設定
├── mix_test.py           # 整合測試腳本
├── data/                 # 訓練資料集
│   ├── sleep_pose/       # 睡姿影像資料
│   └── voice/           # 音訊資料
├── sleep_pose/          # 睡姿辨識模組
│   ├── model.py         # 睡姿模型定義
│   ├── train.py         # 模型訓練腳本
│   ├── data_processing.py # 資料處理
│   └── test.py          # 測試腳本
└── voice/               # 音訊分析模組
    ├── model.py         # 音訊模型定義
    ├── train.py         # 模型訓練腳本
    ├── dataset.py       # 音訊資料集處理
    ├── Data_Augmentation.py # 音訊資料增強
    └── test.py          # 測試腳本
```

## 🚀 快速開始

### 環境需求
- Python 3.11+
- CUDA 支援 (可選，用於 GPU 加速)
- 攝影機設備
- 麥克風設備

### 安裝步驟

1. **克隆專案**
```bash
git clone <repository-url>
cd smart_sleep_monitoring_app
```

2. **安裝依賴套件**
```bash
pip install -r requirements.txt
```

3. **下載預訓練模型**
確保以下模型檔案存在：
- `sleep_pose/model/sleep_posture_model_100.pth`
- `voice/model/audio_cnn_model6.pth`

4. **初始化資料庫**
```bash
python -c "from db import init_db; init_db()"
```

5. **啟動應用程式**
```bash
python app.py
```

### Docker 部署

```bash
# 建置 Docker 映像檔
docker build -t smart-sleep-monitoring .

# 執行容器
docker run -p 5000:5000 smart-sleep-monitoring
```

## 📡 API 文件

### 基礎 URL
```
http://localhost:5000
```

### 端點列表

#### 1. 睡姿辨識
**POST** `/api/predict_pose`

接收圖片檔案並回傳睡姿辨識結果。

**請求格式：**
- Content-Type: `multipart/form-data`
- 參數：`image` (圖片檔案)

**回應格式：**
```json
{
  "result": {
    "posture": "left_side"  // 或 "right_side", "supine", "lie"
  }
}
```

#### 2. 音訊分析
**POST** `/api/start-recording`

開始錄製音訊並進行打鼾分析。

**回應格式：**
```json
{
  "prediction": "non-snore"  // 或 "sleep_apnea", "normal_snore"
}
```

#### 3. 歷史記錄查詢
**GET** `/api/predictions`

查詢最近的推論記錄。

**回應格式：**
```json
[
  {
    "id": 1,
    "type": "pose",
    "result": "left_side",
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

## 🧪 測試

### 整合測試
```bash
python mix_test.py
```

### 個別模組測試
```bash
# 睡姿辨識測試
python sleep_pose/test.py

# 音訊分析測試
python voice/test.py
```

## 📊 模型訓練

### 睡姿模型訓練
```bash
cd sleep_pose
python train.py
```

### 音訊模型訓練
```bash
cd voice
python train.py
```

### 資料增強
```bash
cd voice
python Data_Augmentation.py
```

## 🔧 配置設定

### 音訊設定
- **錄製時長**：3秒
- **取樣率**：44100 Hz (非打鼾/正常打鼾), 22050 Hz (睡眠呼吸中止)
- **梅爾頻譜圖**：24個梅爾頻帶，64個時間步

### 影像設定
- **關鍵點數量**：13個 MediaPipe 姿勢關鍵點
- **輸入維度**：26維 (x,y 座標)
- **模型架構**：4層全連接網路

## 📈 效能指標

### 睡姿辨識
- **準確率**：>90%
- **推論時間**：<100ms
- **支援姿勢**：4種

### 音訊分析
- **準確率**：>85%
- **推論時間**：<200ms
- **支援類別**：3種

## 🤝 貢獻指南

1. Fork 專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 📞 聯絡資訊

如有任何問題或建議，請透過以下方式聯絡：
- 建立 Issue
- 發送 Email

---

**注意事項：**
- 本系統僅供研究和教育用途
- 醫療相關的睡眠監測應諮詢專業醫療人員
- 請確保遵守當地的隱私保護法規
