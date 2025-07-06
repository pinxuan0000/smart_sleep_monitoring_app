# 使用 Python 官方映像檔
FROM python:3.11

# 設定工作目錄
WORKDIR /app


RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 複製本機檔案到容器中
COPY requirements.txt .

# 安裝依賴套件
RUN pip install -r requirements.txt

COPY app.py .
# 開放 Flask 預設埠
EXPOSE 5000

# 啟動 Flask 應用
CMD ["python", "app.py"]
