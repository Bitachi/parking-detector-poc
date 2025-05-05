FROM python:3.10-slim

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# YOLOv5 をクローンし依存関係インストール
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5
RUN pip install --no-cache-dir -r /app/yolov5/requirements.txt

# アプリとモデルをコピー
COPY app.py /app/app.py
COPY yolov5s.pt /app/yolov5s.pt

# Lambda エントリーポイント
CMD ["python", "app.py"]
