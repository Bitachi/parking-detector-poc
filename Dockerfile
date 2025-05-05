FROM python:3.10-slim

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの作成
WORKDIR /app

# YOLOv5をクローン
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# 必要な依存関係をインストール
RUN cd yolov5 && pip install -r requirements.txt
