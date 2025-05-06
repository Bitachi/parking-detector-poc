FROM public.ecr.aws/lambda/python:3.9

# 必要なツール・ライブラリのインストール
RUN yum -y install git wget unzip gcc gcc-c++ cmake

# OpenCV など必要なPythonパッケージをインストール
RUN pip install --upgrade pip
RUN pip install torch torchvision opencv-python-headless

# YOLOv5 のコードをクローン
RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install -r yolov5/requirements.txt

# ハンドラーを設定（app.py の handler 関数をエントリポイントに）
COPY app.py ./
CMD ["app.lambda_handler"]