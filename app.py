import cv2
import torch
import math
import csv

# モデルの読み込み（事前に yolov5s.pt を配置しておく）
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
model.conf = 0.4  # 信頼度の閾値

# 動画ファイルの読み込み
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = int(fps * 5)  # 5秒間隔（例: 30fpsなら150フレーム）
stop_threshold = int(30 / 5)   # 停止判定（30秒以上なら6回連続で位置変化なし）

# 停止判定のためのピクセル距離しきい値
move_threshold = 20

# トラッカー
vehicle_tracker = {}  # {vehicle_id: {'position': (x, y), 'stopped_count': n}}
vehicle_id_counter = 0

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# CSV書き出し準備
csv_file = open('vehicle_detection_results.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Vehicle ID', 'Class', 'X1', 'Y1', 'X2', 'Y2', 'Stopped'])

frame_number = 0

while frame_number < frame_count:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        break

    # 検出処理
    results = model(frame)
    detections = []

    for *box, conf, cls in results.xyxy[0]:
        class_name = model.names[int(cls)]
        if class_name in ['car', 'truck', 'bus', 'motorcycle']:
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append(((cx, cy), (x1, y1, x2, y2), class_name))

    updated_tracker = {}
    matched_ids = []

    for center, bbox, class_name in detections:
        matched_id = None
        for vid, data in vehicle_tracker.items():
            dist = euclidean(center, data['position'])
            if dist < move_threshold:
                matched_id = vid
                break

        if matched_id is not None:
            updated_tracker[matched_id] = {
                'position': center,
                'stopped_count': vehicle_tracker[matched_id]['stopped_count'] + 1
            }
            stopped = 'Yes' if updated_tracker[matched_id]['stopped_count'] >= stop_threshold else 'No'
        else:
            vehicle_id_counter += 1
            matched_id = vehicle_id_counter
            updated_tracker[matched_id] = {
                'position': center,
                'stopped_count': 0
            }
            stopped = 'No'

        # 書き出し
        x1, y1, x2, y2 = bbox
        csv_writer.writerow([frame_number, matched_id, class_name, x1, y1, x2, y2, stopped])

    vehicle_tracker = updated_tracker
    frame_number += frame_interval  # 次のフレームへジャンプ

cap.release()
csv_file.close()
