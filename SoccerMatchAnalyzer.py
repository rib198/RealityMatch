# SoccerMatchAnalyzer (YOLOv5-only version)
# يعتمد فقط على YOLOv5 الرسمي لتحليل اللاعبين والكرة من فيديو مسجل

import torch
import cv2
import json

# --- تحميل النموذج ---
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # يمكنك تغييره إلى yolov5m أو yolov5l لو عندك GPU قوي

# --- إعداد الفيديو ---
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)
frame_index = 0
frames = []

print("🚀 بدء قراءة الفيديو...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()
print(f"✅ تم تحميل {len(frames)} إطارًا من الفيديو")

# --- المعالجة ---
output_frames = []

for frame in frames:
    results = model(frame)
    detections = results.pandas().xyxy[0]  # النتائج على شكل DataFrame

    players = []
    balls = []
    for _, row in detections.iterrows():
        label = row['name']
        x_center = int((row['xmin'] + row['xmax']) / 2)
        y_center = int((row['ymin'] + row['ymax']) / 2)

        if label == 'person':
            players.append({"id": len(players)+1, "team_id": 0, "position": [x_center, y_center]})
        elif label in ['sports ball', 'ball']:
            balls.append({"position": [x_center, y_center]})

    frame_data = {
        "frame_index": frame_index,
        "players": players,
        "goalkeepers": [],
        "referees": [],
        "balls": balls
    }

    output_frames.append(frame_data)
    frame_index += 1

# --- حفظ الملف ---
with open("matchData.json", "w") as f:
    json.dump({"frames": output_frames}, f, indent=2)

print(f"✅ تم استخراج {frame_index} إطارًا إلى matchData.json بنجاح")