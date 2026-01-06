import base64
import os
import cv2
import numpy as np
from flask import Flask, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# 1. 모델 설정
# ==========================================
model = YOLO("yolov8n.pt")
KNOWN_WIDTH = 50.0  # cm
FOCAL_LENGTH = 600

def distance_finder(focal_length, real_object_width, width_in_pixels):
    if width_in_pixels == 0: return 0
    return (real_object_width * focal_length) / width_in_pixels

# ==========================================
# 2. 메인 로직 (폴더 내 이미지 자동 분석)
# ==========================================
@app.route('/')
def home():
    # 현재 폴더에서 이미지 파일들만 찾기 (.jpg, .jpeg, .png 대소문자 무관)
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort() # 이름 순 정렬

    results_html = ""

    if not image_files:
        return "<h1>이미지 파일을 찾을 수 없습니다. (images*.jpeg 파일을 확인해주세요)</h1>"

    for file_name in image_files:
        try:
            # 이미지 읽기
            img = cv2.imread(file_name)
            
            # [중요] 이미지가 제대로 안 읽혔으면 건너뛰기 (에러 방지)
            if img is None:
                continue

            # YOLO 분석
            results = model(img)
            detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    if class_name == 'person':
                        detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w_pixel = x2 - x1
                        
                        dist_cm = distance_finder(FOCAL_LENGTH, KNOWN_WIDTH, w_pixel)
                        dist_m = dist_cm / 100
                        
                        # 박스와 텍스트 그리기
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{dist_m:.2f}m"
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 분석된 이미지를 HTML로 변환
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            results_html += f"""
            <div style="display: inline-block; margin: 10px; border: 1px solid #ccc; padding: 10px;">
                <h3>📂 {file_name}</h3>
                <img src="data:image/jpeg;base64,{img_str}" style="max-width: 400px; height: auto;">
            </div>
            """
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    return render_template_string(TEMPLATE, content=results_html)

# ==========================================
# 3. HTML 템플릿
# ==========================================
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical AI Gallery</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 20px; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <h1>📸 분석 결과 갤러리</h1>
    <p>저장소에 있는 이미지들을 자동으로 분석한 결과입니다.</p>
    <hr>
    {{ content|safe }}
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
