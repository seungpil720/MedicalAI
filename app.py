import base64
import os
import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# 1. 모델 및 설정
# ==========================================
# 모델 로드 (서버 시작 시 1회)
model = YOLO("yolov8n.pt")
KNOWN_WIDTH = 50.0  # cm
FOCAL_LENGTH = 600

def distance_finder(focal_length, real_object_width, width_in_pixels):
    if width_in_pixels == 0: return 0
    return (real_object_width * focal_length) / width_in_pixels

# ==========================================
# 2. 메인 로직
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    # 현재 폴더의 모든 이미지 파일 검색 (대소문자 무관)
    all_files = os.listdir('.')
    image_list = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_list.sort() # 파일명 순서대로 정렬

    selected_filename = None
    img_data = None
    summary_text = ""

    # [POST] 사용자가 사진을 선택하고 '분석' 버튼을 눌렀을 때
    if request.method == 'POST':
        selected_filename = request.form.get('filename')
        
        # 선택된 파일이 실제로 존재하는지 확인 (보안 및 에러 방지)
        if selected_filename and selected_filename in image_list:
            try:
                # 이미지 읽기
                img = cv2.imread(selected_filename)
                
                if img is not None:
                    # YOLO 분석
                    results = model(img)
                    detected_distances = []
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = model.names[cls]

                            if class_name == 'person':
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                w_pixel = x2 - x1
                                
                                dist_cm = distance_finder(FOCAL_LENGTH, KNOWN_WIDTH, w_pixel)
                                dist_m = dist_cm / 100
                                detected_distances.append(f"{dist_m:.2f}m")
                                
                                # 그리기
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{dist_m:.2f}m"
                                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
                                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # 결과 이미지 인코딩
                    _, buffer = cv2.imencode('.jpg', img)
                    img_data = base64.b64encode(buffer).decode('utf-8')
                    
                    if detected_distances:
                        summary_text = f"✅ {len(detected_distances)}명 감지됨 (거리: {', '.join(detected_distances)})"
                    else:
                        summary_text = "❌ 감지된 사람이 없습니다."
                else:
                    summary_text = "⚠️ 이미지 파일을 읽을 수 없습니다."

            except Exception as e:
                summary_text = f"⚠️ 에러 발생: {str(e)}"

    return render_template_string(HTML_TEMPLATE, 
                                  images=image_list, 
                                  selected=selected_filename, 
                                  img_data=img_data, 
                                  summary=summary_text)

# ==========================================
# 3. HTML 템플릿 (디자인)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical AI Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; text-align: center; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; margin-bottom: 20px; }
        select { padding: 10px; font-size: 16px; width: 70%; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }
        button { padding: 10px 20px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result-box { margin-top: 30px; border-top: 2px solid #eee; padding-top: 20px; }
        img { max-width: 100%; height: auto; border-radius: 8px; border: 2px solid #333; margin-top: 15px; }
        .summary { font-size: 1.2em; font-weight: bold; color: #28a745; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Medical AI 거리 분석기</h1>
        <p>서버에 업로드된 이미지 중 하나를 선택하세요.</p>
        
        <form method="POST">
            <select name="filename">
                <option value="" disabled {% if not selected %}selected{% endif %}>-- 이미지 선택 --</option>
                {% for img in images %}
                    <option value="{{ img }}" {% if img == selected %}selected{% endif %}>{{ img }}</option>
                {% endfor %}
            </select>
            <br>
            <button type="submit">🔍 분석 시작</button>
        </form>

        {% if selected %}
            <div class="result-box">
                <h3>📂 선택된 파일: {{ selected }}</h3>
                <div class="summary">{{ summary }}</div>
                {% if img_data %}
                    <img src="data:image/jpeg;base64,{{ img_data }}">
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
