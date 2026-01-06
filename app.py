import base64
import io
import os
import numpy as np
import cv2
from flask import Flask, request, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# 모델 및 상수 설정
# ==========================================
# 서버가 켜질 때 모델을 한 번만 로드합니다 (속도 향상)
model = YOLO("yolov8n.pt")

# 거리 측정 상수 (정확도를 위해 추후 보정 필요)
KNOWN_WIDTH = 50.0  # 대상의 실제 너비 (cm, 예: 사람 어깨 평균)
FOCAL_LENGTH = 600  # 초점 거리 (픽셀 단위, 임의 설정값)

def distance_finder(focal_length, real_object_width, width_in_pixels):
    """ 삼각형 닮음비를 이용한 거리 계산 """
    if width_in_pixels == 0: return 0
    return (real_object_width * focal_length) / width_in_pixels

# ==========================================
# 메인 라우트 (접속 및 처리)
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. 파일 업로드 확인
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        # 2. 이미지를 메모리에서 읽어 OpenCV 형식으로 변환
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # 3. YOLO 모델로 사람/사물 인식
        results = model(img)

        # 4. 결과 그리기 및 거리 계산
        detected_items = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # 예: 'person'(사람)인 경우에만 거리 측정 수행
                if class_name == 'person':
                    # 좌표 추출
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w_pixel = x2 - x1
                    
                    # 거리 계산 (cm -> m 변환)
                    dist_cm = distance_finder(FOCAL_LENGTH, KNOWN_WIDTH, w_pixel)
                    dist_m = dist_cm / 100
                    
                    # 화면에 박스와 텍스트 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {dist_m:.2f}m"
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    detected_items.append(f"{dist_m:.2f}m")

        # 5. 처리된 이미지를 웹용 문자열(Base64)로 변환
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        summary_text = f"감지된 인원: {len(detected_items)}명"
        if detected_items:
            summary_text += f" (거리: {', '.join(detected_items)})"

        return render_template_string(RESULT_HTML, img_data=img_base64, summary=summary_text)

    # GET 요청(처음 접속) 시 업로드 화면 보여주기
    return render_template_string(UPLOAD_HTML)

# ==========================================
# HTML 디자인 (업로드 화면 & 결과 화면)
# ==========================================
UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 거리 측정기</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 40px; background-color: #f0f2f5; }
        .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }
        h1 { color: #1a73e8; margin-bottom: 10px; }
        p { color: #666; margin-bottom: 30px; }
        input[type=file] { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 100%; box-sizing: border-box; }
        button { background-color: #1a73e8; color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 8px; cursor: pointer; transition: background 0.3s; width: 100%; }
        button:hover { background-color: #1557b0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📏 AI 거리 측정기</h1>
        <p>사진을 업로드하면 사람과의 거리를 측정합니다.</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">사진 분석하기</button>
        </form>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>측정 결과</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 20px; background-color: #222; color: white; }
        h1 { margin-top: 20px; }
        .summary { color: #4caf50; font-size: 1.2em; margin-bottom: 20px; font-weight: bold; }
        img { max-width: 100%; height: auto; border: 4px solid #555; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
        .btn { display: inline-block; margin-top: 30px; padding: 10px 25px; background: #1a73e8; color: white; text-decoration: none; border-radius: 25px; transition: background 0.3s; }
        .btn:hover { background: #1557b0; }
    </style>
</head>
<body>
    <h1>분석 결과</h1>
    <div class="summary">{{ summary }}</div>
    <img src="data:image/jpeg;base64,{{ img_data }}" alt="Processed Image">
    <br>
    <a href="/" class="btn">🔄 다른 사진 다시하기</a>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
