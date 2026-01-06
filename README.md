 📏 AI 기반 거리 측정 웹 애플리케이션 (AI Distance Estimator)

이 프로젝트는 Python Flask와 YOLOv8 딥러닝 모델을 활용하여, 사진 속의 사람을 자동으로 인식하고 카메라와의 거리를 추정해주는 웹 플랫폼입니다. 구글 클라우드(Google Cloud Run) 환경에서 실행되도록 설계되었습니다.

 📋 주요 기능 (Features)

1.  이미지 선택 및 분석 (Image Selection):
     서버에 저장된 다양한 샘플 이미지들을 드롭다운 목록으로 제공합니다.
     사용자가 이미지를 선택하면 즉시 분석을 시작합니다.
2.  객체 인식 (Object Detection):
     최신 YOLOv8 모델을 사용하여 이미지 내의 '사람(Person)'을 실시간으로 탐지합니다.
3.  거리 측정 (Distance Estimation):
     삼각형 닮음비(Triangle Similarity) 원리를 적용하여 대상과의 거리를 계산합니다.
     성인 평균 어깨 너비(약 50cm)를 기준값으로 사용하여 거리를 추정합니다.
4.  시각화 (Visualization):
     인식된 대상에 바운딩 박스(Bounding Box)를 그리고, 계산된 거리(m)를 이미지 위에 표시합니다.

 🛠 기술 스택 (Tech Stack)

 Language: Python 3.10
 Web Framework: Flask
 AI & Vision: Ultralytics YOLOv8, OpenCV (Headless), NumPy
 Infrastructure: Docker, Google Cloud Build, Google Cloud Run

 📂 프로젝트 구조 (File Structure)


.
├── app.py                  메인 웹 애플리케이션 (Flask 서버 + 거리 계산 로직)
├── Dockerfile              클라우드 배포를 위한 도커 설정 파일
├── requirements.txt        프로젝트에 필요한 파이썬 라이브러리 목록
├── yolov8n.pt              YOLO AI 모델 (최초 실행 시 자동 다운로드)
└── images.jpeg            분석 테스트용 샘플 이미지 파일들



 ⚙️ 거리 측정 원리 (Logic)

이 앱은 단일 카메라(Monocular Camera) 환경에서 거리를 추정하기 위해 초점 거리(Focal Length) 보정 방식을 사용합니다.

$$ Distance = \frac{\text{Known Width} \times \text{Focal Length}}{\text{Pixel Width}} $$

 Known Width: 대상의 실제 너비 (본 프로젝트에서는 사람 어깨 너비 50cm로 설정)
 Focal Length: 카메라의 초점 거리 상수 (사전 테스트를 통해 600으로 설정됨)
 Pixel Width: AI가 사진에서 인식한 객체의 픽셀 너비

> 참고: 정확한 거리 측정을 위해서는 사용하는 카메라에 맞춰 `app.py` 내의 `FOCAL_LENGTH` 값을 보정(Calibration)해야 합니다.

 🚀 설치 및 실행 방법 (Local)

로컬 컴퓨터에서 테스트하려면 다음 단계를 따르세요.

1. 저장소 클론:

git clone [GitHub 주소]




2. 라이브러리 설치:

pip install -r requirements.txt




3. 앱 실행:

python app.py




4. 접속: 웹 브라우저를 열고 `http://localhost:8080` 으로 접속합니다.

 ☁️ 배포 (Cloud)

이 프로젝트는 `Dockerfile`이 포함되어 있어 Google Cloud Build를 통해 즉시 배포 가능합니다. GitHub에 코드를 Push하면 Cloud Build 트리거가 작동하여 자동으로 배포됩니다.

---

© 2026 Medical AI Project



