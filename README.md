# Object Distance Estimator (YOLOv8 & OpenCV)

사진 속의 사물(예: 사람)을 인식하고, 카메라로부터의 거리를 추정하여 알려주는 파이썬 프로젝트입니다.

## 📋 기능
1. **사물 판별:** YOLOv8 모델을 사용하여 사진 속 객체를 인식합니다.
2. **거리 측정:** 삼각형 닮음비(Triangle Similarity) 원리를 이용해 대략적인 거리를 계산합니다.
3. **결과 출력:** 인식된 사물에 박스를 그리고 거리를 표시한 이미지를 생성합니다.

## ⚙️ 설치 방법 (Installation)

1. 이 저장소를 클론(Clone)합니다.
   ```bash
   git clone [https://github.com/사용자명/레포지토리명.git](https://github.com/사용자명/레포지토리명.git)
