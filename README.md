# 차량 번호판 YOLOv8 프로젝트

이 디렉터리는 한국 차량 번호판 탐지를 위한 YOLOv8 기반 프로젝트입니다.  
모델 가중치(`runs/license_plate_yolov8n/weights/best.pt`)는 포함하지 않지만, 동일한 설정으로 재학습할 수 있는 코드와 안내를 제공합니다.

## 1. 환경 준비

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 데이터셋 경로

1. [Roboflow Korea Car License Plate Dataset](https://universe.roboflow.com/yolov8-uv8pz/korea-car-license-plate-ptpzh/dataset/1)에 접속해 `YOLOv8` 포맷으로 다운로드합니다.
2. 압축을 해제한 뒤 디렉터리를 `Korea Car License Plate/` 이름으로 저장소와 같은 상위 경로에 배치합니다. 예를 들어, 이 프로젝트가 `llama_young/차량번호판인식_yolov8/`에 있다면 데이터셋은 `llama_young/Korea Car License Plate/`에 위치해야 합니다.

`Korea Car License Plate/` 폴더가 이 디렉터리와 같은 상위 경로(`/home/work/llama_young`) 안에 위치해야 합니다.  
데이터 구성은 `Korea Car License Plate/data.yaml`을 참고하세요.

```
llama_young/
├─ Korea Car License Plate/
│  ├─ train/
│  ├─ valid/
│  ├─ test/
│  └─ data.yaml
└─ 차량번호판인식_yolov8/
   ├─ requirements.txt
   ├─ train_license_plate.py
   └─ ...
```

## 3. 학습 실행

```bash
python train_license_plate.py \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0  # GPU 사용 시
```

- 기본 모델: `yolov8n.pt`
- 기본 데이터: `../Korea Car License Plate/data.yaml`
- 학습 결과: `runs/license_plate_yolov8n/`

필요에 따라 `--model`, `--project`, `--name` 등을 조정하세요.

## 4. 노트북

- `license_plate_demo.ipynb`: 기본 추론 예제
- `license_plate_filtered_demo.ipynb`: NMS/박스 필터링 적용 추론 예제
- `model_trainin.ipynb`: 노트북 환경에서 바로 학습을 재현하는 예제
- `ocr_yolo.ipynb`: YOLO 탐지 + CRNN OCR 파이프라인을 통해 번호판 문자열을 추론하는 예제

노트북 실행 시에는 `venv`를 커널로 등록하거나, 1번 셀에서 제공하는 경로 설정 코드를 실행하여 패키지를 불러오세요.

## 5. 모델 평가

학습이 끝난 뒤 `ultralytics`의 검증 커맨드를 활용하면 검증 지표를 확인할 수 있습니다.

```bash
venv/bin/yolo detect val \
    model=runs/license_plate_yolov8n/weights/best.pt \
    data="../Korea Car License Plate/data.yaml"
```
