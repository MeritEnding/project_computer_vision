import cv2
import numpy as np

# YOLO 모델 설정 파일과 가중치 파일의 경로
yolo_config_path = "yolov3.cfg"  # 설정 파일 경로
yolo_weights_path = "yolov3.weights"  # 가중치 파일 경로
yolo_classes_path = "coco.names"  # 클래스 이름 파일 경로

# YOLO 모델 초기화
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# 클래스 이름 로드
with open(yolo_classes_path, 'r') as f:
    classes = f.read().strip().split('\n')

# 이미지 읽기
image = cv2.imread("car.jpg")  # 차량을 포함한 이미지 파일 경로

# YOLO 모델 입력 크기 설정 (보통 416x416 사용)
input_size = (416, 416)

# 이미지를 YOLO 모델에 맞게 전처리
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=input_size, swapRB=True, crop=False)

# YOLO 모델에 전처리된 이미지 입력
net.setInput(blob)

# YOLO 모델 출력 계산
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# 차량 개수 초기화
car_count = 0

# 객체 감지 결과 처리
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # 탐지 신뢰도 조건 (일반적으로 0.5 이상인 경우만 인식)
        if confidence > 0.5 and classes[class_id] == 'car':
            car_count += 1

# 차량 개수 출력
print(f"차량 개수: {car_count}")

# 주차장의 총 주차 공간 수
total_parking_spaces = 100  # 예시로 주차 공간 수를 100으로 가정

# 주차장의 남은 주차 공간 수 계산
remaining_parking_spaces = total_parking_spaces - car_count

# 남은 주차 공간 수 출력
print(f"남은 주차 공간 수: {remaining_parking_spaces}")

# 결과 이미지 저장 (선택 사항)
cv2.imwrite("output.jpg", image)

# 결과 이미지 표시 (선택 사항)
cv2.imshow("Vehicle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
