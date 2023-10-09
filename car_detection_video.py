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

# 동영상 파일 열기
video_capture = cv2.VideoCapture("test2.mp4")  # 동영상 파일 경로

# YOLO 모델 입력 크기 설정 (보통 416x416 사용)
input_size = (416, 416)

# 주차장의 총 주차 공간 수
total_parking_spaces = 100  # 예시로 주차 공간 수를 100으로 가정

while True:
    ret, frame = video_capture.read()  # 프레임 읽어오기
    if not ret:
        break  # 더 이상 프레임이 없을 때 루프 종료

    # 이미지를 YOLO 모델에 맞게 전처리
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=input_size, swapRB=True, crop=False)

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

                # 차량 주위에 네모 박스 그리기
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # 초록색 네모 박스
                cv2.putText(frame, 'car', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)  # 빨간색 텍스트 출력


    # 주차장의 남은 주차 공간 수 계산
    remaining_parking_spaces = total_parking_spaces - car_count

    # 차량 개수 및 남은 주차 공간 수 출력
    print(f"차량 개수: {car_count}")
    cv2.putText(frame, f"car: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    print(f"남은 주차 공간 수: {remaining_parking_spaces}")

    cv2.putText(frame, f"able: {remaining_parking_spaces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # 결과 이미지 표시 (선택 사항)
    cv2.imshow("Vehicle Detection", frame)

    # 'q' 키를 누르면 동영상 재생을 중지하고 창을 닫을 수 있습니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용한 자원 해제
video_capture.release()
cv2.destroyAllWindows()
