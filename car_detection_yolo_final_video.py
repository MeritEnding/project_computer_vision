import cv2
import numpy as np

# YOLO 설정 파일과 가중치 파일 경로
yolo_cfg = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'

# YOLO 클래스 이름 파일 경로
yolo_classes = 'coco.names'

# YOLO 클래스 이름 로드
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# YOLO 모델 로드
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# 웹캠 또는 비디오 파일 열기
video_capture = cv2.VideoCapture('test2.mp4')  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 지정하세요.

# 총 주차 가능한 자리 수 설정
total_parking_spots = 50  # 예시로 50개의 주차 자리가 있다고 가정

while True:
    ret, frame = video_capture.read()
    
    # YOLO 입력 이미지의 크기
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # YOLO 모델에 입력 이미지 설정
    net.setInput(blob)
    
    # YOLO 출력 계산
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)
    
    # 탐지된 객체 정보를 저장할 리스트 초기화
    class_ids = []
    confidences = []
    boxes = []
    
    # 탐지된 객체 정보 추출
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # 탐지 신뢰도 조건 설정
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                
                # 바운딩 박스 좌표 계산
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])
    
    # 겹치는 박스를 제거하는 비최대 억제(NMS) 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # 차량 개수 세기
    car_count = len(indices)
    
    # 주차 가능한 자리 수 계산
    available_parking_spots = total_parking_spots - car_count
    

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # 탐지된 객체를 사각형으로 그리고 클래스 이름과 신뢰도를 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            text = f'{label}: {confidence:.2f}'
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # 결과 화면에 표시
    cv2.putText(frame, f'car: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'car able: {available_parking_spots}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('car detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 해제
video_capture.release()
cv2.destroyAllWindows()
