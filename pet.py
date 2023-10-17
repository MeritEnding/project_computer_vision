import cv2
import numpy as np

# YOLO 설정 파일과 가중치 파일 경로
yolo_config = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'

# YOLO 클래스 이름 파일 경로
class_names = 'coco.names'

# YOLO 모델 불러오기
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# 클래스 이름 로드
with open(class_names, 'r') as f:
    classes = f.read().strip().split('\n')

# 이미지 불러오기
image = cv2.imread('food_image.jpg')

# 이미지 크기 조정 (YOLO는 416x416 크기의 이미지를 사용)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# YOLO 모델에 이미지 전달
net.setInput(blob)

# 객체 감지 수행
outs = net.forward(net.getUnconnectedOutLayersNames())

# 감지된 객체를 리스트에 저장
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # 임계값 설정
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])

            # 바운딩 박스 좌표 계산
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, width, height])

# Non-Maximum Suppression (중복 박스 제거)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 감지된 객체를 이미지에 그리기
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 출력
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
