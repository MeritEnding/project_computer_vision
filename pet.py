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
food_detected = False  # 음식이 감지되었는지 여부를 나타내는 변수

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # 임계값 설정
            if classes[class_id] == 'food':
                food_detected = True
                break  # 음식을 감지했으므로 루프 종료

# 음식이 감지되지 않으면 밥 주는 메시지 출력
if not food_detected:
    # 밥 주는 동작을 수행 (예: 밥그릇에 음식을 추가)
    cv2.putText(image, "음식이 비어있어요. 밥 주세요!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
# 결과 이미지 출력
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
