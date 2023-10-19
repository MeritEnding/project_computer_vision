import cv2
import numpy as np
import streamlit as st

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

# 음식 추가 동작을 수행하는 함수
def add_food_to_bowl(frame):
    # 여기에 음식 추가 동작을 구현
    # 이 예제에서는 음식 아이콘을 그립니다.
    bowl_center_x, bowl_center_y = frame.shape[1] // 2, frame.shape[0] // 2
    food_icon = cv2.imread('food_icon.png')  # 음식 아이콘 이미지 파일을 사용
    food_icon = cv2.resize(food_icon, (100, 100))  # 아이콘 크기 조정

    # 음식 아이콘을 밥그릇 중심에 그립니다.
    x = bowl_center_x - food_icon.shape[1] // 2
    y = bowl_center_y - food_icon.shape[0] // 2
    frame[y:y+food_icon.shape[0], x:x+food_icon.shape[1]] = food_icon

# streamlit 앱 시작
st.title('음식 감지 및 추가 앱')

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 웹캠을 사용하려면 0, 다른 비디오 파일을 사용하려면 파일 경로를 지정

# 초기 음식 수 및 임계값 설정
initial_food_count = 0
threshold_food_count = 5  # 음식 추가를 위한 임계값

if st.button('시작'):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 크기 조정 (YOLO는 416x416 크기의 이미지를 사용)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # YOLO 모델에 이미지 전달
        net.setInput(blob)

        # 객체 감지 수행
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # 감지된 음식 수 계산
        food_count = 0

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'food':
                    food_count += 1

        # 음식 수가 임계값 미만일 때 음식 추가
        if food_count < threshold_food_count:
            add_food_to_bowl(frame)

        # 음식 수를 화면에 표시
        st.header('음식 감지 및 추가 결과')
        st.image(frame, caption=f"음식 수: {food_count}", use_column_width=True)

        # 'q' 키를 누를 때 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠 또는 비디오 스트림 종료
cap.release()
