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
image = cv2.imread("test3.jpeg")  # 차량을 포함한 이미지 파일 경로

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

        # 탐지 신뢰도 조건 (일반적으로 0.4 이상인 경우만 인식)
        if confidence > 0.4 and classes[class_id] == 'car':
            car_count += 1

            # 차량 주위에 네모 박스 그리기
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # 사각형 그리기
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # 초록색 네모 박스
            

# 주차장의 총 주차 공간 수
total_parking_spaces = 200  # 예시로 주차 공간 수를 100으로 가정

# 남은 주차 공간 수 계산
empty_parking_spaces = total_parking_spaces - car_count

# 주차 공간이 만석이면 주차 할 공간이 없습니다
if(empty_parking_spaces==0):
    print("주차 할 공간이 없습니다!")
    cv2.putText(image, f"알림: 주차 공간이 만석되었습니다.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 차량 개수 및 남은 주차 공간 수 출력
print(f"차량 개수: {car_count}")
cv2.putText(image, f"car_detection: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

print(f"남은 주차 공간 수: {empty_parking_spaces}")
cv2.putText(image, f"able: {empty_parking_spaces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

'''
================실시간 감시 시스템 적용 후 코드 구현=================
# 주차장의 총 주차 공간 수
total_parking_spaces = 200  # 예시로 주차 공간 수를 200으로 설정

# 주차장 상태: 주차된 차량의 목록을 저장할 리스트
parked_cars = []

# 주차 공간 수 업데이트 함수
def update_parking_spaces(enter_event, exit_event):
    global total_parking_spaces, parked_cars  # 전역 변수로 사용

    if enter_event:
        # 차량 입차 이벤트 발생
        if total_parking_spaces > 0:
            total_parking_spaces -= 1  # 주차 공간 수 감소
            parked_cars.append(len(parked_cars) + 1)  # 주차된 차량 추가
    elif exit_event:
        # 차량 출차 이벤트 발생
        if len(parked_cars) > 0:
            total_parking_spaces += 1  # 주차 공간 수 증가
            removed_car = parked_cars.pop()  # 주차된 차량 제거
            print(f"차량 {removed_car}가 출차했습니다.")

# 예시로 입출차 이벤트 발생 시 주차 공간 수 업데이트
enter_event = True  # 입차 이벤트 발생 여부
exit_event = False  # 출차 이벤트 발생 여부

# 주차 공간 수 업데이트 호출
update_parking_spaces(enter_event, exit_event)

# 주차장 상태 출력
print(f"현재 주차된 차량: {parked_cars}")
print(f"남은 주차 공간 수: {total_parking_spaces}")

'''


# 결과 이미지 저장 (선택 사항)
cv2.imwrite("output.jpg", image)

# 결과 이미지 표시 (선택 사항)
cv2.imshow("Vehicle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
