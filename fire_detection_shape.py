# 감지 애플리케이션을 개선하기 위해 색상 대신 모양을 감지하고 불의 위치와 크기를 저장하는 코드

import cv2
import numpy as np

# 영상을 읽어옵니다.
image = cv2.imread('your_image.jpg')

# 불을 감지할 색상 범위 설정 (빨간색)
lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])

# 색상 마스크 생성
red_mask = cv2.inRange(image, lower_red, upper_red)

# 빨간색 강조된 이미지에서 모양을 찾습니다.
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 불 모양을 저장할 리스트
fire_shapes = []

# 불 모양을 감지하고 저장합니다.
for contour in contours:
    # 불 모양을 감지하기 위한 최소한의 면적을 설정합니다.
    min_contour_area = 1000
    if cv2.contourArea(contour) > min_contour_area:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 불 모양을 감지한 영역을 이미지에 빨간색 사각형으로 그립니다.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 불 모양의 좌표와 크기를 저장합니다.
        fire_shapes.append({'x': x, 'y': y, 'width': w, 'height': h})

# 감지된 불 모양 정보를 출력합니다.
for i, fire_shape in enumerate(fire_shapes):
    print(f'불 {i + 1}: x={fire_shape["x"]}, y={fire_shape["y"]}, width={fire_shape["width"]}, height={fire_shape["height"]}')

# 결과 이미지를 표시합니다.
cv2.imshow('Fire Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
