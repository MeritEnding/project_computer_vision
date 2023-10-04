import cv2
import numpy as np

# 영상을 읽어옵니다.
image = cv2.imread('your_image.jpg')

# 빨간색을 강조하는 마스크 생성
lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])
red_mask = cv2.inRange(image, lower_red, upper_red)

# 빨간색 강조된 이미지에서 불이 있는 영역을 찾습니다.
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 불이 있는 부분에 빨간색 네모를 그려줍니다.
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 결과 이미지를 표시합니다.
cv2.imshow('Fire Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
