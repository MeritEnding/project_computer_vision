import cv2
import tkinter as tk
from tkinter import filedialog

# 전역 변수 초기화
image_src ='test3.jpeg'  # 이미지 파일 경로 변수



if image_src is not None:
    # 이미지 파일 불러오기
    img = cv2.imread(image_src)
    
    # Haar Cascade 분류기 로드 (차량 인식)
    car_cascade = cv2.CascadeClassifier('cars.xml')
    
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 차량 인식 수행
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))
    car_count =len(cars)
    able_place=100-car_count
    # 차량 인식 결과 표시
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.putText(img, f'Car:{car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f'able:{able_place}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if able_place ==0:
        cv2.putText(img, '자리 만석', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 결과 이미지 표시
    cv2.imshow('Car Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("이미지를 선택해 주세요.")



