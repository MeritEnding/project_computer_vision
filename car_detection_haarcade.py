import cv2
import tkinter as tk
from tkinter import filedialog

# 전역 변수 초기화
video_src = None
total_parking_spaces = 100  # 주차장의 총 주차 공간 수

#추가
# 함수: 비디오 파일 선택
def choose_video_file():
    global video_src
    video_src = filedialog.askopenfilename()

# 함수: 시작 버튼 클릭
def start_detection():
    if video_src is not None:
        cap = cv2.VideoCapture(video_src)
        car_cascade = cv2.CascadeClassifier('cars.xml')
        car_count = 0

        while True:
            ret, img = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cars = car_cascade.detectMultiScale(gray, 1.1, 1)
            car_count = len(cars)  # 감지된 자동차의 수를 센다
            able_place = total_parking_spaces - car_count  # 빈 주차 공간 계산

            for (x, y, w, h) in cars:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, 'car', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
            cv2.putText(img, f'car: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(img, f'able: {able_place}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # 주차 상태 표시
            for i in range(total_parking_spaces):
                color = (0, 255, 0) if i < able_place else (255, 0, 0)
                cv2.rectangle(img, (i * 5, 80), ((i + 1) * 5, 100), color, -1)

            cv2.imshow('Video', img)

            if cv2.waitKey(33) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("주차장 선택")

# GUI 창 생성
root = tk.Tk()
root.title("실시간 주차 현황")
root.geometry("400x300")  # 창 크기 설정

# 제목 레이블
title_label = tk.Label(root, text="실시간 주차 현황", font=("Helvetica", 20))
title_label.pack(pady=10)

# 비디오 파일 선택 프레임
frame = tk.Frame(root)
frame.pack()

select_button = tk.Button(frame, text="주차장 선택", command=choose_video_file)
select_button.grid(row=0, column=0, padx=10)

start_button = tk.Button(frame, text="주차 현황보기", command=start_detection)
start_button.grid(row=0, column=1, padx=10)

# 종료 버튼
exit_button = tk.Button(root, text="종료", command=root.destroy)
exit_button.pack(pady=20)

root.mainloop()
