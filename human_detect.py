import cv2
import numpy as np

# OpenCV의 Haar Cascade Classifier를 사용하여 얼굴을 감지하는 함수
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# 경고 함수
def send_warning():
    print("인구가 너무 많습니다! 경고를 보냅니다.")

# 메인 함수
def main():
    # 카메라 캡처 객체 생성
    cap = cv2.VideoCapture(0)

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        # 얼굴 감지
        faces = detect_faces(frame)

        # 감지된 얼굴 수
        num_faces = len(faces)

        # 인구 수에 따라 경고 보내기
        if num_faces > 10:
            send_warning()

        # 화면에 얼굴 표시
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 화면에 인구 수 표시
        cv2.putText(frame, f'인구 수: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 화면에 출력
        cv2.imshow('Population Counter', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 카메라 객체 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
