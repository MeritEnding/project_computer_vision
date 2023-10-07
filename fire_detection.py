import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import requests

# 기상 데이터를 가져오는 함수
def get_weather_data():
    api_key = 'your_api_key'
    city = 'your_city'
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    
    response = requests.get(url)
    data = response.json()
    
    # 필요한 기상 데이터를 추출합니다 (예: 온도, 상대 습도 등)
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    
    return temperature, humidity

# 화재 위험 예측 모델
def predict_fire_risk(temperature, humidity):
    # 예측 모델을 사용하여 화재 위험을 계산합니다.
    # 예: 낮은 온도와 높은 습도는 낮은 화재 위험을 나타낼 수 있습니다.
    fire_risk = temperature * humidity / 1000
    
    return fire_risk



# 함수를 사용하여 불 감지 알림 보내기
def send_email(subject, message, image_filename):
    # 이메일 설정
    sender_email = 'your_email@gmail.com'  # 발신자 이메일 주소
    sender_password = 'your_password'  # 발신자 이메일 비밀번호
    receiver_email = 'receiver_email@gmail.com'  # 수신자 이메일 주소

    # 이메일 서버 설정 (Google Gmail 예시)
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587  # Gmail의 SMTP 포트

    # 이메일 메시지 생성
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # 이미지 첨부
    with open(image_filename, 'rb') as image_file:
        image_data = image_file.read()
        image = MIMEImage(image_data, name='fire_image.jpg')
        msg.attach(image)

    # 이메일 서버 연결 및 메시지 전송
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("이메일 알림을 보냈습니다.")
    except Exception as e:
        print(f"이메일 알림을 보내는 동안 오류가 발생했습니다: {str(e)}")

# 영상을 읽어옵니다.
if __name__ == '__main__':
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

    # 결과 이미지 저장
    cv2.imwrite("output.jpg", image)
    temperature, humidity = get_weather_data()
    fire_risk = predict_fire_risk(temperature, humidity)

    if len(fire_shapes) > 0 or fire_risk > 0.5:
        subject = "불 감지 및 화재 위험 알림"
        message = f"불이 감지되었습니다. 화재 위험 지수: {fire_risk}"
        send_email(subject, message, "output.jpg")
    else:
        print("화재 감지 및 화재 위험 없음")
        
    # 불이 감지되면 이메일로 알림을 보냅니다.
    if len(fire_shapes) > 0:
        subject = "불 감지 알림"
        message = "불이 감지되었습니다."
        send_email(subject, message, "output.jpg")

    # 결과 이미지 표시 (선택 사항)
    cv2.imshow('Fire Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
