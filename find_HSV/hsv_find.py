import cv2
import numpy as np

def nothing(x):
    pass

# 원하는 너비와 높이 설정
width = 640
height = 480

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 윈도우 생성
cv2.namedWindow('Detection')

# 트랙바 생성
cv2.createTrackbar('Low H', 'Detection', 0, 179, nothing)
cv2.createTrackbar('High H', 'Detection', 0, 179, nothing)
cv2.createTrackbar('Low S', 'Detection', 0, 255, nothing)
cv2.createTrackbar('High S', 'Detection', 0, 255, nothing)
cv2.createTrackbar('Low V', 'Detection', 0, 255, nothing)
cv2.createTrackbar('High V', 'Detection', 0, 255, nothing)

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 트랙바에서 설정된 값 가져오기
    low_h = cv2.getTrackbarPos('Low H', 'Detection')
    high_h = cv2.getTrackbarPos('High H', 'Detection')
    low_s = cv2.getTrackbarPos('Low S', 'Detection')
    high_s = cv2.getTrackbarPos('High S', 'Detection')
    low_v = cv2.getTrackbarPos('Low V', 'Detection')
    high_v = cv2.getTrackbarPos('High V', 'Detection')
    
    # 색상 범위 지정
    lower_orange = np.array([low_h, low_s, low_v])
    upper_orange = np.array([high_h, high_s, high_v])
    
    # 마스크 생성
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 오브젝트 감지
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 컨투어 그리기
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 임계값을 조절하여 작은 노이즈를 제거
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    
    # 결과 출력
    cv2.imshow('Detection', frame)
    
    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 종료
cap.release()
cv2.destroyAllWindows()
