import cv2
import sys 

cap = cv2.VideoCapture("vtest.avi") # 0번카메라 오픈 0부터 시작

if not cap.isOpened(): # 카메라가 없을경우
    print("camera open failed") # fail 메세지
    sys.exit()

while True:

    ret,frame = cap.read() # 정상적인경우 ret true 비정상인경우 false , frame > none

    if not ret: 
        break;
    
    edge = cv2.Canny(frame,50,150) #가장자리만 표현
    
    cv2.imshow("frame",frame)
    cv2.imshow("edge",edge)
    if cv2.waitKey(1) == 27: #27일경우 esc 카메라프레임 1000/x = 30 프레임
        break
    
cap.release()
cv2.destroyAllWindows()



