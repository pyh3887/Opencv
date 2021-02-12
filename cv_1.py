import cv2
import sys
img = cv2.imread("cat.bmp") # 이미지 로딩 플래그 디폴트
#img = cv2.imread("cat.bmp", cv2.IMREAD_GRAYSCALE) #플래그 사용 greyscale > 색이없음

if img is None:
    print("Image load failed!")
    sys.exit()

print(type(img))
print(img.shape)
print(img.dtype)

cv2.namedWindow("image") # cv2에서 창만들기 
cv2.imshow("image",img) # 이미지 창에 이미지 출력
cv2.waitKey() # 키보드 입력시간/창띄우는 시간 지정 (숫자를 안적을시 계속 실행)
#cv2.destroyAllWindows 이미지 창 닫기




