import matplotlib.pyplot as plt
import cv2

# 컬려 영상 출력


imgBGR = cv2.imread('cat.bmp') # cv2 의 기본은 BGR
imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB) # pyplot 은 RGB이므로 컨버팅한다
plt.axis("off")
plt.imshow(imgRGB)
plt.show()

# 그레이스케일 영상 출력

imgGray = cv2.imread("cat.bmp",cv2.IMREAD_GRAYSCALE)
plt.axis('off')
plt.imshow(imgGray, cmap="gray") # cmap gray 지정 
plt.show()

#두개 함께 출력
plt.subplot(121),plt.axis('off'),plt.imshow(imgRGB)
plt.subplot(122),plt.axis('off'),plt.imshow(imgGray,cmap='gray')
plt.show()
