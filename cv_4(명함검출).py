import sys
import cv2
import numpy as np
import pytesseract



src = cv2.imread('namecard1.jpg')

if src is None:
    print("image load failed")
    sys.exit()

#영상의 이진화 
# 영상 픽셀값을 0 또는 1(255) 로 만드는 작업
# 관심영역 vs 비관심영역


def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))  # 칼럼0 -> 칼럼1 순으로 정렬한 인덱스를 반환
    pts = pts[idx]  # x좌표로 정렬

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

dw, dh = 720, 400
srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)
dstQuad = np.array([[0, 0], [0, dh], [dw, dh], [dw, 0]], np.float32)
dst = np.zeros((dh, dw), np.uint8)



src = cv2.resize(src,(0,0),fx=0.5,fy=0.5) # x축으로 0.5배 y축으로 0.5배

src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY) # gray스케일로 convert 

#_,src_bin = cv2.threshold(src_gray,130,255,cv2.THRESH_BINARY) # 이진화 임계값 130
th,src_bin = cv2.threshold(src_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 이진화 임계값을 OTSU로 결정 
print(th)

contous, _ = cv2.findContours(src_bin,cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE) # 외각선 RETR_EXTERNAL > 단순 외각 정보 RETR_TREE > 계층적 

#print(contous) # 외각선 좌표

for pts in contous:
    if cv2.contourArea(pts) < 1000: # 모든 외곽선 면적이 1000px 이하 자료는 넘김
        continue
    approx = cv2.approxPolyDP(pts,cv2.arcLength(pts,True)*0.02,True) # approxPolyDP : 외각선 근사화 마진 설정 중요  
    # 외각선 전체길이 * 0.2
    if len(approx) != 4: # 외각선 4개 인것만 현재 명함 사각형
        continue
    
  
    print(approx)
    
    srcQuad = reorderPts(approx.reshape(4, 2).astype(np.float32))

    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)

    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(dst_rgb, lang='Hangul+eng'))
    cv2.polylines(src,pts,True,(0,0,255)) # 외각선 표현하기 bgr 순서 : 현재 빨강색

cv2.imshow('src',src)
cv2.imshow('src_gray',src_gray)
cv2.imshow('src_bin',src_bin)
cv2.imshow('dst',dst)
#cv2.imshow('contous',contous)
cv2.waitKey()
cv2.destroyAllWindows
