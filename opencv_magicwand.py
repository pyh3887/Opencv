#!/usr/bin/env python
# coding: utf-8

# In[17]:



import cv2
import numpy as np



# 마우스 이벤트 콜백함수 정의

def mouse_callback(event, x, y, flags, param): 
    global points
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 클릭시          
       
         contour(x,y) # 외곽선 그리기 시작 (클릭한 좌표값 넘김)

        
def contour(x,y): # 외각선 그리기 
    
    x = x # 좌표 설정 
    y = y
    
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #  GRAYSCALE        
    
    th , binary_img = cv2.threshold(imgray,133,255,cv2.THRESH_BINARY) # 임계값 120 으로 지정 이진화작업    
    #th , binary_img = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
    
    
    contours,hr = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # 이진화된 이미지를 계층적 구조로 외곽선좌표를 찾음

    #print(len(contours))
    
    include_cont_index = [] 
    cos_result = [] 
    
    
    # 클릭한 좌표가 컨투어 안에 포함되는 컨투어의 인덱스 찾기
    
    for idx,cc in enumerate(contours,0):  # 각 컨투어의 왼쪽,오른쪽,위,아래 max,min 값을 찾은뒤 누른 좌표값이 구역내에 포함되는 컨투어 선택
        
        leftmost = tuple(cc[cc[:,:,0].argmin()][0])  # 컨투어의 좌측최소값
        rightmost = tuple(cc[cc[:,:,0].argmax()][0]) # 오른쪽 최대값
        topmost = tuple(cc[cc[:,:,1].argmin()][0]) # 위쪽 최소값
        botmost = tuple(cc[cc[:,:,1].argmax()][0]) # 아래쪽 최대값
    
        #print(leftmost,rightmost,topmost,botmost) 
        
        if leftmost[0]<= x and x<=rightmost[0] and y>topmost[1]and y<botmost[1] :
            
            include_cont_index.append(idx)            
    
    
    # 걸러낸 컨투어에서 좌표에서 가장 가까운 컨투어 찾기    
    
    for a in include_cont_index:
        vv = contours[a]
        leftmost = tuple(vv[vv[:,:,0].argmin()][0])  # 컨투어의 좌측최소값
        rightmost = tuple(vv[vv[:,:,0].argmax()][0]) # 오른쪽 최대값
        topmost = tuple(vv[vv[:,:,1].argmin()][0]) # 위쪽 최소값
        botmost = tuple(vv[vv[:,:,1].argmax()][0]) # 아래쪽 최대값
        
        result = abs(leftmost[0]-x)+ abs(rightmost[0]-x) + abs(topmost[1]-y) + abs(botmost[1]-y) # 컨투어의 좌표값과 클릭한 좌표의 차를구함
        
        cos_result.append(result) # cos_result 에 차이값 append        
    
    min_index = cos_result.index(min(cos_result)) #  최소값의 리스트에서 최소값의 인덱스를 뽑음
    contour_index = include_cont_index[min_index] #  최소값 인덱스를 걸러진 컨투어의 인덱스와 매칭 
    
    
    cv2.drawContours(img,contours[contour_index],-1,(0,0,255),3) # 좌표값과 거리차이가 가장적은 컨투어를 그린다.
    
    cv2.imshow('image', img) 
    cv2.imshow('thresh',binary_img) # 이진화 이미지 비교 
 
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()



img = cv2.imread("my.jpg")
#img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)    
cv2.namedWindow('image')  #마우스 이벤트 영역 윈도우 생성

print(img.shape) 
cv2.setMouseCallback('image', mouse_callback) # mouse callback


while(True):

        cv2.imshow('image', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:    # ESC 키 눌러졌을 경우 종료
            print("ESC 키 눌러짐")
            break

cv2.destroyAllWindows()



# In[ ]:




