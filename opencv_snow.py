import cv2,dlib,sys
import numpy as np 

scaler = 0.3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 얼굴인식 모델

cap = cv2.VideoCapture("video.mp4")

#라이언 사진 
overlay = cv2.imread('ryan_transparent.png', cv2.IMREAD_UNCHANGED)


# 영상에 사진 오버레이 function

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img


face_roi = []
face_sizes = []

while True:
   # 비디오 읽기
    ret, img = cap.read()
    if not ret:
        break

    # 프레임 크기조정
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # 얼굴찾기 
    if len(face_roi) == 0:
        faces = detector(img, 1)
       
    else:
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        
        #cv2.imshow('roi', roi_img)
        faces = detector(roi_img)

    # no faces
    if len(faces) == 0:
        print('no faces!')
  # find facial landmarks
    for face in faces:
        if len(face_roi) == 0:
            dlib_shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else:
            dlib_shape = predictor(roi_img, face)
            shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])


    dlib_shape = predictor(img,face) # 이미지의 얼굴영역의 특징점찾기
    
    shape_2d = np.array([[p.x,p.y] for p in dlib_shape.parts()])

    # 얼굴 특징의 최댓값 최솟값 위치 찾기
    top_left = np.min(shape_2d,axis=0)
    bottom_right = np.max(shape_2d,axis=0)
    face_size = max(bottom_right-top_left)

    center_x, center_y = np.mean(shape_2d,axis=0).astype(np.int)
    result = overlay_transparent(ori,overlay,center_x-10,center_y-30,overlay_size=(face_size,face_size))
    # 이미지안의 얼굴에 표시하기
    img = cv2.rectangle(img,pt1=(face.left(),face.top()),pt2=(face.right(),face.bottom()),color=(255,255,255),
        thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        cv2.circle(img,center = tuple(s),radius=1,color=(255,255,255),
        thickness=2, lineType=cv2.LINE_AA)

    # 얼굴의 특징 좌표 최소값과 최대값을 구함
    cv2.circle(img,center=tuple(top_left), radius=1,color=(0,255,0),
        thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img,center=tuple(bottom_right), radius=1,color=(0,255,0),
        thickness=2, lineType=cv2.LINE_AA)
    
    cv2.circle(img,center=tuple((center_x,center_y)), radius=1,color=(255,0,0),
        thickness=2, lineType=cv2.LINE_AA)


    #cv2.imshow('img',img)
    cv2.imshow('result',result)
    if cv2.waitKey(1) == 27: #27일경우 esc 카메라프레임 1000/x = 30 프레임
        break