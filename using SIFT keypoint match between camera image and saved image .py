import cv2
import numpy as np
inputimg=cv2.imread(r"D:\project\OMR\camera clicked\me.jpg")
img=cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)
sift=cv2.xfeatures2d.SIFT_create()
kp,desc=sift.detectAndCompute(img,None)
img=cv2.drawKeypoints(img,kp,img)

index_params=dict(algorithm=0, trees=5)
search_params=dict()
flann =cv2.FlannBasedMatcher(index_params,search_params)


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, rame = cam.read()
        frame=cv2.cvtColor(rame,cv2.COLOR_BGR2GRAY)
        kp_frame,desc_frame=sift.detectAndCompute(frame,None)
        frame=cv2.drawKeypoints(frame,kp_frame,frame)

        matches=flann.knnMatch(desc,desc_frame,k=2)
        

        good_points=[]
        for m,n in matches:
            if m.distance<0.7*n.distance:
                good_points.append(m)
        img3=cv2.drawMatches(img,kp,frame,kp_frame,good_points,frame)
        img4=cv2.resize(img3,(600,600))
        if mirror: 
            frame= cv2.flip(frame, 2)
        #cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        #cv2.imshow("me ",img)
        cv2.imshow("img4",img4)
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
