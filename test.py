import cv2
import numpy as np

img = cv2.imread('media/1_test.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('old_thresh', thresh)
thresh = cv2.merge((thresh, thresh, thresh))
cv2.imshow('thresh', thresh)
cnt = contours[1]
cv2.drawContours(thresh, [cnt], 0, (0, 255, 0), 2)
print(contours)

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(thresh,center,radius,(0,255,0),2)
cv2.imshow('res', thresh)


cv2.waitKey(0)