import cv2
import numpy as np
import matplotlib.pyplot as plt

ref = cv2.imread('media/3.2_ref2.png')
ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
ref_hist = cv2.calcHist([ref_hsv], [0, 1], None, [180, 256], [0, 180, 0, 255])
# cv2.normalize(ref_hist, ref_hist, 0, 10, cv2.NORM_MINMAX)
plt.plot(ref_hist)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel3 = np.ones((5,5))

vid = cv2.VideoCapture('media/3.2.mp4')
freq = 0
fps = vid.get(cv2.CAP_PROP_FPS)
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

'''

 ON PYCHARM: HOLD CONTROL AND CLICK ON ANY FUNCTION TO OPEN ITS DOCUMENTATION!! 

'''
y = []
oscillation = 0
num_frames = int(total_frames - 2)
duration = num_frames / fps
y_mean = 0
while vid.isOpened():
    ret, img = vid.read()
    if ret:
        img = img[:, :500, :]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow('original', img)

        mask = cv2.calcBackProject([img_hsv], [0, 1], ref_hist, [0, 180, 0, 256], 1)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('mask', mask)

        res = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.erode(res, kernel2, iterations=3)
        res = cv2.dilate(res, kernel2, iterations=10)

        contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cv2.merge((res, res, res))
        cv2.drawContours(cnt, contours, -1, (0, 255, 0), 3)

        cv2.imshow('cnt', cnt)

        m = cv2.moments(res)
        if m['m00'] == 0:
            continue
        else:
            cx = int(m['m10'] / m['m00'])
            y_act = m['m01'] / m['m00']
            cy = int(y_act)
            res = cv2.merge((res, res, res))
            centroid = cv2.circle(res, (cx, cy), 2, (0, 0, 255), -1)
            cv2.imshow('centroid', centroid)

            y.append(y_act)

        if len(y) >= num_frames:
            break

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()

duration = len(y) / fps
for i in range(len(y)-1):
    y_mean = sum(y)/len(y)
    if (y[i]-y_mean)*(y[i+1]-y_mean) < 0:
        oscillation = oscillation + 0.5
freq = oscillation/duration
print("frequency= ", freq)

x = range(len(y))
x = np.divide(x, fps)
y = [i-y_mean for i in y]
plt.figure()
plt.plot(x, y)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Oscillation plot')
plt.show()
