import cv2
import numpy as np
import matplotlib.pyplot as plt

vid = cv2.VideoCapture('media/1.mp4')
freq = 0
fps = vid.get(cv2.CAP_PROP_FPS)
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

y = []
oscillation = 0
num_frames = int(total_frames - 200)
duration = num_frames/fps
y_mean = 0

while vid.isOpened():
    ok, img = vid.read()
    if ok:
        roi = img[100:400, 1400:1700, :]
        # cv2.imshow('roi', roi)

        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('original_gray', img_gray)

        threshold = np.max(img_gray)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0, 0)
        _, thresh = cv2.threshold(img_gray, threshold / 2, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh', thresh)

        kernel = np.ones((5, 5))
        res = cv2.erode(thresh, kernel, iterations=5)
        res = cv2.dilate(res, kernel, iterations=5)
        # cv2.imshow('final', res)

        m = cv2.moments(res)
        cx = int(m['m10'] / m['m00'])
        y_act = m['m01'] / m['m00']
        cy = int(y_act)
        res = cv2.merge((res, res, res))
        centroid = cv2.circle(res, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow('centroid', centroid)

        y.append(y_act)

        if len(y) >= num_frames:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            exit(0)
    else:
        break

vid.release()

for i in range(num_frames-2):
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
