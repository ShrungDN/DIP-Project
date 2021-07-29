import cv2
import numpy as np 
import matplotlib.pyplot as plt

vid = cv2.VideoCapture('media/3.2.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

object_detector = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=20)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

y_list = []
oscillation = 0
num_frames = int(total_frames - 2)
y_mean = 0
while vid.isOpened():
    ret, img = vid.read()
    if ret:
        img = img[150:300, 230:360]
        mask = object_detector.apply(img)
        ret, mask = cv2.threshold(mask, 90, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150:
                # cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                yc = y + h/2
                y_list.append(yc)

        if len(y_list) >= num_frames:
            break

        cv2.imshow('mask', mask)
        cv2.imshow('original', img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else: 
        break

vid.release()
cv2.destroyAllWindows()

y_list = [y_list[i] for i in range(len(y_list)) if i > 6]
duration = len(y_list) / fps
for i in range(len(y_list)-1):
    y_mean = sum(y_list)/len(y_list)
    if (y_list[i]-y_mean)*(y_list[i+1]-y_mean) < 0:
        oscillation = oscillation + 0.5
freq = oscillation/duration
print("frequency= ", freq)

x = range(len(y_list))
x = np.divide(x, fps)
y_list = [i-y_mean for i in y_list]
plt.figure()
plt.plot(x, y_list)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Oscillation plot')
plt.show()
