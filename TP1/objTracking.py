import KallmanFilter as kf
import Detector as dt
import cv2
import numpy as np

kalman_filter = kf.KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

video_capture = cv2.VideoCapture("randomball.avi")

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

centers = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    center = dt.detect(frame)
    centers.append(center)

    if center:
        (x, y) = kalman_filter.predict()
        cv2.rectangle(frame, (int(x - 13), int(y - 13)), (int(x + 13), int(y + 13)), (255, 0, 0), 2)

        (x_est, y_est) = kalman_filter.update(center[0])
        cv2.rectangle(frame, (int(x_est - 13), int(y_est - 13)), (int(x_est + 13), int(y_est + 13)), (0, 0, 255), 2)

    for c in range(1, len(centers)):
        cv2.line(frame, (int(centers[c - 1][0][0][0]), int(centers[c - 1][0][1][0])), (int(centers[c][0][0][0]), int(centers[c][0][1][0])), (128, 0, 128), 2)

    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('frame',1) == -1:
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()