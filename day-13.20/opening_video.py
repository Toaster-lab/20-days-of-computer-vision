import time
import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture('day-13.20/sample-5s-720p.mp4')

prev_time = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray_filter = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculating frame:
    now = time.perf_counter()
    fps = 1 / (now - prev_time)
    prev_time = now

    #showing fps
    cv.putText(gray_filter, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    cv.imshow('frame', gray_filter)
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()
