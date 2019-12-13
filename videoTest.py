import cv2
import numpy as np
import time
from handGestureRecognition import handRecognition,resizeImage

# this is a video test. similar to the live video test except that the input is a premade video

startTime = time.time()
cap = cv2.VideoCapture("video/handVideo1.mp4")

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame is the current image in the video
    if ret == False:
        break
    # frame = resizeImage(frame)
    output = handRecognition(frame)
    i = i + 1
    print("Number of fingers: ", output[1])
    newOutput = resizeImage(output[0])
    frame = resizeImage(frame)
    cv2.imshow("output", newOutput)
    cv2.imshow("Original", frame)

    if cv2.waitKey(5) == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

elapsedTime = time.time() - startTime
print("Iterations: ", i)
print("Elapsed Time: ", elapsedTime)
print("Iterations per second: ", i/elapsedTime)
