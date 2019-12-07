import cv2
import numpy as np
import time
from handGestureRecognition import handRecognition,resizeImage

startTime = time.time()
cap = cv2.VideoCapture(0)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame is the current image in the video
    if ret == False:
        break
    # frame = resizeImage(frame)
    try:
        output = handRecognition(frame)
        print("Number of fingers: ", output[1])
        cv2.imshow("output", output[0])
        cv2.imshow("Original", frame)
    except:
        pass
    i = i + 1


    if cv2.waitKey(5) == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

elapsedTime = time.time() - startTime
print("Iterations: ", i)
print("Elapsed Time: ", elapsedTime)
print("Iterations per second: ", i/elapsedTime)
