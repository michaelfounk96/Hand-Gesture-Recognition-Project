import cv2
import numpy as np
import time
from handGestureRecognition import handRecognition,resizeImage

# this i a live video test. I run a while loop that runs my function for each frame that it receives
# i also check the start and end times to calculate the number of iterations that is does per second

startTime = time.time()
cap = cv2.VideoCapture(0)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
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
