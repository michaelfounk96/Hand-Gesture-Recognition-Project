import cv2
import numpy as np
import os

from handGestureRecognition import handRecognition,resizeImage
# this is a batch file test that i run through all of the pictures i have in the images folder
# the completed ones that i have are sorted into the ones that fail and the ones that succeed

for filename in os.listdir('images'):
    if filename != "Failed" and filename != "Completed":
        img = cv2.imread('images/' + filename)
        output = handRecognition(img)
        newFileName = filename.split(".", 1)[0]
        print(newFileName)
        newFileName = 'images/Completed/' + newFileName + 'Completed.jpg'
        print(newFileName)
        cv2.imwrite(newFileName, output[0])
        print(filename + " has: " + str(output[1]) + " fingers")
