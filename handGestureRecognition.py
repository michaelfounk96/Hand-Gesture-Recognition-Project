import cv2
import numpy as np
import math

def resizeImage(img):
    x,y,z = img.shape
    resized = cv2.resize(img, (int(y/2),int(x/2)))
    return resized

def distanceBetweenPoints(p1, p2):
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return distance

def handRecognition(img):
    # img = resizeImage(img)
    #something new
    y, x, z = img.shape
    first_mask = np.zeros((x,y))
    first_mask = 1.05 < img[:,:,2] / img[:,:,1]
    first_mask = np.logical_and(first_mask, img[:,:,2] / img[:,:,1] < 4)
    first_mask = first_mask.astype("uint8")

    contours, hierarchy = cv2.findContours(first_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    premorph_mask = np.ones(img.shape[:2], dtype="uint8") * 255
    cv2.drawContours(premorph_mask, [c], -1, 0, -1)
    premorph_mask = cv2.bitwise_not(premorph_mask)

    morph_size = 10
    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
    erode_mask = cv2.erode(premorph_mask.astype(np.uint8), morph_kern, iterations=2)
    mask = cv2.dilate(erode_mask, morph_kern, iterations=1)

    sumx = 0
    sumy = 0
    number = 0

    something = np.argwhere(mask == 255)
    number = len(something)
    sumy = sum(something[:,:-1])
    sumx = sum(something[:,-1])

    cogx = int(sumx/number)
    cogy = int(sumy/number)
    cog = (cogx, cogy)

    farthest = 0
    point = (0,0)

    for item in c:
        temp = distanceBetweenPoints(cog, item[0])
        if farthest < temp:
            farthest = temp
            point = (item[0][0], item[0][1])

    distance = int(0.7*farthest)
    axes = (distance,distance)
    pointsOnCircle = cv2.ellipse2Poly(cog, axes, 0, 0, 360, 1)

    fingers = -1
    bool = 0
    pointsThatStart = []
    for points in pointsOnCircle:
        currentPoint = (points[1], points[0])
        if currentPoint[0] >= y or currentPoint[1] >= x or currentPoint[0] <= 0 or currentPoint[1] <= 0:
            continue
        if mask[currentPoint] == 255:
            if bool == 0:
                bool = 1
                fingers = fingers + 1
                pointsThatStart.append((points[0], points[1]))
        else:
            bool = 0

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(mask, cog, int(0.7*farthest), (0,0,255), 1)
    cv2.circle(mask, cog, 5, (0,255,255), 2)
    cv2.circle(mask, point, 5, (0,255,255), 2)
    cv2.line(mask, cog, point, (0,255,255), 1)

    for value in pointsThatStart:
        cv2.circle(mask, value, 5, (0,255,0), 2)

    return (mask, fingers)

img = cv2.imread('images/hand3.jpg')
img = resizeImage(img)
final = handRecognition(img)
font = cv2.FONT_HERSHEY_SIMPLEX
output = np.zeros((200,200,3), np.uint8)
cv2.putText(output, str(final[1]),(100,100), font, 1,(255,255,255),2)
cv2.imshow("Number of Fingers",output)
cv2.imshow("Original Image", img)
cv2.imshow('After Hand Gesture Recognition', final[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
