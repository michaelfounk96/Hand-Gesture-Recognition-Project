import cv2
import numpy as np
import math

def resizeImage(img):
    # this is a funtion to resize the image when needed for better results and to fit screen
    # need to change the variables in the resize function depending on the use
    x,y,z = img.shape
    resized = cv2.resize(img, (int(y/2),int(x/2)))
    return resized

def distanceBetweenPoints(p1, p2):
    # function for calculation of the distance between two points
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return distance

def handRecognition(img):
    # main function for the hand gesture Recognition
    # input is the image that we want to test the hand gesture

    # we start with creating a mask
    # This mask is the same size as the original image and is created where the pixels have a ratio of Red/Green in between 1.05 and 4
    # this is the ratio that was given in the paper and found to be working really well for recognizing skin color
    y, x, z = img.shape
    first_mask = np.zeros((x,y))
    first_mask = 1.05 < img[:,:,2] / img[:,:,1]
    first_mask = np.logical_and(first_mask, img[:,:,2] / img[:,:,1] < 4)
    first_mask = first_mask.astype("uint8")

    # then we find all the contours in the mask and remove all of them except for the biggest one
    # we assume that the biggest contour is the one that is the hand
    contours, hierarchy = cv2.findContours(first_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    premorph_mask = np.ones(img.shape[:2], dtype="uint8") * 255
    cv2.drawContours(premorph_mask, [c], -1, 0, -1)
    premorph_mask = cv2.bitwise_not(premorph_mask)

    # this part i added for some extra accuracy in the prediction
    # ive tried several variations of erosions and dialations. Some good ones are i = 4 and d = 2, i = 5 and d = 3, i = 2 and d = 1
    # i found that with 4 erosions and then 2 dialtion is the best over most of the sample images
    # most of the fingers and the hand seem to be full and as accurate as possible
    morph_size = 10
    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
    erode_mask = cv2.erode(premorph_mask.astype(np.uint8), morph_kern, iterations=4)
    mask = cv2.dilate(erode_mask, morph_kern, iterations=2)

    # next we calculate the center of gravity(cog)
    sumx = 0
    sumy = 0
    number = 0

    maskHits = np.argwhere(mask == 255)
    number = len(maskHits)
    sumy = sum(maskHits[:,:-1])
    sumx = sum(maskHits[:,-1])

    cogx = int(sumx/number)
    cogy = int(sumy/number)
    cog = (cogx, cogy)

    # we also calculate the farthest point from the center of gravity along the contour of the hand (c)
    farthest = 0
    point = (0,0)

    for item in c:
        temp = distanceBetweenPoints(cog, item[0])
        if farthest < temp:
            farthest = temp
            point = (item[0][0], item[0][1])

    # next i draw the circle that is created centered at the center of gravity with a radius of 0.7 times the farthest point on the contour
    # the o.7 was given in the paper and works pretty well
    distance = int(0.7*farthest)
    axes = (distance,distance)
    pointsOnCircle = cv2.ellipse2Poly(cog, axes, 0, 0, 360, 1)

    # we start with -1 fingers because our algorithm will count the wrist when doing this, so we have to subtract 1 to account for that
    # we run a for loop through each of the points in the circle and see if in the mask they are equal to 255
    # we also keep a running boolean that flips on and off when we go through one finger. ON: in the finger, OFF: left the finger
    # through this process we count the number of times around the circle the mask is 255 (hits a finger + 1 for wrist)
    # in the end we are left with the number of fingers that we counted and that is the final result of the algorithm
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

    # here i add circle, cog, farthest point and each point where a finger is hit to the mask
    # simply for display purposes
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(mask, cog, int(0.7*farthest), (0,0,255), 1)
    cv2.circle(mask, cog, 5, (0,255,255), 2)
    cv2.circle(mask, point, 5, (0,255,255), 2)
    cv2.line(mask, cog, point, (0,255,255), 1)

    for value in pointsThatStart:
        cv2.circle(mask, value, 5, (0,255,0), 2)

    # result of this function outputs the mask created (with circle, cog, fathest point, each point where finger hit) and the number of fingers that were counted
    return (mask, fingers)

# this is where i call the funtion and choose the image file
img = cv2.imread('images/hand3.jpg')
final = handRecognition(img)
differentSize = resizeImage(final[0])
img = resizeImage(img)
# rest of the code is just to display everything that was mentioned: original image, mask image (with circle, cog, fathest point, each point where finger hit)
# and the output number of fingers that are up
font = cv2.FONT_HERSHEY_SIMPLEX
output = np.zeros((200,200,3), np.uint8)
cv2.putText(output, str(final[1]),(100,100), font, 1,(255,255,255),2)
cv2.imshow("Number of Fingers",output)
cv2.imshow("Original Image", img)
cv2.imshow('After Hand Gesture Recognition', differentSize)
cv2.waitKey(0)
cv2.destroyAllWindows()
