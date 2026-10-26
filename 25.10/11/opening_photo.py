# imports - how the refrence the used libaries
import cv2 as cv
import sys

#specify which image you want to use:
img = cv.imread(cv.samples.findFile("25.10/starry_night.png"))

#check if image is loaded
if img is None:
    sys.exit("Could not find image")

# showing image:
cv.imshow("Hallo", img) #arguments: windows name, image to display

#How to close window when user presses any key
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("day-13.20/starry_night.png", img)