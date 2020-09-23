import cv2
import numpy as np
import math

import vision_helpers

vh = vision_helpers.VisionHelpers()

cap = cv2.VideoCapture(0)

ret, im = vh.get_suitable_image(cap)

if not ret:
	print("No image obtained. Exiting...")
	exit()

status = False

# Get blue thresholds
min_blue, max_blue = vh.colour_thresholder(im.copy(), "Blue Thresholds")
print(min_blue)
print(max_blue)
# min_blue = np.array([100, 0, 0])
# max_blue = np.array([255, 80,80])
keypoints = vh.get_blue_circles(im.copy(), min_blue, max_blue)
print("Number of keypoints: " + str(len(keypoints)))
im_with_keypoints = cv2.drawKeypoints(im.copy(), 
	keypoints, np.array([]), (0,0,255), 
	cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("keypoints", im_with_keypoints)
cv2.waitKey(0)

if len(keypoints) != 9:
	print("Incorrect number of keypoints detected")
else:
	keypoints_array = vh.sort_keypoints(keypoints)
	realworldpoints_array = np.array([[20,20],[200,20],[380,20],
		[20,200],[200,200],[380,200], 
		[20,380],[200,380],[380,380]], np.float32)
	h,_ = cv2.findHomography(keypoints_array, realworldpoints_array)
	status = True



# --------- GET THE SHAPES ---------- #
im2 = im.copy()
min_red, max_red = vh.colour_thresholder(im2, "Red Thresholds")
mask_red = cv2.inRange(im2, np.array(min_red),np.array(max_red))

red_circles = vh.get_shapes(mask_red, 10, 10000, 0.8, 1.5)

cv2.drawContours(im, red_circles, -1, (0,255,0), 2)
cv2.imshow("Image with contours", im)
cv2.waitKey(0)

if status:
	red_circles_realworld = vh.perspective_transform(red_circles, h)
	print(red_squares_realworld)

cv2.destroyAllWindows()