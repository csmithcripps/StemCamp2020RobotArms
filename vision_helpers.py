import cv2
import numpy as np
import math

class VisionHelpers():
	"""docstring for vision_helpers"""
	def __init__(self):
		# do stuff
		print("vision helpers initialised")

	def get_suitable_image(self, cap):
		key = -1
		while key != ord('a'):	
			ret, im = cap.read()
			if not ret:
				return [ret, _]
			cv2.imshow("camera_image", im)
			key = cv2.waitKey(1)
			if key == ord('q') or key == 27:
				return [False, _]
		return [ret, im]

	def get_blue_circles(self, image, min_, max_):
		mask = cv2.inRange(image, np.array(min_), np.array(max_))
		im_masked=cv2.bitwise_and(image, image, mask=mask)
		im_masked=cv2.bitwise_not(im_masked)
		params = cv2.SimpleBlobDetector_Params()
		params.filterByInertia = False
		params.filterByConvexity = False
		params.filterByArea = True
		params.minArea = 100
		params.maxArea = 10000
		detector = cv2.SimpleBlobDetector_create(params)
		keypoints = detector.detect(im_masked)
		return keypoints

	def sort_keypoints(self, keypoints):
		keypoints_list = []
		for keypoint in keypoints:
			dict_item = {"x" : round(keypoint.pt[0]), "y" : round(keypoint.pt[1])}
			keypoints_list.append(dict_item)
		keypoints_array = []
		keypoints_list.sort(key=lambda item: (item['x'], item['y']))
		for item in keypoints_list:
			print(item)
			keypoints_array.append([item['y'], item['x']])
		return np.array(keypoints_array, np.float32)

	def get_shapes(self, mask, minArea, maxArea, minCircularity, maxCircularity): # returns all contours matching the input params
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours_area = []
		shapes = []
		# calculate area and filter into new array
		for con in contours:
		    area = cv2.contourArea(con)
		    if minArea < area < maxArea:
		        contours_area.append(con)
		# check if contour is of circular shape
		for con in contours_area:
		    perimeter = cv2.arcLength(con, True)
		    area = cv2.contourArea(con)
		    if perimeter == 0:
		        break
		    circularity = 4*math.pi*(area/(perimeter*perimeter))
		    print(circularity)
		    if minCircularity < circularity < maxCircularity:
		    	shapes.append(con)
		return shapes

	def perspective_transform(self, contours_list, h):
		real_world_coordinates = []
		for cont in contours_list:
			x_c,y_c,width,height = cv2.boundingRect(cont)
			uc = x_c + width/2
			uv = y_c + height/2
			a = np.array([[uc, uv]], dtype='float32')
			a = np.array([a])
			print(a)
			points = cv2.perspectiveTransform(a, h)
			real_world_coordinates.append(points)
		return real_world_coordinates


	def colour_thresholder(self, img, window_name):

		def nothing(x):
		  pass

		cv2.namedWindow(window_name)

		cv2.createTrackbar("BMax", window_name,0,255,nothing)
		cv2.createTrackbar("BMin", window_name,0,255,nothing)
		cv2.createTrackbar("GMax", window_name,0,255,nothing)
		cv2.createTrackbar("GMin", window_name,0,255,nothing)
		cv2.createTrackbar("RMax", window_name,0,255,nothing)
		cv2.createTrackbar("RMin", window_name,0,255,nothing)

		cv2.setTrackbarPos("BMax", window_name, 255)
		cv2.setTrackbarPos("BMin", window_name, 0)
		cv2.setTrackbarPos("GMax", window_name, 255)
		cv2.setTrackbarPos("GMin", window_name, 0)
		cv2.setTrackbarPos("RMax", window_name, 255)
		cv2.setTrackbarPos("RMin", window_name, 0)

		while(1):
		   bmax=cv2.getTrackbarPos("BMax", window_name)
		   bmin=cv2.getTrackbarPos("BMin", window_name)
		   gmax=cv2.getTrackbarPos("GMax", window_name)
		   gmin=cv2.getTrackbarPos("GMin", window_name)
		   rmax=cv2.getTrackbarPos("RMax", window_name)
		   rmin=cv2.getTrackbarPos("RMin", window_name)

		   min_ = np.array([bmin,gmin,rmin])
		   max_ = np.array([bmax,gmax,rmax])

		   mask = cv2.inRange(img, min_, max_)

		   thresholded_img = cv2.bitwise_and(img, img, mask= mask)

		   cv2.imshow("thresholded",thresholded_img)

		   k = cv2.waitKey(1) & 0xFF

		   # exit if q or esc are pressed
		   if (k == ord('q') or k == 27):
		     return
		   if k == ord('a'):
		   	 break

		cv2.destroyWindow(window_name)
		cv2.destroyWindow("thresholded")
		return [min_, max_]

		
		