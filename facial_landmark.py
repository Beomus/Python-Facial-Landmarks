from imutils import face_utils
import imutils
import numpy as np
import dlib
import cv2

img_path = 'input.jpg'
shape_predictor = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# loading the image
image = cv2.imread(img_path)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect the face
rects = detector(gray, 1)

for x, rect in enumerate(rects):
	# finding the facial landmarks
	shape = predictor(gray, rect)
	# convert the coordinates to np array 
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to OpenCV bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	# draw the bounding box
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# put text in the bounding box
	cv2.putText(image, f'Face number {x}', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 2)

	# draw x, y coordinates for the facial landmarks
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Result", image)
cv2.imwrite('output.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
