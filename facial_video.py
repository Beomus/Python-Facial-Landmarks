from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import time


print('[INFO] Initiating detector...')
shape_predictor = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

print('[INFO] Initiating camera...')
vs = VideoStream(scr=0).start()
time.sleep(2)

print('[INFO] Detecting...')
run = True
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # we're using 0 here instead of 1 since we are doing live detection
	# using 1 would not cause any problems but there are some inaccuracies
	# while the face is turned sideway.
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        run = False
        break

cv2.destroyAllWindows()