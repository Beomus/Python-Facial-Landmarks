from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import cv2
import dlib
import time


def eye_ar(eye):
    # calculate the euclidean distances between the two sets of vertical eye landmarks x, y coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # calculate the euclidean distance between horizontal eye landmark x, y coordinate 
    C = dist.euclidean(eye[0], eye[3])

    # calcuate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# the aspect ratio to count one blink
EYE_AR_THRESH = 0.2
# the number of frames where ar need to be below the threshold to count one blink
EYE_AR_CONSEC_FRAMES = 3

# frame counter
COUNTER = 0 
# total blinks
TOTAL = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

print('[INFO] Initiating detector...')
shape_predictor = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

print('[INFO] Initiating camera...')
vs = VideoStream(scr=0).start()
time.sleep(2)

print('[INFO] Detecting...')
run = True
while run:
    frame = vs.read()

    # resize if you're not getting consistent FPS
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the coordinates for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # calculating the aspect ratio for each eye
        leftEAR = eye_ar(leftEye)
        rightEAR = eye_ar(rightEye)

        # average the aspect ratio between two eyes
        ear = (leftEAR + rightEAR) / 2.0

        # calculate the convex hull for each eye then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            # if the EAR is lower than the threshold, place a counter
            COUNTER += 1
        else:
            # if the eyes were closed for more than 3 frames
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # increase the count of blinks
                TOTAL += 1
            # reset the frame counter
            COUNTER = 0

        # putting the info on the screen
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear}", (300, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    # showing the frames
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        run = False
        break
vs.stop()
cv2.destroyAllWindows()
