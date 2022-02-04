from scipy.spatial import distance as dist
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import time
import dlib
import cv2
import queue
import library
import RPi.GPIO as GPIO
from time import sleep

class driver_monitoring_system:
    EAR_THRESHOLD = 0.24
    EAR_SUBSEQUENT_FRAMES = 4
    COUNTER = 0
    HEAD_COUNTER = 0
    FACE_COUNTER = 0
    MAR_THRESHOLD = 0.08
    HEAD_SUBSEQUENT_FRAMES = 6
    NO_FACE_DETECTION_SUBSEQUENT_FRAMES = 4
    PITCH_UP = 5.0
    PITCH_DOWN = -25.0
    YAW_UP = 25.0
    YAW_DOWN = -22.0
    line_pairs = [[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 0],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 4],
                  [0, 4],
                  [1, 5],
                  [2, 6],
                  [3, 7]]
     
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    buzzer = 23
    GPIO.setup(buzzer,GPIO.OUT)
    GPIO.output(buzzer,GPIO.LOW)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("landmarks.dat")
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(1.0)

    while True:
            frame = vs.read()
            frame = library.resize(frame, width=570)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray)

            if len(rects) > 0:
                    FACE_COUNTER = 0
                    shape = predictor(gray, rects[0])
                    coords = np.zeros((68, 2), int)
                    for i in range(0, 68):
                            coords[i] = (shape.part(i).x, shape.part(i).y)
                    reprojected_points, euler_angle = library.get_head_pose(coords)
                    if euler_angle[0, 0] > PITCH_UP or euler_angle[0, 0] < PITCH_DOWN or euler_angle[1, 0] > YAW_UP or euler_angle[1, 0] < YAW_DOWN:
                            HEAD_COUNTER += 1
                    else:
                            HEAD_COUNTER = 0
                    leftEye = coords[42:48]
                    rightEye = coords[36:42]
                    leftEAR = library.eye_aspect_ratio(leftEye)
                    rightEAR = library.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    if ear < EAR_THRESHOLD:
                            COUNTER += 1
                    else:
                            COUNTER = 0
            if len(rects) > 0:
                    if euler_angle[0, 0] > PITCH_UP or euler_angle[0, 0] < PITCH_DOWN or euler_angle[1, 0] > YAW_UP or euler_angle[1, 0] < YAW_DOWN:
                            if HEAD_COUNTER >= HEAD_SUBSEQUENT_FRAMES:
                                    GPIO.output(buzzer,GPIO.HIGH)
                                    time.sleep(0.05)
                                    GPIO.output(buzzer,GPIO.LOW)
                    if ear < EAR_THRESHOLD:
                            if COUNTER >= EAR_SUBSEQUENT_FRAMES:
                                    GPIO.output(buzzer,GPIO.HIGH)
                                    time.sleep(0.05)
                                    GPIO.output(buzzer,GPIO.LOW)