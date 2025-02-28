import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame

pygame.mixer.init()
pygame.mixer.music.load("C:/Users/Acer/Music/alarm.wav")  # Load an alert sound file


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (3.0 * D)
    return mar

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.6
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Acer/Music/shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        mouth = landmarks[48:68]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                pygame.mixer.music.play()
                cv2.putText(frame,"DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,1.5,(0, 0, 255),3)
        else:
            COUNTER = 0

        if mar > MOUTH_AR_THRESH:
            pygame.mixer.music.play()
            cv2.putText(frame, "YAWNING DETECTED!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()