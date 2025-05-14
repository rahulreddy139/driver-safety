import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import threading

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold for closed eyes
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3

# Counter variables
closed_eyes_frames = 0
alarm_playing = False

# Alarm function
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        playsound(r"C:\Users\NITYA REDDY\Downloads\facility-siren-loopable-100687.mp3")


  # Make sure this file exists
        alarm_playing = False

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Eye landmarks (MediaPipe's 468-point model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if eyes are closed
            if ear < EAR_THRESHOLD:
                closed_eyes_frames += 1
                status = "Eyes Closed"

                if closed_eyes_frames >= CONSEC_FRAMES:
                    threading.Thread(target=play_alarm).start()
            else:
                closed_eyes_frames = 0
                status = "Eyes Open"

            # Draw eye landmarks
            for x, y in left_eye + right_eye:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Display EAR and status
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Eye Closure Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
