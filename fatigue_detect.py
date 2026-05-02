import cv2
import numpy as np
import mediapipe as mp
import pygame
import threading

# -------------------- SOUND SETUP --------------------
pygame.mixer.init()
pygame.mixer.music.load("alert_sound.mp3")

alert_playing = False

def play_alert_sound():
    global alert_playing
    if not alert_playing:
        pygame.mixer.music.play(-1)
        alert_playing = True

def stop_alert_sound():
    global alert_playing
    if alert_playing:
        pygame.mixer.music.stop()
        alert_playing = False

# -------------------- MEDIAPIPE SETUP --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

# Status counters
sleep = 0
drowsy = 0
active = 0

status = ""
color = (0, 0, 0)

# -------------------- FUNCTIONS --------------------
def compute(ptA, ptB):
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

def eye_aspect_ratio(eye_points, landmarks):
    up = compute(landmarks[eye_points[1]], landmarks[eye_points[5]]) + \
         compute(landmarks[eye_points[2]], landmarks[eye_points[4]])
    down = compute(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return up / (2.0 * down)

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w = frame.shape[:2]
            landmarks = []

            # Convert normalized points to pixel coordinates
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))

            # Calculate EAR
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)

            # -------------------- LOGIC --------------------
            if left_ear < 0.21 or right_ear < 0.21:
                sleep += 1
                drowsy = 0
                active = 0

                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)
                    threading.Thread(target=play_alert_sound).start()

            elif 0.21 <= left_ear < 0.25 or 0.21 <= right_ear < 0.25:
                sleep = 0
                active = 0
                drowsy += 1

                if drowsy > 6:
                    status = "Drowsy !"
                    color = (255, 0, 0)
                    threading.Thread(target=play_alert_sound).start()

            else:
                sleep = 0
                drowsy = 0
                active += 1

                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)
                    stop_alert_sound()

            # -------------------- DISPLAY --------------------
            cv2.putText(frame, status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Draw eye landmarks
            for point in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[point], 2, (255, 255, 255), -1)

    cv2.imshow("Driver Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
