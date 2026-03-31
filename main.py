import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7
)

# Meme settings
MEME_SIZE = (300, 300)

meme_paths = {
    "rollsafe": "memes/rollsafe.jpg",
    "pointing": "memes/pointing.jpg",
    "shock": "memes/shock.jpg",
    "calm": "memes/calm.jpg",
    "thinking": "memes/thinking.jpg"
}

# Load memes
memes = {}
for name, path in meme_paths.items():
    if os.path.exists(path):
        memes[name] = cv2.resize(cv2.imread(path), MEME_SIZE)
    else:
        placeholder = np.zeros((MEME_SIZE[1], MEME_SIZE[0], 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Missing: {name}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        memes[name] = placeholder


# ---------------- GESTURE DETECTION ----------------
def detect_gesture(hand_landmarks, face_landmarks):
    # Key Landmarks
    index_tip = hand_landmarks.landmark[8]
    wrist = hand_landmarks.landmark[0]
    temple = face_landmarks.landmark[127]
    chin = face_landmarks.landmark[152]
    forehead_top = face_landmarks.landmark[10]

    # Calculate distance from finger to center of face (Landmark 1 is the nose)
    nose = face_landmarks.landmark[1]
    dist_to_face_center = np.sqrt((index_tip.x - nose.x)**2 + (index_tip.y - nose.y)**2)

    # 1. SHOCK: Hand must be ABOVE forehead AND CLOSE to the face/head
    if index_tip.y < forehead_top.y + 0.05 and dist_to_face_center < 0.25:
        return "shock"

    # 2. POINTING: Index tip raised high but FURTHER AWAY from the face center
    # This prevents the upward finger from being called "shock"
    if index_tip.y < wrist.y - 0.15 and dist_to_face_center >= 0.25:
        return "pointing"

    # 3. ROLLSAFE: Finger near temple
    dist_to_temple = np.sqrt((index_tip.x - temple.x)**2 + (index_tip.y - temple.y)**2)
    if dist_to_temple < 0.08:
        return "rollsafe"

    # 4. THINKING: Finger near chin
    dist_to_chin = np.sqrt((index_tip.x - chin.x)**2 + (index_tip.y - chin.y)**2)
    if dist_to_chin < 0.08:
        return "thinking"
    
    return None


# ---------------- FACE DETECTION ----------------
def detect_face_states(face_landmarks):
    # Mouth → shock
    mouth_dist = abs(face_landmarks.landmark[13].y -
                     face_landmarks.landmark[14].y)

    if mouth_dist > 0.03:
        return "shock"

    # Eyes → calm
    eye_dist = abs(face_landmarks.landmark[159].y -
                   face_landmarks.landmark[145].y)

    if eye_dist < 0.01:
        return "calm"

    return None


# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)

print("System Ready. Press ESC to exit.")

prev_meme = None
stable_count = 0
display_meme = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_res = hands.process(rgb_frame)
    face_res = face_mesh.process(rgb_frame)

    active_meme = None

    # Face first
    if face_res.multi_face_landmarks:
        face_lms = face_res.multi_face_landmarks[0]
        active_meme = detect_face_states(face_lms)

        # Hand overrides face
        if hand_res.multi_hand_landmarks:
            hand_lms = hand_res.multi_hand_landmarks[0]
            gesture = detect_gesture(hand_lms, face_lms)
            if gesture:
                active_meme = gesture

    # -------- STABILITY FILTER --------
    if active_meme == prev_meme:
        stable_count += 1
    else:
        stable_count = 0

    if stable_count > 3:
        display_meme = active_meme

    prev_meme = active_meme

    # -------- UI --------
    h, w, _ = frame.shape
    canvas = np.zeros((h, w + MEME_SIZE[0], 3), dtype=np.uint8)
    canvas[:h, :w] = frame

    if display_meme:
        meme_img = memes[display_meme]
        y_offset = (h - MEME_SIZE[1]) // 2

        canvas[y_offset:y_offset + MEME_SIZE[1], w:] = meme_img

        cv2.putText(canvas, display_meme.upper(),
                    (w + 10, y_offset - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Meme Gesture Recognition", canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()