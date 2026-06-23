import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, hand_label):
	tips = [8, 12, 16, 20]
	count = 0

	#thumb
	thumb_tip_x = hand_landmarks.landmark[4].x
	thumb_joint_x = hand_landmarks.landmark[3].x

	if hand_label == "Right":
		if thumb_tip_x < thumb_joint_x:
			count += 1

	else:
		if thumb_tip_x > thumb_joint_x:
			count += 1

	
	# other fingers
	for tip in tips:
		tip_y = hand_landmarks.landmark[tip].y
		below_y = hand_landmarks.landmark[tip - 2].y
		if tip_y < below_y:
			count +=1
	return count

while True:
	ret, frame = cap.read()
	if not ret:
		break

	frame = cv2.flip(frame, 1)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = hands.process(rgb)


	if results.multi_hand_landmarks and results.multi_handedness:
		for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
			hand_label = handedness.classification[0].label
			mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
			fingers = count_fingers(hand_landmarks, hand_label)
			cv2.putText(frame, f'{hand_label} Hand - Fingers: {fingers}', (10, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	cv2.imshow('Gesture Recognition', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
