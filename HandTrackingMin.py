import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # Use 0 for the default camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if success:  # Check if image capture was successful
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        results = hands.process(imgRGB)

        # Draw the landmarks on the image
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Print the id and landmark of each hand, the hand landmarks are indexed from 0 to 20, each with an x, y, z value representing the position of the landmark
                for id, lm in enumerate(handLms.landmark):
                    print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if id == 0: # Draw a circle on the first landmark
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


        # Calculate FPS the time it takes to process the image
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the image
        cv2.imshow("Image", img)
        cv2.waitKey(1)
