import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = bool(mode)  # Ensure it is a boolean
        self.maxHands = int(maxHands)  # Ensure it is an integer
        self.detectionCon = float(detectionCon)  # Ensure it is a float
        self.trackCon = float(trackCon)  # Ensure it is a float

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)  # Create a hands object
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        self.results = self.hands.process(imgRGB)

        # Draw the landmarks on the image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findsPostion(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmList

    def findNumber(self, img, lmList):
        tipIds = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in tipIds[1:]:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Calculate the total number of extended fingers
        totalFingers = fingers.count(1)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 25)

        return totalFingers

    def howManyGlasses(self,img,lmList):
        image = cv2.imread('inglourious-basterds_-3-glasses.jpg')
        if image is None:
            return
        # Resize the image to fit on the video feed
        h, w, _ = img.shape
        image = cv2.resize(image, (w, h))
        thumbUp = False
        firstFingerUp = False
        secondFingerUp = False
        tipIds = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
            thumbUp = True
        else:
            fingers.append(0)
            thumbUp = False

        # Fingers
        for id in tipIds[1:]:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # First and second fingers up?
        firstFingerUp = bool(fingers[1])
        secondFingerUp = bool(fingers[2])
        therdFingerUp = bool(fingers[3])


        # Calculate the total number of extended fingers
        totalFingers = fingers.count(1)
        if therdFingerUp and firstFingerUp and secondFingerUp:
            overlay = cv2.addWeighted(img, 0.5, image, 0.5, 0)
            cv2.imshow("Image", overlay)
        else:
            cv2.putText(img, f"{totalFingers} glasses", (45, 375), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 0), 5)

        return totalFingers , therdFingerUp and firstFingerUp and secondFingerUp

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    detector = HandDetector()

    while True:
        tries = False
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findsPostion(img,draw=False) # Don't draw the landmarks
        if len(lmList) != 0:
            number, tries = detector.howManyGlasses(img, lmList)
            print("Number shown: ", number)

        # Calculate FPS the time it takes to process the image
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the image
        if tries:
            img = cv2.imread('inglourious-basterds_-3-glasses.jpg')
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
