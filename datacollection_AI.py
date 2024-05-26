import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300


while True:
        # Read a frame from the camera
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

            # Clamp the bounding box
        x = np.clip(x - offset, 0, img.shape[1])
        y = np.clip(y - offset, 0, img.shape[0])
        w = np.clip(w + 2*offset, 0, img.shape[1] - x)
        h = np.clip(h + 2*offset, 0, img.shape[0] - y)

        imgCrop = img[y-offset : y + h+offset, x-offset : x + w+offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCropShape = imgCrop.shape
        imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

        # Display the frame
    # cv2.imshow('Camera Feed', img)
    cv2.imshow('Image', img)
        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break