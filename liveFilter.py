# Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Defining classifiers
nose_cascade = cv2.CascadeClassifier("cascades/nose.xml")
eyes_cascade = cv2.CascadeClassifier("cascades/eyes.xml")


# Saving the accessories
mustache_image = cv2.imread("assets/mustache.png", -1)
glasses_image = cv2.imread("assets/glasses.png", -1)


# Resize function
def resizeImage(img, height=None, width=None):
    (h, w) = img.shape[:2]

    if height is None:

        return cv2.resize(img, (width, int(width * h / w)))

    if width is None:

        return cv2.resize(img, (int(height * w / h), height))


# Pasting the accessories
def pasteAccessories(img):
    modified_image = np.copy(img)

    for nose in nose_cascade.detectMultiScale(img, 1.3, 5):

        x, y, w, h = nose
        # cv2.rectangle(modified_image, (x,y), (x+w, y+h), (255,0,0), 3)
        noseOffset = 20
        new_mustache_image = resizeImage(mustache_image, width=w + noseOffset)

        for i in range(new_mustache_image.shape[0]):
            for j in range(new_mustache_image.shape[1]):
                if new_mustache_image[i][j][3] > 0:
                    modified_image[y + 3 * h // 5 + i, x + j -
                                   noseOffset // 2, :] = new_mustache_image[i, j, :3]

    for eyes in eyes_cascade.detectMultiScale(img, 1.3, 5):

        x, y, w, h = eyes
        eyesOffset = 0
        new_glasses_image = resizeImage(glasses_image, width=w + eyesOffset)

        for i in range(new_glasses_image.shape[0]):
            for j in range(new_glasses_image.shape[1]):
                if new_glasses_image[i][j][3] > 0:
                    modified_image[y + i, x + j - eyesOffset //
                                   2, :] = new_glasses_image[i, j, :3]

    return modified_image


# Capturing the video from primary camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    modified_frame = pasteAccessories(frame)

    cv2.imshow("Video", modified_frame)

    # Wait for user input
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        break   # Break on pressing q

# Cleanup

cap.release()
cv2.destroyAllWindows()
