import cv2
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt

imgpath = r'C:\Users\halif\programming\projectFaceDetection\ImagesFromCustomer\image.png'

def takePicture():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Take a picture")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Take a picture", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            # SPACE pressed
            cv2.imwrite(imgpath, frame)
            print("taking picture!")
            break

    cam.release()

    cv2.destroyAllWindows()



def faceRecognize():
    print("Proccessing...")
    image = cv2.imread(imgpath)
    imgplot = plt.imshow(image)

    try:
        analyze = DeepFace.analyze(img_path=imgpath, actions=['age','race','gender', 'emotion'])
        print("results:")
        print("  age (estimated):", analyze['age'])
        print("  dominant race:", analyze['dominant_race'])
        print("  gender:", analyze['gender'])
        print("  dominant emotion:", analyze['dominant_emotion'])

    except:
        print("The machine could not detect a face!")

if __name__ == '__main__':
    takePicture()
    faceRecognize()
