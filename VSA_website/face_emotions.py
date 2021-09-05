#importing cv2 and matplotlib
import cv2
import matplotlib.pyplot as plt
#importing deepface library and DeepFace
from deepface import DeepFace

class face_emotion:
    def image_load(self,image_path):
        #loading image
        img =  cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #loading image
        print(img)
        # color_img = cv2.cvtColor(img, cv2.COLOR_GRAYSCALE)
        # print(color_img)
        #calling VGGFace
        # model_name = "VGG-Face"
        # model = DeepFace.build_model(model_name) 
        prediction = DeepFace.analyze(img_path = img, enforce_detection=False)
        print(prediction['dominant_emotion'])

fe = face_emotion()
fe.image_load('../media/video_frames/abc (10).jpg')