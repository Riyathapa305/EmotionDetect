import cv2 as cv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from main import EmotionCNN  
from preprocess import CustomPreprocess  



model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv.VideoCapture(0)
preprocess = CustomPreprocess()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]  
        
        img = preprocess(roi)  
        img = img.unsqueeze(0)  
        
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            mood = emotion_labels[predicted.item()]
        
        
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, mood, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv.imshow("Emotion Detector", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


