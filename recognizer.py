import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,155,155),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<=40):
            if(Id==1):
                Id="Patryk"
            elif(Id==2):
                Id="Joey Tribbiani"
        else:
            Id="Unknown"
        font = cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor) 
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
