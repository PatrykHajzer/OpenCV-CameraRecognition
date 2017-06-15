import cv2,os
import numpy as np
from PIL import Image
#recognizer, dla tej funkcji musisz miec zainstalowany opencv_contrib
#LBPHFaceRecognizer jest 1 z 3 reconizerow dostepnych  w opencv
# dwa pozostałe to EigenFaceRecognizer i FisherFaceRecognizer
recognizer = cv2.face.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #tworzy diwe tablice
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        #tu jest magia kotrej nie rozumiem
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
#zapisuje plik a arrays w przygotowanym katalogu trainner
recognizer.save('trainner/trainner.yml')
