import cv2
import numpy as np
#jak w poprzednich plikach
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
#ustawienia czcionki 
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
#ustawienie kamery z sprzetu
cam = cv2.VideoCapture(0)

while True:
    #czytanie kamery
    #czesci doklądnie taka sama jak w dataset do wykrywania twarzy
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,155,155),2)
        #tutaj porównoje podobienstwo z faces z id trainner/trainner.yml
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        #ustawiamy prawdopodowbienstwo, powinno byc ustawione na 50( jesli nie rozpoznaje Twojej twarzy, zwiększaj stopniowo wartosc
        # z drugiej strony jesli wartosc jest za duza bedzie przypisywal rozpoznanie kazdej twarzy ktora pojawi sie w obiektywie
        if(conf<=40):
            if(Id==1):
                #tutaj mozesz zmienic na swoje imie
                Id="Patryk"
        else:
            Id="Unknown"
            #tutaj wypisuje tekst juz rozpoznanaego ID
        font = cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor)
    #wyswietla wynik na oknie 'im'
    cv2.imshow('im',im) 
    #zamkniecie programu wciskajac 'q'
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
