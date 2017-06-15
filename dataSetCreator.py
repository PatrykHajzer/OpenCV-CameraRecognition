import cv2
cam = cv2.VideoCapture(0)
# tą funkcja program dobiera się do kamery na Twoim sprzęcie, domyslnie jest to 0. Gdyby nie pojawiało się zmień wartość na -1 lub 1
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#gotowy detektor od haarcascade - ważne , musi być pobrany ten plik

Id=raw_input('enter your id')
sampleNum=0
#numer sampla ( w tym przypadku 20 zdjęć jdenego id)
while(True):
    ret, img = cam.read()
    #standardowa funkcja zczytyjąca obraz z camery
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, 1.3, 5)
    #detektor korzysta z funkcji multiscale dzieki ktorej moze rozpoznac twarz o dowolnych rozmiarach
    #dzieki atrybutowi gray detektor bedzie miał mniej koloróœ do analizowania(zamiast B- Blue , G-Green, R- Red) analizuje tylko jeden kolor grey
    for (x,y,w,h) in faces:
        #po wykryciu twarzy tworzy kwadrat/prostokąt o punktach pop rzekątenj x,y i x+width,y+height, (kolor w formacie BGR) i grubość ramki
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #każda kolejna twarz ma inny numer sampla
        sampleNum=sampleNum+1
        #format zapisu otrzymanego obrazu do pliku
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        #funkcja tworząca okno o nazwie frame(jesli ta funkjca jest w petli bedzie to oznaczac ze ono sie pojawi tylko jesli detektor rozponzna gdzies twarz)
        cv2.imshow('frame',img)
    #wyłącznie programy jesli wcisnie sie 'q'
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif sampleNum>20:
        #tu mozesz zmieniac liczbę zdjęc- pamietaj wiecej zdjęc wieksza dokładność przy rozpoznawaniu i wiecej roboty dla trenera
        break
#standart w opencv do zamykania połacznia z kamerą i zmakniecia wsyzstkich okien
cam.release()
cv2.destroyAllWindows()
