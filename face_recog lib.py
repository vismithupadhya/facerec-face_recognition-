import face_recognition
import cv2
import numpy as np
import os

path = r'D:\facerecogdatabase'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    currentImg = cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,1,1)
    img = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDir =  face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDir)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1= faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
        else:
            print("unknown")
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,"UNKNOWN",(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            
    cv2.imshow('Webcam',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
