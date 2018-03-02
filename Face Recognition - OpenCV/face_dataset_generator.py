import cv2
cam = cv2.VideoCapture(0)

detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Id=(input('enter your id: '))
num_img=0

while(True):
    
    _, image_frame = cam.read()

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        num_img += 1

        cv2.imwrite("dataset/User." + str(Id) + '.' + str(num_img) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif num_img>60:
        break

cam.release()
cv2.destroyAllWindows()
