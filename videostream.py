import cv2

cap = cv2.VideoCapture(0)
face_casacde = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_casacde.detectMultiScale(gray_frame,1.3,5)
    
    cv2.imshow("Video Frame",frame)
    cv2.imshow("Gray Frame",gray_frame)



    # wait foe user input -q , then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
