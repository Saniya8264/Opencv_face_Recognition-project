# Write a puthon Script that captures images from your webcam video stream
# Extract all Faces from the images frame (using haarcascades)
# stores the face information into numpy arrays

#1. read and show video stream, capture images
#2. detect Faces and show bounding box
# 3. Flatten the largest face image(gray scale) and save in a numpy array
#4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# init camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")



skip =0
face_data =[]
dataset_path ='./data/'


file_name = input("enter the name of the person :")

while True:
    ret,frame = cap.read()

    if ret==False:
        continue
 
    gray_farme =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    # pick the last face because last face is largest according to area
    for face in faces[-1:]:
        x,y,w,h =face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


        # extract (crop out the rquired face) : Region of Interset
        offset = 10
        face_section = frame[y- offset:y+h+offset, x-offset:x+w+offset]
        face_section =cv2.resize(face_section,(100,100))


        skip += 1

        if skip%10 ==0:
            face_data.append(face_section)
            print(len(face_data))






    cv2.imshow("Frame",frame)
    cv2.imshow("Face Section",face_section)




    key_pressed = cv2.waitkey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert  our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+ dataset_path +file_name+'.npy')

cap.release()
cap.destroyAllwindows()

