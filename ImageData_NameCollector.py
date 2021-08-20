import cv2 
import numpy as np
import pandas as pd

cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:/Users/ASUS/Downloads/haarcascade_frontalface_alt.xml")
dataset_path = "C:/Users/ASUS/Downloads/veyom"
skip=0
face_data=[]
file_name=input("Enter the name of the person : ")


while True:
	ret,frame=cap.read()
	if ret ==False:
		continue 
	faces=face_cascade.detectMultiScale(frame,1.3,5)

	if len(faces)==0:
		continue
	#Selecting the largest Face 
	faces=sorted(faces,key=lambda f:f[2]*f[3])
	for face in faces[-1:] :
		x,y,w,h=face
		
		offset=10
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		face_section=frame[y-offset:y+offset+h,x-offset:x+offset+w]
		face_section=cv2.resize(face_section,(100,100))
		skip+=1
		if skip%10==0:
			face_data.append(face_section)
	cv2.imshow("Full Frame",frame)
	cv2.imshow("Cropped image",face_section)
	keyPressed=cv2.waitKey(1)&0xFF
	if keyPressed==ord("q"):
		break

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Succesfully saved at "+dataset_path+file_name+'.npy')



cap.release()
cv2.destroyAllWindows()


