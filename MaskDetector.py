import cv2
from imutils.video.videostream import VideoStream 
import numpy as np
from tensorflow.keras.models import load_model
# from tensorflow import *
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array
# import imutils



# faceCascade=cv2.CascadeClassifier("Resouces/face_haarcascade.xml")
prototxtFile=r"/media/shirin/DATA/PROJECT PYTHON MASK DET/FACE DET/deploy.prototxt"
weightsFile=r"/media/shirin/DATA/PROJECT PYTHON MASK DET/FACE DET/res10_300x300_ssd_iter_140000.caffemodel"
faceNet=cv2.dnn.readNet(prototxtFile,weightsFile)

maskModel=load_model("/media/shirin/DATA/PROJECT PYTHON MASK DET/July_6Model.model")

vs=VideoStream(src=0).start()

# width as 500

while True:
    img=vs.read()
    # img=imutils.resize(img, width=500)
    blob=cv2.dnn.blobFromImage(img,1.0,(224,224),(104.0,177.0,123.0))
    faceNet.setInput(blob)

    det=faceNet.forward()
    h=img.shape[0]
    w=img.shape[1]

    faces=[]
    position=[]
    prediction=[]

    for i in range(0 , det.shape[2]):
        prob=det[0,0,i,2]

        # print("printitng probability")
        # print(prob)
        if(prob>0.5):
            box=det[0,0,i,3:7]
            box[0]*=w
            box[1]*=h
            box[2]*=w
            box[3]*=h
            (startX, startY, endX, endY)=box.astype("int")

            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX),min(h-1,endY))
            
            # print(startX, startY, endX, endY)
            face=img[startY:endY,startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            
            faces.append(face)
            position.append((startX, startY, endX, endY))

    if(len(faces)>0):
        faces=np.array(faces,dtype="float32")
        prediction=maskModel.predict(faces, batch_size=32)


    for i in range(0,len(position)): 
        (startX, startY, endX, endY)=position[i]
        # print("------------------printing position of i----------------")
        # print(position[i])
        (WithMask, WithoutMask)=prediction[i]
        # print("*************printitng predicttion of i**************")
        # print(prediction[i])

        label=""
        colour=()
        if(WithMask>WithoutMask):
            label="With Mask"
            colour=(0,255,0)
        else:
            label="Without Mask"
            colour=(0,0,255)

        cv2.putText(img,label,(startX,startY-10),
            cv2.FONT_HERSHEY_DUPLEX,0.5,colour,2)
        cv2.rectangle(img,(startX,startY),(endX,endY),colour,2)



    cv2.imshow('mdwksl',img)
    
    if(cv2.waitKey(1)&0xFF==ord('q')):
        break


















    
    # face=faceCascade.detectMultiScale(img,1.1,4)

    # for x,y,w,h in face:
    #     newimg=img[x:x+w,y:y+h]
    #     print("-------------------------printing x,y,w,h of face--------------------------")
    #     print(x,y,w,h)
 
    # if len(newimg) > 0:
    #     cv2.imshow('newimg',newimg)
    #     shapedimg=cv2.resize(newimg,(224,224))
    #     pred=maskModel.predict(np.reshape(shapedimg,(1,224,224,3)))
    #     mask=pred[0][0]
    #     wmask=pred[0][1]
    #     for x,y,w,h in face:
    #         if(mask<wmask):
    #             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #             print("---------------printing face rectangle coord---------------")
    #             print(x,y,w,h)
    #         else:
    #             cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


