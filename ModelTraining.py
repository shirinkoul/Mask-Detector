import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import * #imgtoarray,loadimg,ImageDataGenerator is used from this library
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model # connect both the models
from tensorflow.keras.applications.mobilenet_v2 import * #preprocess input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from imutils.paths import *
import numpy as np
import cv2


init_lr=1e-4
ePochs=2
bs=32
print("[INFO] processing...")
directory=r"/media/shirin/DATA/PROJECT PYTHON MASK DET/Face Mask Dataset/Test"
categories=['WithMask','WithoutMask']

data=[]
labels=[]

for category in categories:
    path = os.path.join(directory,category)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))
        image=img_to_array(image)
        image=preprocess_input(image)

        data.append(image)
        labels.append(category)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data=np.array(data,dtype="float32")
labels=np.array(labels)

(trainX, testX, trainY, testY)=train_test_split(data, labels, test_size=0.2,stratify=labels, random_state=42)

aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,
    width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,
    horizontal_flip=True)

baseModel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

model=Model(inputs=baseModel.input,outputs=headModel)

for layer in baseModel.layers:
    layer.trainable=False

print("[INFO] compliling mode...")    

optAdam=Adam(learning_rate=init_lr, decay=init_lr/ePochs)
model.compile(loss="binary_crossentropy", optimizer=optAdam, metrics=['accuracy'])

print("[INFO] training...")

H=model.fit(aug.flow(trainX, trainY, batch_size=bs),
steps_per_epoch=len(trainX)//bs,
validation_data=(testX, testY),
validation_steps=len(testX)//bs,
epochs=ePochs)

print("[INFO] evaluating...")
predIdxs=model.predict(testX, batch_size=bs)

predIdxs= np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1),predIdxs,
target_names=lb.classes_))

# model.save('July_6model2.model',save_format='h5')
print("[INFO] saved model.")
# model.summary()  



















