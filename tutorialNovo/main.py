from PIL import Image 
import numpy as np
import os
import cv2
from keras.utils import np_utils

data=[]
labels=[]
animalsArray = ["cats", "dogs", "birds", "fishs"]
count = 0
for animal in animalsArray:
    cats=os.listdir(animal)
    for cat in cats:
        imag=cv2.imread(animal+"/"+cat)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(count)
    count += 1

animals=np.array(data)
labels=np.array(labels)

np.save("animals",animals)
np.save("labels",labels)

animals=np.load("animals.npy")
labels=np.load("labels.npy")

s=np.arange(animals.shape[0]) 
np.random.shuffle(s) 
animals=animals[s] 
labels=labels[s]

num_classes=len(np.unique(labels)) 
data_length=len(animals)

(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

import keras
from keras.utils import np_utils
#One hot encoding
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)
y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
#make model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4,activation="softmax"))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50
          ,epochs=100,verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)
    
def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=animalsArray[label_index]
    print(animal)
    print("The predicted Animal is a "+animal+" with accuracy =    "+str(acc))

predict_animal("meuLancer.jpg")
print("-------")
predict_animal("jatinho.jpeg")
