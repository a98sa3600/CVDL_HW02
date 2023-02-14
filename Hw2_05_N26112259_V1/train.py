import tensorflow as tf
import os, cv2, random 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt    
import keras
from keras.layers import  Dense, Flatten, Activation,Conv2D, MaxPooling2D,Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import ResNet50V2
import tensorflow_addons as tfa

def build_model():
    Model = tf.keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, 
                                   input_shape=(224,224,3), pooling=None)
    x = Model.output
    x = Flatten(name='Flatten')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid', name='Dense')(x)
    model = tf.keras.models.Model(inputs=Model.input , outputs=output_layer) 
    return model
    

#Data_Set
IMAGE_SIZE = (224,224)
BATCH_SIZE = 100 
EPOCH = 20
LEARNING_RATE=8e-5

train = tf.keras.utils.image_dataset_from_directory('./Dataset_OpenCvDl_Hw2_Q5/training_dataset',image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
validation = tf.keras.utils.image_dataset_from_directory('./Dataset_OpenCvDl_Hw2_Q5/validation_dataset',image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

#Model_Set
model = build_model()
model.summary()

#1st loss function
#loss_function =tfa.losses.SigmoidFocalCrossEntropy(alpha=0.4, gamma=1.0 )

#2st loss function
loss_function =tf.losses.BinaryCrossentropy()


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Model.compile
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
# history = Model.fit(train,None,batch_size=32,validation_data=validation, validation_steps=len(validation), epochs=5)
# Model.save('Q5_Model.h5')

# Model.save
history =model.fit(train,
                steps_per_epoch = len(train),
                validation_data = validation,
                validation_steps = len(validation),
                epochs = EPOCH)
model.save('Q5_Model_loss2.h5')


plt.subplots(figsize=(6,4))
plt.plot(history.epoch,history.history["loss"],color="red",label="Training Loss")
plt.plot(history.epoch,history.history["val_loss"],color="blue",label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.savefig("Loss.png")

plt.subplots(figsize=(6,4))
plt.plot(history.epoch,history.history["accuracy"],color="red",label="Training Accuracy")
plt.plot(history.epoch,history.history["val_accuracy"],color="blue",label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("ACCURACY.png")
plt.show()