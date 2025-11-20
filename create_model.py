import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import pandas as pd
import pickle

# imports are being weird, still works at runtime though
# do roc curve and confusion matrix

# LOAD DATA
def LoadData(csvPath, imgDir):
    df = pd.read_csv(csvPath)
    images = []
    raceLabels = []

    for index, row in df.iterrows():
        imgPath = f"{imgDir}\{row.fileName}"
        img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(256,256))
        imgArray = tf.keras.preprocessing.image.img_to_array(img)
        imgArray = imgArray / 255.0

        images.append(imgArray)
        races = [row.sprint, row.mile, row.medium, row.long]
        raceLabels.append(races)

    return np.array(images), np.array(raceLabels)


# BUILDING MODEL
def BuildModel(inputShape = (256, 256, 3), numLabels = 4):

    model = models.Sequential([
        # convolution block 1
        layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu',
            input_shape=inputShape,
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),

        # convolution block 2
        layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),

        # convolution block 3
        layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='relu',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),

        # dense layers
        layers.Dense(
            units=256,
            activation='relu'
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(
            units=128,
            activation='relu'
        ),
        layers.Dropout(0.5),

        # output
        layers.Dense(
            units=numLabels,
            activation='sigmoid'
        )
    ])

    return model

def CompileModel(model, learningRate = 0.001):
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate),
        metrics = [
            'binary_accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

# TRAIN MODEL
def TrainModel(model, xTrain, yTrain, xVal, yVal, epochs=50, batch_size=32):
    dataGen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'modelName.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5
        )
    ]

    history = model.fit(
        dataGen.flow(xTrain, yTrain, batch_size=batch_size),
        validation_data=(xVal, yVal),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history