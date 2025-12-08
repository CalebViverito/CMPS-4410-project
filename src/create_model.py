import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import pandas as pd
from model_functions import *
from sklearn.model_selection import train_test_split
import datetime

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
            filters=16,
            kernel_size=(3,3),
            activation='relu',
            input_shape=inputShape,
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.3),

        # convolution block 2
        layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.4),

        # convolution block 3
        layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.6),

        layers.Flatten(),

        # dense layers
        layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.6),

        # output
        layers.Dense(
            units=numLabels,
            activation='sigmoid'
        )
    ])

    return model

def CalculateClassWeights(yTrain):
    posCounts = yTrain.sum(axis=0)
    total = len(yTrain)
    weights = total / (4 * posCounts)

    return {i: weights[i] for i in range(4)}

# def CalcAggressiveClassWeights(yTrain):
#     posCounts = yTrain.sum(axis=0)
#     total = len(yTrain)
#     weights = []

#     # inverse frequency w aggressive scaling
#     for i in range(4):
#         if posCounts[i] > 0:
#             weights[i] = np.sqrt(total / (4 * posCounts[i]))
#         else:
#             weights[i] = 1.0
    
#     # normalize medium weight to = 1.0
#     mediumWeight = weights[2]
#     for i in range(4):
#         weights[i] = weights[i] / mediumWeight
    
#     return weights


def CompileModel(model, learningRate = 0.001, classWeights=None):
    # if given class weights, calculate loss using weighted bce, else just default to bce
    if classWeights:
        weightsList = [classWeights[i] for i in range(4)]
        def WeightedBinaryCrossentropy(yTrue, yPred):
            yTrue = tf.cast(yTrue, tf.float32)
            yPred = tf.cast(yPred, tf.float32)

            bce = tf.keras.backend.binary_crossentropy(yTrue, yPred)

            weights = tf.constant(weightsList, dtype=tf.float32)
            weightedBce = bce * weights

            return tf.reduce_mean(weightedBce)
        
        loss = WeightedBinaryCrossentropy
    else:
        loss = 'binary_crossentropy'

    model.compile(
        loss = loss,
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate),
        metrics = [
            'binary_accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

def CompileWithWeights(model, yTrain, learningRate=0.0005):
    posCounts = yTrain.sum(axis=0)
    total = len(yTrain)
    weights = []

    # inverse frequency w aggressive scaling
    for i in range(4):
        if posCounts[i] > 0:
            weights.append(float(np.sqrt(total / (4 * posCounts[i]))))
        else:
            weights.append(1.0)
    
    # normalize medium weight to = 1.0
    mediumWeight = weights[2]
    # i think this is causing indice issues when loading the model after training it
    weights = [w / mediumWeight for w in weights]

    weightsTensor = tf.constant(weights, dtype=tf.float32)


    def weighted_loss(yTrue, yPred):
        yTrue = tf.cast(yTrue, tf.float32)
        yPred = tf.cast(yPred, tf.float32)
        
        bce = tf.keras.backend.binary_crossentropy(yTrue, yPred)
        weightedBce = bce * weightsTensor

        return tf.reduce_mean(weightedBce)
    
    model.compile(
        loss=weighted_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model

# TRAIN MODEL
def TrainModel(model, xTrain, yTrain, xVal, yVal, epochs=50, batch_size=32):
    # image augmentations
    dataGen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.25,
        brightness_range=[0.6,1.4],
        shear_range=0.2,
        fill_mode='nearest'
    )

    # callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'modelName.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss',
            mode='min',
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            monitor='val_loss',
            mode='min'
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

# CREATE MODEL
def CreateModel():
    csvPath = "horsedata.csv"
    imgDir = "db"
    xData, yData = LoadData(csvPath, imgDir)
    xTrain, xVal, yTrain, yVal = train_test_split(
        xData, yData,
        test_size=0.2,
        random_state=42
    )

    model = BuildModel(inputShape=(256,256,3), numLabels=4)
    model.summary()
    #classWeights = CalculateClassWeights(yTrain)
    #model = CompileModel(model, learningRate=0.0005, classWeights=classWeights)

    model = CompileWithWeights(model, yTrain)

    history = TrainModel(
        model,
        xTrain, yTrain,
        xVal, yVal,
        epochs=50,
        batch_size=16
    )

    time = datetime.datetime.now().strftime("%H-%M_%m-%d-%y")
    SaveHistory(history, time)
    RenameModel(history, time)