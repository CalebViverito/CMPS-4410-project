import datetime
import os
from create_model import *
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

def LoadModelAndTestData(modelPath, csvPath, imgDir):
    model = tf.keras.models.load_model(modelPath, compile=False)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    xData, yData = LoadData(csvPath, imgDir)
    xTrainVal, xTest, yTrainVal, yTest = train_test_split(
            xData, yData,
            test_size=0.2,
            random_state=42
        )
    return model, xTest, yTest, xTrainVal, yTrainVal
    

def EvalModel(model, xTest, yTest, modelPath, threshold=0.5):
    # predictions
    yPred = model.predict(xTest)
    yPredBinary = (yPred > threshold).astype(int)

    # model performance
    loss, accuracy, precision, recall = model.evaluate(xTest, yTest)
    f1 = 2 * (precision * recall) / (precision + recall)

    time = datetime.datetime.now().strftime("%H-%M_%m-%d-%y")
    
    with open(f'model_eval\{modelPath}_{time}.txt', 'w') as f:
        f.write(f'{modelPath}\n')
        f.write(f'\nModel Performance')
        f.write(f'\n\tTest Loss: {loss:.4f}')
        f.write(f'\n\tTest Accuracy: {accuracy:.4f} ({accuracy*100:.4f}%)')
        f.write(f'\n\tTest Precision: {precision:.4f}')
        f.write(f'\n\tTest Recall: {recall:.4f}')
        f.write(f'\n\tTest F1 Score: {f1:.4f}\n')
        f.write(f'\nPer-Class performance')

    # label performance
    labels = ['Sprint', 'Mile', 'Medium', 'Long']
    for i, label in enumerate(labels):
        classTrue = yTest[:,i]
        classPred = yPredBinary[:,i]

        tp = np.sum((classTrue == 1) & (classPred == 1))
        fp = np.sum((classTrue == 0) & (classPred == 1))
        fn = np.sum((classTrue == 1) & (classPred == 0))
        tn = np.sum((classTrue == 0) & (classPred == 0))

        classPrecision = tp / (tp + fp) if (tp + fp) > 0 else 0
        classRecall = tp / (tp + fn) if (tp + fn) > 0 else 0
        classF1 = 2 * (classPrecision * classRecall) / (classPrecision + classRecall) if (classPrecision + classRecall) > 0 else 0
        classAccuracy = (tp + tn) / len(classTrue)

        with open(f'model_eval\{modelPath}_{time}.txt', 'a') as f:
            f.write(f'\n{label:8}')
            f.write(f'\n\tAccuracy: {classAccuracy:.3f}')
            f.write(f'\n\tPrecision: {classPrecision:.3f}')
            f.write(f'\n\tRecall: {classRecall:.3f}')
            f.write(f'\n\tF1: {classF1:.3f}')
    
    return yPred, yPredBinary

def TestNewData(model, imgDir, modelPath, threshold=0.5):
    labels = ['Sprint', 'Mile', 'Medium', 'Long']
    time = datetime.datetime.now().strftime("%H-%M_%m-%d-%y")
    with open(f'model_result\{modelPath}_{time}.txt', 'w') as f:
        f.write(f'{modelPath}')
        f.write(f'\nModel Results')

    for file in os.listdir(imgDir):
        
        imgPath = os.path.join(imgDir, file)
        img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(256,256))
        imgArray = tf.keras.preprocessing.image.img_to_array(img)
        imgArray = imgArray / 255.0
        imgArray = np.expand_dims(imgArray, axis=0)

        pred = model.predict(imgArray)[0]
        predBinary = (pred > threshold).astype(int)

        with open(f'model_result\{modelPath}_{time}.txt', 'a') as f:
            f.write(f'\n\nIMAGE: {imgPath}')
            for i, label in enumerate(labels):
                status = "1" if predBinary[i] == 1 else "0"
                f.write(f'\n\t {status} {label}: {pred[i]:.3f}')

def EvalModelWithTraining(model, xTrain, yTrain, xTest, yTest, modelPath, threshold=0.5):
    labels = ['Sprint', 'Mile', 'Medium', 'Long']
    
    # Training predictions
    yTrainPred = model.predict(xTrain)
    yTrainPredBinary = (yTrainPred > threshold).astype(int)
    
    # Test predictions
    yTestPred = model.predict(xTest)
    yTestPredBinary = (yTestPred > threshold).astype(int)
    
    # Generate both confusion matrices
    PlotConfusionMatrices(yTrain, yTrainPredBinary, f"{modelPath}_TRAIN")
    PlotConfusionMatrices(yTest, yTestPredBinary, f"{modelPath}_TEST")
    
    return yTestPred, yTestPredBinary
