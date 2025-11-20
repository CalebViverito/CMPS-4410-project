import json
import os
import pickle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import pandas as pd
import jsonpickle

# ADJUST WEIGHTS
def AdjustWeights(model):
    for i, layer in enumerate(model.layers):
        print(f'Layer {i}: {layer.name} - {type(layer).__name__}')

        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            weights = layer.get_weights()
            print(f'    Number of weight arrays: {len(weights)}')
            for j, w in enumerate(weights):
                print(f'        Weight {j} shape: {w.shape}')
    
    firstConv = model.layers[0]
    currentWeights = firstConv.get_weights()

    modifiedWeights = [w * 0.9 for w in currentWeights]
    firstConv.set_weights(modifiedWeights)
    print('\nweights modified')

    return model


# SUMMARIZE MODEL
def SummarizeTraining(history):
    historyDict = history.history

    # find best epoch
    valLoss = historyDict['val_loss']
    bestEpoch = np.argmin(valLoss) + 1
    bestValLoss = valLoss[bestEpoch - 1]

    # get best epoch stats
    trainLoss = historyDict['loss'][bestEpoch - 1]
    valAccuracy = historyDict['val_binary_accuracy'][bestEpoch - 1]
    trainAccuracy = historyDict['binary_accuracy'][bestEpoch - 1]
    valPrecision = historyDict['val_precision'][bestEpoch - 1]
    valRecall = historyDict['val_recall'][bestEpoch - 1]

    # F1
    if valPrecision + valRecall > 0:
        F1 = 2 * (valPrecision * valRecall) / (valPrecision + valRecall)
    else:
        F1 = 0.0

    # epoch stats
    totalEpochs = len(valLoss)
    finalValLoss = valLoss[-1]
    finalValAccuracy = historyDict['val_binary_accuracy'][-1]

    overfit = trainAccuracy - valAccuracy

    return {
        'best_epoch': bestEpoch,
        'total_epochs': totalEpochs,
        'best_val_loss': bestValLoss,
        'best_val_accuracy': valAccuracy,
        'best_train_accuracy': trainAccuracy,
        'best_val_precision': valPrecision,
        'best_val_recall': valRecall,
        'best_f1_score': F1,
        'final_val_loss': finalValLoss,
        'final_val_accuracy': finalValAccuracy,
        'overfitting_gap': overfit
    }

#  RENAMING MODEL FILE BASED OFF EPOCH STATS
def RenameModel(history):
    summary = SummarizeTraining(history)
    bestEpoch = summary['best_epoch']
    bestValLoss = summary['best_val_loss']
    bestValAccuracy = summary['best_val_accuracy']

    newName = f'models\epoch{bestEpoch}loss{bestValLoss:.4f}acc{bestValAccuracy:.4f}.h5'
    os.rename('modelName.h5', newName)

# HISTORY SAVING
def SaveHistory(history):
    summary = SummarizeTraining(history)
    bestEpoch = summary['best_epoch']
    bestValLoss = summary['best_val_loss']
    bestValAccuracy = summary['best_val_accuracy']

    fileName = f'model_history\epoch{bestEpoch}loss{bestValLoss:.4f}acc{bestValAccuracy:.4f}'
    with open(f'{fileName}.pkl', 'wb') as f:
        pickle.dump(history.history, f)
        pickleData = pickle.dumps(history.history)

    jsonFile = f'{fileName}.json'

    historyDict = {}
    for key, value in history.history.items():
        if isinstance(value, np.ndarray):
            historyDict[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            historyDict[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            historyDict[key] = int(value)
        else:
            historyDict[key] = value

    with open(jsonFile, 'w') as f:
        json.dump(historyDict, f, indent=2)

    
def LoadHistory(fileName):
    with open(fileName, 'rb') as f:
        historyDict = pickle.load(f)

    class HistoryWrapper:
        def __init__(self, historyDict):
            self.history = historyDict
    return HistoryWrapper(historyDict)
