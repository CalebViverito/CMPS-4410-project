import json
import os
import pickle
from sklearn.metrics import auc, confusion_matrix, roc_curve
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import seaborn as sns
import pandas as pd
import datetime

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

# RENAMING MODEL FILE BASED OFF EPOCH STATS
def RenameModel(history, time):
    summary = SummarizeTraining(history)
    bestEpoch = summary['best_epoch']
    bestValLoss = summary['best_val_loss']
    bestValAccuracy = summary['best_val_accuracy']

    newName = f'models\epoch{bestEpoch}_loss{bestValLoss:.4f}_acc{bestValAccuracy:.4f}_{time}.h5'
    os.rename('modelName.h5', newName)

# HISTORY SAVING
def SaveHistory(history, time):
    summary = SummarizeTraining(history)
    bestEpoch = summary['best_epoch']
    bestValLoss = summary['best_val_loss']
    bestValAccuracy = summary['best_val_accuracy']

    # save as pickle
    fileName = f'model_history\epoch{bestEpoch}_loss{bestValLoss:.4f}_acc{bestValAccuracy:.4f}_{time}'
    with open(f'{fileName}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # save as json
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

def PlotROCCurves(yTest, yPred, modelPath):
    labels = ['Sprint', 'Mile', 'Medium', 'Long']
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Ensure arrays are 2D
    yTest = np.array(yTest)
    yPred = np.array(yPred)
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        # Calculate ROC curve and AUC for each class
        y_true_class = yTest[:, i].ravel()
        y_pred_class = yPred[:, i].ravel()
        
        fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_class)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Label Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save figure
    time = datetime.datetime.now().strftime("%H-%M_%m-%d-%y")
    plt.savefig(f'model_plots/rc_{modelPath}_{time}.png', dpi=300, bbox_inches='tight')
    plt.close()

def PlotConfusionMatrices(yTest, yPredBinary, modelPath):
    """
    Plot confusion matrices for each class in multi-label classification
    """
    labels = ['Sprint', 'Mile', 'Medium', 'Long']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (label, ax) in enumerate(zip(labels, axes)):
        # Create confusion matrix for this class
        cm = confusion_matrix(yTest[:, i], yPredBinary[:, i], labels=[0, 1])
        
        # Ensure cm is 2x2 (handle edge cases)
        if cm.shape != (2, 2):
            # Create a proper 2x2 matrix
            cm_full = np.zeros((2, 2), dtype=int)
            if cm.shape == (1, 1):
                # Only one class present
                if yTest[:, i].sum() == 0 and yPredBinary[:, i].sum() == 0:
                    cm_full[0, 0] = cm[0, 0]  # All true negatives
                elif yTest[:, i].sum() == len(yTest) and yPredBinary[:, i].sum() == len(yPredBinary):
                    cm_full[1, 1] = cm[0, 0]  # All true positives
            cm = cm_full
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        
        ax.set_title(f'{label} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    plt.tight_layout()
    
    # Save figure
    time = datetime.datetime.now().strftime("%H-%M_%m-%d-%y")
    plt.savefig(f'model_plots/cm_{modelPath}_{time}.png', dpi=300, bbox_inches='tight')
    plt.close()


def PlotROCCurves(yTest, yPred, modelPath):
    """
    Plot ROC curves for each class in multi-label classification
    """
    labels = ['Sprint', 'Mile', 'Medium', 'Long']
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Ensure arrays are 2D
    yTest = np.array(yTest)
    yPred = np.array(yPred)
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        # Calculate ROC curve and AUC for each class
        y_true_class = yTest[:, i].ravel()
        y_pred_class = yPred[:, i].ravel()
        
        fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_class)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Label Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save figure
    time = datetime.datetime.now().strftime("%H-%M_%m-%d-%y")
    plt.savefig(f'model_eval/roc_curve_{modelPath}_{time}.png', dpi=300, bbox_inches='tight')
    plt.close()
    