from create_model import *
from model_functions import *
from sklearn.model_selection import train_test_split
# TODO: ROC CURVE AND CONFUSION MATRIX

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
model = CompileModel(model, learningRate=0.001)

history = TrainModel(
    model,
    xTrain, yTrain,
    xVal, yVal,
    epochs=5,
    batch_size=32
)

summary = SummarizeTraining(history)
SaveHistory(history)
RenameModel(history)