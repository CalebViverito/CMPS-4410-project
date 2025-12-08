from create_model import CreateModel
from test_model import *

modelPath = f'epoch13_loss1.8940_acc0.7002_20-20_12-02-25'
testCsvPath = f'testdata.csv'
testImgDir = f'test_data'

#CreateModel()

# MODEL EVAL
#model, xTest, yTest, xTrain, yTrain = LoadModelAndTestData(f'models\{modelPath}.h5', testCsvPath, testImgDir)
#EvalModel(model, xTest, yTest, modelPath)

# TEST ON NEW DATA
#newTestImg = f'test_data'
#TestNewData(model, newTestImg, modelPath)
