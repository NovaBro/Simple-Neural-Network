import numpy as np
import matplotlib.pyplot as plt
import extraFunc as mineFunc


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

#----SETTINGS----
learningRate = 0.0000001
epoch = 1000 #how many Cycles
maxVal = 50 #max value in range of inputs, also min value is -maxVal
totalInputs = 10 #number or entries in dataset

weights0 = np.random.random((3,2))
weights1 = np.random.random((1,3))

bias0 = np.random.random((3,1))
bias1 = np.random.random((1,1))
#----------------

#Since weights and bias start randomly, you can get unlucky and go out of bounds,
#get numbers that are way to large for python to handle
def resetFunc(): 
    Rweights0 = np.random.random((3,2))
    Rweights1 = np.random.random((1,3))

    Rbias0 = np.random.random((3,1))
    Rbias1 = np.random.random((1,1))
    return Rweights0, Rweights1, Rbias0, Rbias1

def reluFunc(x):
    if (x < 0):
        return 0
    return x

def DerivReluFunc(x):
    if (x < 0):
        return 0
    return 1

def correctValue(xIN, yIN):
    return xIN**2 + yIN**2

vFunc = np.vectorize(reluFunc)
vFunc2 = np.vectorize(DerivReluFunc)

#Partial derivative of each weight, refer to note sheet for drawn out diagram of equations.
#The partial derivative is dependent on next and previous nodes, which each weight is connected to
#Not all weights and connected to all nodes, so must be careful on multiplication on nodes, weights, for partial derivative
def findSlope(weightSlope, AnyWeights, PrevNodes, NextNodes): #zval??
    for r in range(np.shape(AnyWeights)[0]):
        for c in range(np.shape(AnyWeights)[1]):
            slopeVal = DerivReluFunc(NextNodes[r]) * AnyWeights[r][c] * PrevNodes[c]
            weightSlope[r][c] = slopeVal

def generateData(totalInputs, maxVal):
    dataArray = np.random.random((totalInputs, 2)) * 2 * maxVal - maxVal
    return dataArray

def oneTrainingCycle(weights0, weights1, bias0, bias1, inputValArray):
    global vFunc, vFunc2

    weightSlope0 = np.zeros(np.shape(weights0)) #Initialize where to store partial derivatives "Slopes"
    weightSlope1 = np.zeros(np.shape(weights1)) 
    sumTotal = 0

    #for x /for y in range(4):
    for i in range(10):
        inputVal = inputValArray[i]
        inputVal = inputVal[:, np.newaxis]
        node1 = vFunc(np.matmul(weights0,inputVal) + bias0)
        node2 = vFunc(np.matmul(weights1,node1) + bias1)
        sumTotal += (node2 - correctValue(inputVal[0], inputVal[1])) ** 2

        findSlope(weightSlope0, weights0, inputVal, node1)

        weightSlope1 = 2 * (node2 - correctValue(inputVal[0], inputVal[1])) * vFunc2(node2) * np.transpose(node1)
        for r in range(np.shape(weightSlope0)[0]):
            weightSlope0[r] = 2 * (node2 - correctValue(inputVal[0], inputVal[1])) * vFunc2(node2) * weights1[0][r] * weightSlope0[r][:]

        biasSlope1 = 2 * (node2 - correctValue(inputVal[0], inputVal[1])) * vFunc2(node2) * bias1
        biasSlope0 =  2 * (node2 - correctValue(inputVal[0], inputVal[1])) * vFunc2(node2) * np.transpose(weights1) * vFunc2(node1) * bias0

        weights1 = weights1 - learningRate * weightSlope1
        weights0 = weights0 - learningRate * weightSlope0

        bias1 = bias1 - learningRate * biasSlope1
        bias0 = bias0 - learningRate * biasSlope0
    #-------

    meanSquared = sumTotal / 16
    return weights0, weights1, bias0, bias1, meanSquared


meanSquaredValues = np.array([])
inputValArray = generateData(totalInputs, maxVal)

for t in range(epoch):
    epochTime = np.arange(epoch)
    weights0, weights1, bias0, bias1, meanSquared = oneTrainingCycle(weights0, weights1, bias0, bias1, inputValArray)
    meanSquaredValues = np.append(meanSquaredValues, meanSquared) 

ax2.plot(epochTime, meanSquaredValues, '.r')

plt.show()