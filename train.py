import neuralnet
import json
import pickle

#Trains the neural net on the data stored in the testdata.pkl file
#Trains the net to do basic digit recognition

def Indicator(n):
    if n > .5: return 1
    else     : return 0

def testNet():
    "Tests the net against inputs ranging from -100,-100 to 100,100"
    right = 0
    wrong = 0

    expectedList = []
    actualList = []
    i = 0
    for dataPoint in testData:
        myNet.ClearNodes()
        myNet.SetInputs(dataPoint[0])
        outputs = map (Indicator,myNet.ComputeOutput())
        expected = dataPoint[1]
        if expected == outputs:
            right += 1
        else:
            wrong += 1
        if i%100 == 0:
            print "expected:", expected, "actual:", outputs
        i += 1

    print "RIGHT:", right
    print "WRONG:", wrong


def trainNetDistr(testData,associatedWeights):
    "Trains the net giving a particular list of data and associated weights for the test data"
    myNet.ResetNet()
    length = len(testData)
    print max(associatedWeights)
    print min(associatedWeights)
    for j in range(150): #iterate 120 times over the data set
        print "Training iteration", j
        if j%10 == 0:
            testNet()
        if j%50 == 0:
            pass

        for i in range(length):
            data = testData[i]
            featureVec = data[0]
            target = data[1]
            weight = (length/40.0)*associatedWeights[i]
            myNet.TrainNet(featureVec,target,weight)

def calcNewWeights(testData,associatedWeights):
    """
    Calculates the pseuduo loss and uses it to calculate the new weights;
    just a possibly incorrect implementation of the adaboosting algorithm as described at this link:
        http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015178
    """
    outputs = []
    pseudoError = 0
    for i in range(len(testData)):
        data = testData[i]
        featureVec = data[0]
        targets = data[1]
        weight = associatedWeights[i]

        myNet.ClearNodes()
        myNet.SetInputs(featureVec)
        outputList = myNet.ComputeOutput()
        outputs.append(outputList)

        k = 1
        for i in range(len(targets)):
            target = targets[i]
            output = outputList[i]
            if target == 1:
                k += -output
            else:
                k += output

        pseudoError += weight*k
        if pseudoError < 0:
            print k
            print weight

    print pseudoError
    alpha = pseudoError/(1-pseudoError)
    if alpha < 0:
        print alpha
        return False

    #create newWeights
    newWeights = []
    for i in range(len(associatedWeights)):
        target = testData[i][1]
        output = outputs[i]
        weight = associatedWeights[i]
        k = 0
        for i in range(len(targets)):
            target = targets[i]
            output = outputList[i]
            if target == 1:
                k += output
            else:
                k += -output

        j = alpha**k
        newWeight = weight*j
        newWeights.append(newWeight)

    #normalize the newWeights
    sumOfWeights = sum(newWeights)
    if sumOfWeights == 0:
        return False #end algorithm here
    normalizer = 1/sumOfWeights
    for i in range(len(newWeights)):
        newWeights[i] = newWeights[i]*normalizer
    #print newWeights

    length = len(testData)
    #print associatedWeights[:20]
    #print newWeights[:20]

    return newWeights

myNet = neuralnet.NeuralNet()
myNet.ConstructNet(121,[22,50,10])

dataFile = open("testdata.pkl","r")

testData = json.loads(dataFile.read())
print testData[:10]
testWeights  =  [1.0/len(testData)]*len(testData) #creates a list of 1/n

#adaboosting algorithm
for i in range(5):
    myNet.StoreData()
    trainNetDistr(testData,testWeights)
    testNet()
    testWeights = calcNewWeights(testData, testWeights)
    if testWeights == False:
        break
myNet.StoreData()
testNet()
