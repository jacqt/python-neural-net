import neuralnet
import json
import pickle
#This script tests the the net stored under BestNet.pkl using the test data
#stored under testdata.pkl

def Indicator(n):
    if n > .5: return 1
    else     : return 0

def testNet():
    "Tests the net against inputs"
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
    print "ACCURACY:", float(100*right)/(right+wrong)
myNet = pickle.load(open("BestNet.pkl","rb"))

trainingDataFile = open("testdata.pkl","r")
testData = json.loads(trainingDataFile.read())
print testData[:10]

testNet()
