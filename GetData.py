import convert
import itertools
import os
import random
import pickle
import json

#This file reads all the images stored in the ./data folder
#Then the file converts them all into feature vectors, normalizes the
#values, and then writes the data to testdata.pkl

def GetData():
    "Gets the data"
    curDir = os.getcwd()
    zeroesDir = os.path.join(curDir,"data/zero")
    onesDir = os.path.join(curDir,"data/one")
    twosDir = os.path.join(curDir,"data/two")
    threesDir = os.path.join(curDir,"data/three")
    foursDir = os.path.join(curDir,"data/four")
    fiveDirs = os.path.join(curDir,"data/five")
    sixDirs = os.path.join(curDir,"data/six")
    sevenDirs = os.path.join(curDir,"data/seven")
    eightDirs = os.path.join(curDir,"data/eight")
    nineDirs = os.path.join(curDir,"data/nine")

    zeroesDirList = os.listdir(zeroesDir)
    onesDirList = os.listdir(onesDir)
    twosDirList = os.listdir(twosDir)
    threesDirList = os.listdir(threesDir)
    foursDirList = os.listdir(foursDir)
    fiveDirsList = os.listdir(fiveDirs)
    sixDirsList = os.listdir(sixDirs)
    sevenDirsList = os.listdir(sevenDirs)
    eightDirsList = os.listdir(eightDirs)
    nineDirsList = os.listdir(nineDirs)
    listoflistofdirs = [zeroesDirList,onesDirList,twosDirList,threesDirList,foursDirList,fiveDirsList,sixDirsList,sevenDirsList,eightDirsList,nineDirsList]
    listofdirs = [zeroesDir,onesDir,twosDir,threesDir,foursDir,fiveDirs,sixDirs,sevenDirs,eightDirs,nineDirs]

    print zeroesDirList[:10]

    data = []

    for i in range(len(listofdirs)):
        listdir = listoflistofdirs[i]
        prevpath = listofdirs[i]
        for zFile in listdir:
            path = os.path.join(prevpath,zFile)
            feature = convert.getFeatureVec(path)
            k = [0]*10
            k[i] = 1
            dataPoint = [feature,k]
            data.append(dataPoint)

    data = normalizeAll(data)
    random.shuffle(data)
    return data

def normalizeAll(data):
    s = 0
    for datapoint in data:
        j = datapoint[0]
        s += sum(j)/float(len(j))
    k = s/float(len(data))
    print "NORMALIZING FACTOR:", k
    return map ((lambda datap: [map(lambda z:(z-k)/100.0,datap[0]),datap[1]]), data)


def normalize(image):
    return map (normalize1,list(itertools.chain(*(image))))

def normalize1(n):
    return ((255-n*16)- 177)/100.0

def LargeData():
    digits = datasets.load_digits()
    total = 0
    data = []
    for i in range(len(digits.images)):
        feature = normalize(digits.images[i])
        target = [0]*10
        target[digits.target[i]] = 1
        data.append([feature,target])
    print data[0]
    #k = GetData()
    a = open("testdata.pkl","w")
    dataJson = json.dumps(data)
    a.write(dataJson)
    a.close()

def renameFiles(name):
    curDir = os.getcwd()
    dDir = os.path.join(curDir,"data\\"+name)
    dList = os.listdir(dDir)
    c = 1
    for zFile in dList:
        old = os.path.join(dDir,zFile)
        new = os.path.join(dDir,"'"+str(c)+".png")
        print old,new
        os.rename(old,new)
        c+=1

    dList = os.listdir(dDir)
    c = 1
    for zFile in dList:
        old = os.path.join(dDir,zFile)
        new = os.path.join(dDir,str(c)+".png")
        print old,new
        os.rename(old,new)
        c+=1

if __name__ == "__main__":
    data = GetData()
    a = open("testdata.pkl","w")
    dataJson = json.dumps(data)
    a.write(dataJson)
