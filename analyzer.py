import sys
import os
import string
import pdb
import math
from collections import defaultdict

def main():
    preProcessing("trainingSet.txt", "testSet.txt")

def preProcessing(trainingSetFile, testSetFile):
    # Making the vocab
    vocab = set()
    with open(trainingSetFile) as file:
        for line in file:
            line = line.translate(str.maketrans('', '', string.punctuation)).lower()
            words = line.strip().split(' ')
            for word in words:
                if not word in string.whitespace and not word in vocab:
                    if word != '1' and word != '0':
                        vocab.add(word)
    vocab = sorted(vocab)
    # for x in vocab:
    #     print(x)

    trainingFeatureList = []
    with open(trainingSetFile) as file:
        for line in file:
            line = line.translate(str.maketrans('', '', string.punctuation)).lower()
            words = line.strip().split(' ')
            listOfWordsInALine = []
            listOfBoolsForALine = []
            for word in words:
                if not word in string.whitespace:
                    listOfWordsInALine.append(word)
            # So now I have a line without the tab spaces.
            # Lets create the first feature vector
            for vocabWord in vocab:
                if vocabWord in listOfWordsInALine:
                    listOfBoolsForALine.append(1)
                else:
                    listOfBoolsForALine.append(0)
            # Now M columns have been done. Add the class label
            listOfBoolsForALine.append(int(listOfWordsInALine[-1]))
            # print(len(listOfBoolsForALine))
            # Now I've added the class label
            trainingFeatureList.append(listOfBoolsForALine)
            # Now we've added this list to the list of features

    testFeatureList = []
    with open(testSetFile) as file:
        for line in file:
            line = line.translate(str.maketrans('', '', string.punctuation)).lower()
            words = line.strip().split(' ')
            listOfWordsInALine = []
            listOfBoolsForALine = []
            for word in words:
                if not word in string.whitespace:
                    listOfWordsInALine.append(word)
            # So now I have a line without the tab spaces.
            # Lets create the first feature vector
            for vocabWord in vocab:
                if vocabWord in listOfWordsInALine:
                    listOfBoolsForALine.append(1)
                else:
                    listOfBoolsForALine.append(0)
            # Now M columns have been done. Add the class label
            listOfBoolsForALine.append(int(listOfWordsInALine[-1]))
            # print(len(listOfBoolsForALine))
            # Now I've added the class label
            testFeatureList.append(listOfBoolsForALine)
            # Now we've added this list to the list of features

    fileTrain = open('preprocessed_train.txt', 'w')
    for vocabWord in vocab:
        fileTrain.write(str(vocabWord))
        fileTrain.write(",")
    fileTrain.write("classlabel")
    fileTrain.write("\n")
    for eachSentence in testFeatureList:
        for eachBool in eachSentence:
            fileTrain.write(str(eachBool))
            fileTrain.write(",")
        fileTrain.write("\n")
    fileTrain.close()

    fileTest = open('preprocessed_test.txt', 'w')
    for vocabWord in vocab:
        fileTest.write(str(vocabWord))
        fileTest.write(",")
    fileTest.write("\n")
    for eachSentence in testFeatureList:
        for eachBool in eachSentence:
            fileTest.write(str(eachBool))
            fileTest.write(",")
        fileTest.write("\n")
    fileTest.close()

    numCorrect = bayesCalculate(trainingFeatureList, trainingFeatureList)
    print("For trainingSet as training data and trainingSet as testing data:")
    print("Matching Entries: ", numCorrect, "/", len(trainingFeatureList))
    print("Accuracy: ", (numCorrect)/(len(trainingFeatureList)))

    print("")
    print("For trainingSet as training data and testingSet as testing data:")
    numCorrect = bayesCalculate(trainingFeatureList, testFeatureList)
    print("Matching Entries: ", numCorrect, "/", len(testFeatureList))
    print("Accuracy: ", (numCorrect)/(len(testFeatureList)))

def probabilityOfOccurence(trainingFeatureList, testFeatureList, wordNum, sentenceNum, check):
    counter = 0
    for lineNum in range(0, len(trainingFeatureList)):
        if(trainingFeatureList[lineNum][wordNum] == testFeatureList[sentenceNum][wordNum] and trainingFeatureList[lineNum][-1] == check):
            counter += 1   
    counter += 1
    return counter

def bayesCalculate(trainingFeatureList, testFeatureList):
    counter = 0
    for sentenceNum in range(0, len(testFeatureList)):
        if recursiveProbabilityChecker(trainingFeatureList, testFeatureList, sentenceNum) == testFeatureList[sentenceNum][-1]:
            counter = counter + 1   
    return counter

def recursiveProbabilityChecker(trainingFeatureList, testFeatureList, sentenceNum):
    global1s = 0
    global0s = 0

    for line in trainingFeatureList:
        if line[-1] == 0:
            global0s += 1
        if line[-1] == 1:
            global1s += 1
    
    probabilityOf1s = math.log10((global1s)/(len(trainingFeatureList)))
    probabilityOf0s = math.log10((global0s)/(len(trainingFeatureList)))
    
    recursivePOf1s = 0
    recursivePOf0s = 0
    for i in range(0, len(testFeatureList[0])-1):
        recursivePOf1s += math.log10((probabilityOfOccurence(trainingFeatureList, testFeatureList, i, sentenceNum, 1))/(global1s + 2))
        recursivePOf0s += math.log10((probabilityOfOccurence(trainingFeatureList, testFeatureList, i, sentenceNum, 0))/(global0s + 2))
    probabilityOf1s += recursivePOf1s 
    probabilityOf0s += recursivePOf0s

    return 0 if (probabilityOf1s < probabilityOf0s) else 1

main()
