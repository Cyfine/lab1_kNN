import numpy
import numpy as np
from numpy import *
import operator
from os import listdir


# numpy.set_printoptions(threshold=np.inf)


def file2matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print("\n Original features")
print(datingDataMat)
print("\n Labels:")
print(datingLabels[0:20])


def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(minVals, (m, 1))
    normDataset = normDataset / tile(ranges, (m, 1))
    return normDataset, ranges, minVals


normMat, ranges, minVals = autoNorm(datingDataMat)
print('\n Normalized features:')
print(normMat)


def classify0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabels = labels[sortedDistIndices[i]]
        classCount[voteIlabels] = classCount.get(voteIlabels, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classifyPerson():
    resultList = ['not at all', ' in small doses', 'in large doses']
    ffMiles = float(input('\n Feature 1:'))
    percentTats = float(input("\n Feature 2:"))
    iceCream = float(input("\n Feature 3:"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifier_result = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("\n You will probably like this person: ", resultList[classifier_result - 1])
