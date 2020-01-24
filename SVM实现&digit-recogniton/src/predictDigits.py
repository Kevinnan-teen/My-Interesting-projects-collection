#!/usr/bin/env python3
# -*-coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
from SVM3 import smoP, kernelTrans


def img2vector(filename):
    """
    将32x32的二进制图像转换为1x784向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x784向量
    """
    if filename.split('.')[-1] == 'jpg':
        image = Image.open( filename )
        returnVect = np.array( image )
        returnVect[ returnVect > 0 ] = 1
        returnVect = returnVect.reshape( 1, -1 )
    else:
        fr = open(filename)
        for i in range(28):
            lineStr = fr.readline()
            for j in range(28):
                returnVect[0,28*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName, classnum):
    """
    加载图片
    Parameters:
        dirName     - 文件夹名
    Returns:
        trainingMat - 数据矩阵
        labels    - 数据标签
    """
    labels = []
    trainingFileList = os.listdir(dirName)           
    m = len(trainingFileList)
    trainingMat = np.zeros((m,784))
    for i in range(m):
        fileName = trainingFileList[i]   
        classNum = int(fileName.split('_')[0])        # 类别号
        if classNum == classnum: 
            labels.append(-1)
        else: 
            labels.append(1)
        trainingMat[i,:] = img2vector(dirName + '/' + fileName)     # 将32*32的灰度图转换为784列向量
    return trainingMat, labels   

def predictDigits( classNum , kTup=('rbf', 10)):



    dataArr,labelArr = loadImages('../train', classNum)
    print(dataArr.shape)
    print(len(dataArr))
    
    
    b_path = '../weights/' + str(classNum) + '_' + kTup[0] + '_b.npy'
    alphas_path = '../weights/' + str(classNum) + '_' + kTup[0] + '_alphas.npy'

    if os.path.exists(b_path) and os.path.exists(alphas_path): 
        # 模型已训练， 加载使用
        print("加载模型!")
        b = np.mat( np.load(b_path) )
        alphas = np.mat( np.load(alphas_path) )
    else: 
        # 使用Platt SMO开始训练
        print("开始训练!")
        b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
        np.save( b_path, b )
        np.save( alphas_path, alphas) 

    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    svInd = np.nonzero(alphas.A>0)[0]
    
    sVs=datMat[svInd]           # 支持向量
    labelSV = labelMat[svInd];
    # print("支持向量个数:%d" % np.shape(sVs)[0])

    
    # 训练集数据测试
    print( "数字" + str(classNum) )
    print("<<==============训练集测试==============>>")
    trainResultFile = open('../results/trainResultFile.txt', 'a') 
    trainResultFile.write( "<<==============训练集测试==============>>\n" )

    trainSeterrorCount = 0      # 训练集预测错误个数
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 
        if np.sign(predict) != np.sign(labelArr[i]): 
            trainSeterrorCount += 1
            print( str( trainSeterrorCount ) + ',   预测错误!')
            trainResultFile.write( str( trainSeterrorCount ) +   '   预测错误!\n')
    print("数字" + str(classNum) + ":训练集错误率: %f%%" % ( float(trainSeterrorCount)/m * 100) )
    trainResultFile.write( "数字" + str(classNum) + ":训练集错误率: %f%%\n" % ( float(trainSeterrorCount)/m * 100) )
    trainResultFile.close()
    

    # 测试集数据测试
    print( "数字" + str(classNum) )
    print("<<==============测试集测试==============>>")
    testResultFile = open( '../results/testResultFile.txt', 'a' )
    testResultFile.write( "<<==============测试集测试==============>>\n" )

    dataArr,labelArr = loadImages('../test', classNum)
    testSeterrorCount = 0       # 测试集预测错误个数
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m,n = np.shape(datMat)

    global doit
    if doit == True:
        global Pre
        Pre = np.zeros( (m, 10) )
        
        doit = False


    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b

        Pre[i, classNum] = predict      # 各类预测值

        if np.sign(predict) != np.sign(labelArr[i]): 
            testSeterrorCount += 1  
            print( str(testSeterrorCount) + ',   预测错误!')
            testResultFile.write(str( testSeterrorCount ) + ',   预测错误!\n')
    print( "数字" + str(classNum) + ":测试集错误率: %f%%" % ( float(testSeterrorCount)/m * 100 ) )
    testResultFile.write("数字" + str(classNum) +  ":测试集错误率: %f%%\n" % ( float(testSeterrorCount)/m * 100 ) )
    testResultFile.close()


if __name__ == '__main__':
    Pre = None
    doit = True

    for i in range(10):
        predictDigits(i, kTup=['lin'])

    np.savetxt('../results/predictValues.txt', Pre, fmt="%2f")



