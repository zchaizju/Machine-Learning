# !/usr/bin/env python3
# coding=utf-8
"""
Support Vector Machine,SVM
Author  :Chai Zheng
Blog    :http://blog.csdn.net/chai_zheng/
Github  :https://github.com/Chai-Zheng/Machine-Learning
Email   :zchaizju@gmail.com
Date    :2017.10.3
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

class SVM():
    def __init__(self,dataset,labels,C,toler,kernelOption):
        self.train_x = dataset
        self.train_y = labels
        self.C = C
        self.toler = toler
        self.numSamples = dataset.shape[0]
        self.alphas = np.zeros((self.numSamples,1))
        self.b = 0
        self.errorCache = np.zeros((self.numSamples,2))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x,self.kernelOpt)

def calcKernelValue(matrix_x,sample_x,kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = np.zeros((1,numSamples))
    if kernelType == 'linear':
            kernelValue = np.dot(matrix_x,sample_x.T)
    elif kernelType == 'rbf':
            sigma = kernelOption[1]
            if sigma == 0:
                sigma =1
            for i in range(numSamples):
                diff = matrix_x[i,:] - sample_x
                kernelValue[0,i] = np.exp((np.dot(diff,diff.T)/(-2.0*sigma**2)))
    else:
        raise NameError('Not support kernel type! You should use linear or rbf!')
    return kernelValue

def calcKernelMatrix(train_x,kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = np.zeros((numSamples,numSamples))
    for i in range(numSamples):
        kernelMatrix[i,:] = calcKernelValue(train_x,train_x[i,:],kernelOption)
    return kernelMatrix

def calcError(svm,alpha_k):     #计算第k个样本的误差，k∈[1，m]
    output_k = float(np.dot((svm.alphas*svm.train_y).T,svm.kernelMat[:,alpha_k])+svm.b)
    error_k = output_k-float(svm.train_y[alpha_k])
    return error_k

def updateError(svm,alpha_k):   #更新误差
    error = calcError(svm,alpha_k)
    svm.errorCache[alpha_k] = [1,error]

def innerLoop(svm,alpha_1,error_1,train_x): #内循环，根据alpha1确定alpha2
    svm.errorCache[alpha_1] =[1,error_1]
    candidateAlphaList = np.nonzero(svm.errorCache[:,0])[0]
    maxStep = 0
    alpha_2 = 0
    error_2 = 0
    numSample = train_x.shape[0]
    if len(candidateAlphaList)>1:
        #找出|E2-E1|最大的alpha2
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_1:
                continue
            error_k = calcError(svm,alpha_k)
            if abs(error_1-error_k)>maxStep:
                maxStep = abs(error_1-error_k)
                alpha_2 = alpha_k
                error_2 = error_k
    else:   #第一次进入，随机选择alpha2
        while alpha_2 == alpha_1:   #alpha_2不能等于alpha_1
            alpha_2 = np.random.randint(svm.numSamples)
        error_2 = calcError(svm,alpha_2)

        #采用下述方式来初始化alpha_2位置，可稳定结果。采用上述方法会稍微不稳定，但这是正常的。
        # if alpha_1 == numSample:
        #     alpha_2 = numSample - 1
        # else:
        #     alpha_2 = alpha_1 + 1
        # error_2 = calcError(svm,alpha_2)

    return alpha_2,error_2

def outsideLoop(svm,alpha_1,train_x):
    error_1 = calcError(svm,alpha_1)

    #检查alpha_1是否违背KKT条件
    if ((svm.alphas[alpha_1]<svm.C) and (svm.alphas[alpha_1]>0) and ((svm.train_y[alpha_1]*error_1>svm.toler) or (svm.train_y[alpha_1]*error_1< -svm.toler)))\
                or((svm.train_y[alpha_1]*error_1<-svm.toler) and (svm.alphas[alpha_1]<svm.C))\
            or((svm.train_y[alpha_1]*error_1>svm.toler) and (svm.alphas[alpha_1]>0)):

        #固定alpha1，求alpha2
        alpha_2,error_2 = innerLoop(svm,alpha_1,error_1,train_x)
        alpha_1_old = svm.alphas[alpha_1].copy()    #拷贝，分配新的内存
        alpha_2_old = svm.alphas[alpha_2].copy()

        #alpha2的取值范围，其中L=<alpha2<=H，参见李航《统计学习方法》P126
        if svm.train_y[alpha_1] != svm.train_y[alpha_2]:
            L = max(0,alpha_2_old-alpha_1_old)
            H = min(svm.C,svm.C+alpha_2_old-alpha_1_old)
        else:
            L = max(0,alpha_2_old+alpha_1_old-svm.C)
            H = min(svm.C,alpha_2_old+alpha_1_old)

        eta = svm.kernelMat[alpha_1,alpha_1]+svm.kernelMat[alpha_2,alpha_2]-2.0*svm.kernelMat[alpha_1,alpha_2]
        svm.alphas[alpha_2] += svm.train_y[alpha_2]*(error_1-error_2)/eta  #计算alpha2_new

        #对alpha2进行剪辑
        if svm.alphas[alpha_2]>H:
            svm.alphas[alpha_2] = H
        elif svm.alphas[alpha_2]<L:
            svm.alphas[alpha_2] = L

        #如果alpha2无变化，返回，重选alpha1
        if abs(svm.alphas[alpha_2]-alpha_2_old)<0.00001:
            updateError(svm,alpha_2)
            return 0

        #更新alpha1
        svm.alphas[alpha_1] += svm.train_y[alpha_1]*svm.train_y[alpha_2]*(alpha_2_old-svm.alphas[alpha_2])

        #更新b
        b1 = svm.b-error_1-svm.train_y[alpha_1]*svm.kernelMat[alpha_1,alpha_1]*(svm.alphas[alpha_1]-alpha_1_old)\
                          -svm.train_y[alpha_2]*svm.kernelMat[alpha_2,alpha_1]*(svm.alphas[alpha_2]-alpha_2_old)
        b2 = svm.b-error_2-svm.train_y[alpha_1]*svm.kernelMat[alpha_1,alpha_2]*(svm.alphas[alpha_1]-alpha_1_old)\
                          -svm.train_y[alpha_2]*svm.kernelMat[alpha_2,alpha_2]*(svm.alphas[alpha_2]-alpha_2_old)

        #alpha2经剪辑，始终在(0,C)内。若1也满足，那么b1=b2；若1不满足，取均值
        if (svm.alphas[alpha_1]>0) and (svm.alphas[alpha_1]<svm.C):
            svm.b = b1
        else:
            svm.b = (b1+b2)/2.0

        updateError(svm,alpha_1)
        updateError(svm,alpha_2)
        return 1
    else:
        return 0

def SVMtrain(train_x,train_y,C,toler,maxIter,kernelOption=('rbf',1.0)):
    startTime = time.time()
    svm = SVM(train_x,train_y,C,toler,kernelOption)
    alphaPairsChanged = 1
    iterCount = 0

    # 迭代终止条件：
    #     1.到达最大迭代次数
    #     2.迭代完所有样本后alpha不再变化，也就是所有alpha均满足KTT条件

    while(iterCount<maxIter) and (alphaPairsChanged>0):
        alphaPairsChanged = 0   #标记在该次循环中，alpha有无被优化
        SupportAlphaList = np.nonzero((svm.alphas>0)*(svm.alphas<svm.C))[0] #支撑向量序号列表

        for i in SupportAlphaList:          #遍历支持向量
            alphaPairsChanged += outsideLoop(svm,i,train_x)

        for i in range(svm.numSamples):     #遍历所有样本
            alphaPairsChanged += outsideLoop(svm,i,train_x)

        iterCount += 1

    print('---Training Completed.Took %f s.Using %s kernel.'%((time.time()-startTime),kernelOption[0]))
    return svm

def SVMtest(svm,test_x,test_y):
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        kernelValue = calcKernelValue(svm.train_x,test_x[i,:],svm.kernelOpt)
        predict = np.dot(kernelValue,svm.train_y*svm.alphas)+svm.b
        if np.sign(predict) == np.sign(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount/numTestSamples)
    return accuracy

def SVMvisible(svm):    #仅针对二变量样本可视化，即被注释掉的训练数据，非葡萄酒数据
    w = np.zeros((2,1))
    for i in range(svm.numSamples):
        if svm.train_y[i] == -1:
            plt.plot(svm.train_x[i,0],svm.train_x[i,1],'or')
        elif svm.train_y[i] ==1:
            plt.plot(svm.train_x[i,0],svm.train_x[i,1],'ob')
        w += (svm.alphas[i]*svm.train_y[i]*svm.train_x[i,:].T).reshape(2,1)

    supportVectorIndex = np.nonzero(svm.alphas>0)[0]
    for i in supportVectorIndex:
        plt.plot(svm.train_x[i,0],svm.train_x[i,1],'oy')
    min_x = min(svm.train_x[:,0])
    max_x = max(svm.train_x[:,0])
    min_y = float((-svm.b-w[0]*min_x)/w[1])
    max_y = float((-svm.b-w[0]*max_x)/w[1])
    plt.plot([min_x,max_x],[min_y,max_y],'-g')
    plt.show()

if __name__ =='__main__':

    print('Step 1.Loading data...')
    #构建10个训练样本，6个测试样本，线性可分,若采用被注释的数据，可将本程序的最后一行取消注释，从而可视化结果
    # train_data = np.array([[2.95,6.63,1],[2.53,7.79,1],[3.57,5.65,1],[2.16,6.22,-1],[3.27,3.52,-1],[3,7,1],[3,8,1],[3,2,-1],[2,9,1],[2,4,-1]])
    # test_data = np.array([[3.16,5.47,1],[2.58,4.46,-1],[2,2,-1],[3,4,-1],[5,100,1],[6,1000,1]])
    #
    # train_x = train_data[:,0:2]
    # train_y = train_data[:,2].reshape(10,1)
    # test_x = test_data[:,0:2]
    # test_y = test_data[:,2].reshape(6,1)

    #数据集下载http://download.csdn.net/download/chai_zheng/10009314
    train_data = np.loadtxt("Wine_Train.txt",delimiter=',') #载入葡萄酒数据集
    test_data = np.loadtxt("Wine_Test.txt",delimiter=',')
    train_x = train_data[:,1:14]
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x) #数据标准化
    train_y = train_data[:,0].reshape(65,1)
    for i in range(len(train_y)):
        if train_y[i] == 1:     #修改标签为±1
            train_y[i] = -1
        if train_y[i] == 2:
            train_y[i] = 1
    test_x = test_data[:,1:14]
    test_x = scaler.transform(test_x)   #数据标准化
    test_y = test_data[:,0].reshape(65,1)
    for i in range(len(test_y)):
        if test_y[i] == 1:
            test_y[i] = -1
        if test_y[i] == 2:
            test_y[i] = 1

    print('---Loading completed.')
    print('Step 2.Training...')
    C = 0.6
    toler = 0.001
    maxIter = 100
    svmClassifier = SVMtrain(train_x,train_y,C,toler,maxIter,kernelOption=('rbf',2))
    print('Step 3.Testing...')
    accuracy = SVMtest(svmClassifier,test_x,test_y)
    print('---Testing completed.Accuracy: %.3f%%'%(accuracy*100))
    # SVMvisible(svmClassifier)
