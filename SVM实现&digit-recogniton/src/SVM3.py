# -*-coding:utf-8 -*-
import numpy as np
import random



class optParameters:
    """
    optParameters: SVM要优化的参数
    Parameters：
        data  - 数据矩阵
        classLabels - 数据标签
        C           - 松弛变量
        toler       - 容错率
        kTup        - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
    """
    def __init__(self, data, classLabels, C, toler, kTup):
        self.X = data                                     
        self.labelMat = classLabels                       
        self.C = C                                        
        self.tol = toler                                 
        self.m = np.shape(data)[0]                       
        self.alphas = np.mat(np.zeros((self.m,1)))         #初始化alpha参数为0   
        self.b = 0                                         #初始化参数b为0
        self.eCache = np.mat(np.zeros((self.m,2)))         #初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = np.mat(np.zeros((self.m,self.m)))         #初始化核K
        for i in range(self.m):                            #计算所有数据的核K
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup):
    """
    kernelTrans:通过核函数将数据转换更高维的空间
    Parameters:
        X - 数据矩阵
        A - 单个数据的向量
        kTup - 包含核函数信息的元组
    Returns:
        K - 计算的核K
    """
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': 
        K = X * A.T                                     #线性核函数,只进行内积。
    elif kTup[0] == 'rbf':                              #高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))                     #计算高斯核K
    else: 
        raise NameError('核函数无法识别')
    return K                                             #返回计算的核K



def calcEk(P, k):
    """
    calcEk(): 
        计算误差
    Parameters：
        P  - SVM参数
        k  - 标号为k的数据
    Returns:
        Ek - 标号为k的数据误差
    """
    fXk = float(np.multiply(P.alphas,P.labelMat).T*P.K[:,k] + P.b)
    Ek = fXk - float(P.labelMat[k])
    return Ek

def selectJrand(i, m):
    """
    selectJrand:
        随机选择alpha_j的索引值
    Parameters:
        i - alpha_i的索引值
        m - alpha参数个数
    Returns:
        j - alpha_j的索引值
    """
    j = i                                 #选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def selectJ(i, P, Ei):
    """
    内循环启发方式2
    Parameters：
        i - 标号为i的数据的索引值
        P  - SVM参数
        Ei - 标号为i的数据误差
    Returns:
        j, maxK - 标号为j或maxK的数据的索引值
        Ej - 标号为j的数据误差
    """
    maxK = -1; maxDeltaE = 0; Ej = 0                         #初始化
    P.eCache[i] = [1,Ei]                                      #根据Ei更新误差缓存
    validEcacheList = np.nonzero(P.eCache[:,0].A)[0]        #返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:                            #有不为0的误差
        for k in validEcacheList:                           #遍历,找到最大的Ek
            if k == i: continue                             #不计算i,浪费时间
            Ek = calcEk(P, k)                                
            deltaE = abs(Ei - Ek)                            
            if (deltaE > maxDeltaE):                        
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej                                        
    else:                                                   #没有不为0的误差
        j = selectJrand(i, P.m)                            #随机选择alpha_j的索引值
        Ej = calcEk(P, j)                                  
    return j, Ej                                             

def updateEk(P, k):
    """
    计算Ek,并更新误差缓存
    Parameters：
        P  - SVM参数
        k - 标号为k的数据的索引值
    Returns:
        无
    """
    Ek = calcEk(P, k)                                        #计算Ek
    P.eCache[k] = [1,Ek]                                    #更新误差缓存


def clipAlpha(aj,H,L):
    """
    修剪alpha_j
    Parameters:
        aj - alpha_j的值
        H - alpha上限
        L - alpha下限
    Returns:
        aj - 修剪后的alpah_j的值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerL(i, P):
    """
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        P  - SVM参数
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    #步骤1：计算误差Ei
    Ei = calcEk(P, i)
    #优化alpha,设定一定的容错率。
    if ((P.labelMat[i] * Ei < -P.tol) and (P.alphas[i] < P.C)) or ((P.labelMat[i] * Ei > P.tol) and (P.alphas[i] > 0)):
        #使用内循环启发方式2选择alpha_j,并计算Ej
        j,Ej = selectJ(i, P, Ei)
        #保存更新前的aplpha值，使用深拷贝
        alphaIold = P.alphas[i].copy(); alphaJold = P.alphas[j].copy();
        #步骤2：计算上下界L和H
        if (P.labelMat[i] != P.labelMat[j]):
            L = max(0, P.alphas[j] - P.alphas[i])
            H = min(P.C, P.C + P.alphas[j] - P.alphas[i])
        else:
            L = max(0, P.alphas[j] + P.alphas[i] - P.C)
            H = min(P.C, P.alphas[j] + P.alphas[i])
        if L == H:
            print("L==H")
            return 0
        #步骤3：计算eta
        eta = 2.0 * P.K[i,j] - P.K[i,i] - P.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        #步骤4：更新alpha_j
        P.alphas[j] -= P.labelMat[j] * (Ei - Ej)/eta
        #步骤5：修剪alpha_j
        P.alphas[j] = clipAlpha(P.alphas[j],H,L)
        #更新Ej至误差缓存
        updateEk(P, j)
        if (abs(P.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        #步骤6：更新alpha_i
        P.alphas[i] += P.labelMat[j]*P.labelMat[i]*(alphaJold - P.alphas[j])
        #更新Ei至误差缓存
        updateEk(P, i)
        #步骤7：更新b_1和b_2
        b1 = P.b - Ei- P.labelMat[i]*(P.alphas[i]-alphaIold)*P.K[i,i] - P.labelMat[j]*(P.alphas[j]-alphaJold)*P.K[i,j]
        b2 = P.b - Ej- P.labelMat[i]*(P.alphas[i]-alphaIold)*P.K[i,j]- P.labelMat[j]*(P.alphas[j]-alphaJold)*P.K[j,j]
        #步骤8：根据b_1和b_2更新b
        if (0 < P.alphas[i]) and (P.C > P.alphas[i]): P.b = b1
        elif (0 < P.alphas[j]) and (P.C > P.alphas[j]): P.b = b2
        else: P.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
    """
    线性SMO算法
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
        kTup - 包含核函数信息的元组
    Returns:
        P.b - SMO算法计算的b
        P.alphas - SMO算法计算的alphas
    """
    P = optParameters(np.mat(dataMatIn), np.mat(classLabels).T, C, toler, kTup)    # 构建数据结构容纳所有数据
    iter = 1                                                                        #初始化当前迭代次数
    entireSet = True; alphaPairsChanged = 0
    while (iter <= maxIter) and ((alphaPairsChanged > 0) or (entireSet)):     #遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:                                                                                                           
            for i in range(P.m):       
                alphaPairsChanged += innerL(i,P)                                     #使用优化的SMO算法
                print("第%d次迭代,样本:%d" % (iter,i))
            iter += 1
        else:                                                                        #遍历非边界值
            nonBoundIs = np.nonzero((P.alphas.A > 0) * (P.alphas.A < C))[0]          #遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,P)
                print("第%d次迭代,样本:%d" % (iter,i))
            iter += 1
        if entireSet:                                                                 #遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):         #如果alpha没有更新,计算全样本遍历
            entireSet = True 
        print("迭代次数: %d" % iter)
    return P.b, P.alphas                       



