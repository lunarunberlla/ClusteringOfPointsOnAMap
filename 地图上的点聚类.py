
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

"""
函数说明：
       文本文件的解析函数
       加载数据集
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat



"""
函数说明：
       根据公式返回地球表面两点间的距离
"""
def distSlC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos((vecB[0,0] - vecA[0,0])*pi/180)
    return arccos(a+b)*6371.0



"""
函数说明：
       为给定数据集构建一个包含k个随机质心的集合
       随机质心必须要在整个数据集的边界之内，可以通过找到数据集每一维的最小和最大值来完成
       创建k个点作为起始质心，可以随机选择(位于数据边界内)
　　当任意一个点的簇分配结果发生改变时
　　　　对数据集中每一个点
　　　　　　　　对每个质心
　　　　　　　　　　计算质心与数据点之间的距离
　　　　　　　　将数据点分配到距其最近的簇
　　　　对每一个簇，计算簇中所有点的均值并将均值作为质心
"""
def randCent(dataSet, k):
    #获取数据集中的坐标维度
    #得到n=2
    n = shape(dataSet)[1]
    #初始化为一个(k,n)的矩阵保存随机生成的k个质心
    centroids = mat(zeros((k,n)))
    #遍历数据集的每一维度
    for j in range(n):
        #获取该列的最小值
        minJ = min(dataSet[:,j])
        #获取该列的最大值
        maxJ = max(dataSet[:,j])
        #得到该列数据的范围(最大值-最小值)
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids


""" 
函数说明：
       k-means 聚类算法
       该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
       这个过程重复数次，直到数据点的簇分配结果不再改变位置。 
"""
def kMeans(dataSet, k, distMeas=distSlC, createCent=randCent):
    #返回行数
    m = shape(dataSet)[0]
    #创建一个与dataSet行数一样，但是有两列的m*2矩阵，用来保存簇分配结果
    clusterAssment = mat(zeros((m, 2)))
    #createCent=randCent,既然都使用randCent了，那就是创建随机k个质心
    centroids = createCent(dataSet, k)
    #聚类结果是否发生变化的布尔类型
    clusterChanged = True
    while clusterChanged:
        #聚类结果是否发生变化置为false
        clusterChanged = False
        #遍历数据集每一个样本向量
        #循环每一个数据点并分配到最近的质心中去
        for i in range(m):
            #初始化最小距离最正无穷
            minDist = inf
            #最小距离对应索引为-1
            minIndex = -1
            #依次循环k个质心
            for j in range(k):
                #计算某个数据点到质心的距离，欧式距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                #如果距离比minDist(最小距离)还小
                if distJI < minDist:
                    #更新minDist(最小距离)和最小质心的index(索引)
                    minDist = distJI
                    minIndex = j
            #当前聚类结果中第i个样本的聚类结果发生变化：
            if clusterAssment[i, 0] != minIndex:
                #布尔类型置为true，继续聚类算法
                clusterChanged = True
                #更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
                clusterAssment[i, :] = minIndex,minDist**2
            #打印k-均值聚类的质心
            print(centroids)
        #遍历每一个质心
        for cent in range(k):
            #获取该簇中的所有点
            #将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]
            #将质心修改为簇中所有点的平均值，mean 就是求平均值的
            #计算这些数据的平均值(axis=0:求列的均值)，作为该类质心向量
            centroids[cent,:] = mean(ptsInClust, axis=0)
    #返回k个聚类，聚类结果与误差
    return centroids, clusterAssment


"""
函数说明：
       二分k-均值聚类算法
"""
def biKmeans(dataSet, k, distMeas=distSlC):
    # 返回行数
    m = shape(dataSet)[0]
    # m x 2 列的矩阵 用来存储每个样本簇系数和误差值
    clusterAssment = mat(zeros((m, 2)))
    # 获取各列的均值
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 存储簇质心
    centList = [centroid0]
    # 遍历每个数据样本 返回每个样本到指定簇中心的距离
    for j in range(m):
        # 计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    # 循环，直至二分k-均值达到k类为止
    while (len(centList) < k):
        # 将当前最小平方误差置为正无穷
        lowestSSE = inf
        # 遍历当前每个聚类
        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算该类划分后两个类的误差平方和
            sseSplit = sum(splitClustAss[:, 1])
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                # 第i类作为本次划分类
                bestCentToSplit = i
                # 第i类划分后得到的两个质心向量
                bestNewCents = centroidMat
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss = splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowestSSE = sseSplit + sseNotSplit
                # 数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为 当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号连续不出现空缺
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # 更新质心列表中的变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加新的类的质心向量
        centList.append(bestNewCents[1, :].tolist()[0])
        # 更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    # 返回聚类结果
    return mat(centList), clusterAssment


"""
函数说明：
      聚类及簇绘图函数
      打开places.txt文件获取第4列和第5列，分别对应纬度和经度。
      基于这些经纬度对的列表创建一个矩阵。
      在这些数据点上运行biKmeans()
      使用distSLC()函数作为聚类中使用的距离计算方法。
      最后将簇以及簇质心画在图上
"""
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSlC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    #看每个人的文本存在哪里
    imgP = plt.imread(r'.\Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ == '__main__':
    clusterClubs(5)
