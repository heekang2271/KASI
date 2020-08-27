import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from collections import Counter
import warnings

warnings.filterwarnings(action = 'ignore') # 기울기 구할 때 경고메세지 무시
mpl.rcParams['agg.path.chunksize'] = 10000 # matplotlib 데이터 크기 제한을 없애줌

# 파일 리스트 생성
def getFileList(path, removeNum = 0):
    fileList = os.listdir(path)

    # 이 부분 수정필요,, 파일 리스트에서 각도별로 찍은 폴더들 제외하기 위함
    if fileList[0] == "000":
        fileList = fileList[4:]

    header = (fits.open(path + "/" + fileList[0]))[0].header["NAXIS2"], \
             (fits.open(path + "/" + fileList[0]))[0].header["NAXIS1"], \
             (fits.open(path + "/" + fileList[0]))[0].header["EXPTIME"], \
             (fits.open(path + "/" + fileList[len(fileList)-1]))[0].header["EXPTIME"]


    if removeNum != 0:
        fileList = removeJunkFile(fileList, removeNum)

    return fileList, header


# 파일 리스트에서 필요없는 파일 제거
# num은 촬영 횟수 당 지울 파일의 수
def removeJunkFile(fileList, num):
    newFileList = []

    for i in range(len(fileList)//num):
        if i % 2 == 1:
            newFileList += fileList[i*num : i*num + num]

    return newFileList

# CCD, CMOS에서 다 사용하려 만들었지만 CMOS 데이터 조건상 이 함수를 사용하면 메모리가 터져 CCD에서만 사용
def getData(fileList, path, start = 0, end = None, x = None, y = None, roi = None):
    sepFileList = fileList[start:end]
    dataList = []

    for file in sepFileList:
        data = (fits.open(path + "/" + file))[0].data
        if roi != None:
            data = data[y:y+2*roi, x:x+2*roi]
        dataList.append(data)

    return dataList

class CCDCamera :
    def __init__(self, biasPath, lightPath, roi, removeNum):
        self.biasFileList, self.biasFileHeader = getFileList(biasPath, removeNum)
        self.lightFileList, self.lightFileHeader = getFileList(lightPath, removeNum)
        self.col = self.lightFileHeader[1]
        self.row = self.lightFileHeader[0]
        self.roi = roi
        self.biasPath = biasPath
        self.lightPath = lightPath

        self.run()

    def run(self):
        variance, signal, exp = self.dataProcessing()
        self.drawGraph(variance, signal, exp)

    def dataProcessing(self):
        variance = []
        signal = []

        x = self.col // 2 - self.roi
        y = self.row // 2 - self.roi

        biasDataList = getData(self.biasFileList, self.biasPath, x = x, y = y, roi = self.roi)
        meanBiasData = np.mean(biasDataList, axis = 0)

        lightDataList = getData(self.lightFileList, self.lightPath, x = x, y = y, roi = self.roi)

        for i in range(len(lightDataList)):
            lightDataList[i] = (lightDataList[i]) - meanBiasData

        for i in range(1, len(lightDataList), 2):
            diffData = lightDataList[i] - lightDataList[i-1]
            average = (lightDataList[i] + lightDataList[i-1]) / 2.0
            variance.append(np.var(diffData) / np.sqrt(2))
            signal.append(np.mean(average))

        exp = [i for i in range(int(self.lightFileHeader[2]), int(self.lightFileHeader[3] + 1))]

        return variance, signal, exp

    # 추세선 기울기로 FullWell에 해당하는 점을 구하는 함수
    def getFullWellPoint(self, variance, signal, ratio):
        for i in range(2, len(variance)):
            slopeLeft = (np.polyfit(signal[:i], variance[:i], 1))[0]
            slopeRight = (np.polyfit(signal[:i+1], variance[:i+1], 1))[0]

            if np.abs(slopeRight / slopeLeft - 1) <= ratio:
                continue
            else:
                return i

    def drawGraph(self, variance, signal, exp):
        logScaleTemporalNoise = np.log10(np.sqrt(np.array(variance) / 2.0))
        logScaleSignal = np.log10(signal)
        point = self.getFullWellPoint(np.log10(variance), logScaleSignal, 0.02)


        plt.figure(figsize=(15, 12))

        # ADU vs Exp
        plt.subplot(221)
        plt.plot(exp, signal, marker='o', color='r')
        plt.title('ADU vs Exp.Time', size=18)
        plt.xlabel('Exposure Time [sec]', size=15)
        plt.ylabel('ADU', size=15)
        plt.grid()

        # Signal vs Variance
        plt.subplot(222)
        plt.plot(signal[:point], variance[:point], 'ro')
        plt.title('Signal vs Variance', size=18)
        plt.xlabel('Average Signal - offset [ADU]', size=15)
        plt.ylabel('Variance', size=15)
        fit = np.polyfit(signal[:point], variance[:point], 1)
        conversionGain = 1 / fit[0]
        P1 = np.poly1d(fit)
        plt.plot(signal[:point], P1(signal[:point]), 'r')
        S = str(round(fit[0], 2)) + 'X + ' + str(round(fit[1], 2))
        plt.text(np.mean(signal[:point]), np.mean(P1(signal[:point])) - 250, S, size=15)
        plt.grid()

        # Log(Average Signal - offset) vs Log(Temporal Noise)
        plt.subplot(223)
        plt.plot(logScaleSignal[:point], logScaleTemporalNoise[:point], 'ro')
        plt.title('Log(Average Signal - offset) vs Log(Temporal Noise)', size=18)
        plt.xlabel('Log(Average Signal - offset)[ADU]', size=15)
        plt.ylabel('Log(Temporal Noise)[ADU]', size=15)
        fit2 = np.polyfit(logScaleSignal[:point], logScaleTemporalNoise[:point], 1)
        P2 = np.poly1d(fit2)
        plt.plot(logScaleSignal[:point], P2(logScaleSignal[:point]), 'r')
        S = str(round(fit2[0], 2)) + 'X + ' + str(round(fit2[1], 2))
        plt.text(np.mean(logScaleSignal[:point]), np.mean(P2(logScaleSignal[:point]) - 0.1), S, size=15)
        plt.grid()

        # FullWellCapacity
        plt.subplot(224)
        plt.plot(logScaleSignal, logScaleTemporalNoise, 'rx')
        plt.title('Full Well Capacity [e-]', size=18)
        plt.xlabel('Log(Average Signal - offset)[ADU]', size=15)
        plt.ylabel('log(Temporal Noise)[ADU]', size=15)
        temp = list(logScaleTemporalNoise)
        idx = temp.index(max(logScaleTemporalNoise))
        max_x = logScaleSignal[idx]
        max_y = logScaleTemporalNoise[idx]
        x = [max_x, max_x]
        y = [max_y * 0.5, max_y * 1.5]
        plt.plot(x, y, '-r', LineWidth=3)
        full_well = np.around(signal[idx] * conversionGain, 4)
        S = "Max = (" + str(round(max_x, 4)) + ", " + str(round(max_y, 4)) + ")\n"
        S += "Full Well Capacity : " + str(full_well)
        plt.text(logScaleSignal[0], 0.0, S, size=15)
        plt.grid()

        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95, wspace=0.8, hspace=0.4)
        plt.show()

class CMOSCamera :
    def __init__(self, biasPath, lightPath, removeNum):
        self.biasFileList, self.biasFileHeader = getFileList(biasPath, 10)
        self.lightFileList, self.lightFileHeader = getFileList(lightPath, removeNum)
        self.col = self.lightFileHeader[1] # 헤더에서 col 정보
        self.row = self.lightFileHeader[0] # 헤더에서 row 정보
        self.expNum = int((self.lightFileHeader[3] - self.lightFileHeader[2] + 1)) # 헤더에서 노출시간을 통해 노출시간 몇 구간인지 계산
        self.biasPath = biasPath
        self.lightPath = lightPath
        self.iteration = len(self.lightFileList) // self.expNum # 노출시간당 몇 장 찍었는지 계산

        self.run()

    def run(self):
        ratio = 0.02 # 추세선 기울기 오차 범위
        self.dataProcessing(ratio) # conversionGain, fullWell을 구하는 함수
        #self.drawGraph()

    def dataScan(self):
        biasDataList = getData(self.biasFileList, self.biasPath)
        meanBiasDataList = np.mean(biasDataList, axis = 0)
        iter = self.iteration

        # 타입을 .astype 으로 바꾸는 것보다 numpy 선언하면서 dtype으로 타입을 주면 훨씬 빠름
        varDataList = np.zeros((self.expNum, self.row, self.col), dtype=np.float32)
        stdDataList = np.zeros((self.expNum, self.row, self.col), dtype=np.float32)
        meanDataList = np.zeros((self.expNum, self.row, self.col), dtype=np.float32)

        for i in range(0, self.expNum):
            start = time.time()

            minusLightDataList = np.zeros((iter//2, self.row, self.col), dtype=np.float32)
            meanLightDataList = np.zeros((iter//2, self.row, self.col), dtype=np.float32)
            sumMinusDataList = np.zeros((1, self.row, self.col), dtype=np.float32) # 분산을 구하는 시간을 줄이고자
            sumMeanDataList = np.zeros((1, self.row, self.col), dtype=np.float32) # 평균을 구하는 시간을 줄이고자

            for j in range(i*iter, i*iter + iter, 2):
                # 사진을 두 장씩 불러와 서로의 차이값과 평균값을 구하며, 그 값들의 합을 구해 함수 사용을 하지 않고 분산과 평균을 빠르게 구함
                data1 = (fits.open(self.lightPath + "/" + self.lightFileList[j]))[0].data - meanBiasDataList
                data2 = (fits.open(self.lightPath + "/" + self.lightFileList[j+1]))[0].data - meanBiasDataList

                idx = (j%iter) // 2
                minusLightDataList[idx] = (data2 - data1) / np.sqrt(2)
                meanLightDataList[idx] = (data1 + data2) / 2
                sumMeanDataList += meanLightDataList[idx]
                sumMinusDataList += minusLightDataList[idx]

            # 평균
            meanDataList[i] = sumMeanDataList / (iter//2)

            # 분산
            varData = np.zeros((1, self.row, self.col), dtype=np.float32)
            minusMean = sumMinusDataList / (iter//2)
            for k in range(len(minusLightDataList)):
                var = minusLightDataList[k] - minusMean
                var *= var
                varData += var
            varData /= (iter//2)
            varDataList[i] = varData

            # 표준편차
            #stdDataList[i] = np.sqrt(varDataList[i])

            runTime = time.time() - start
            m = runTime // 60
            s = int(runTime - m * 60)
            print('#(%d/%d) : %d분 %d초' %(i+1, self.expNum, m, s))

        return varDataList, meanDataList

    def dataProcessing(self, slopeRatio):
        start = time.time()
        varList, meanList = self.dataScan()

        conversionGain = np.zeros((self.row, self.col), dtype=np.float32)
        FullWell = np.zeros((self.row, self.col), dtype=np.float32)

        deadPixel = []

        for i in range(self.row):
            runTime = time.time()
            for j in range(self.col):
                sepVarList = varList[:, i, j]
                sepMeanList = meanList[:, i, j]

                # dead pixel 발견시 건너뜀과 동시에 위치 저장
                if np.sum(sepVarList) == 0:
                    deadPixel.append([i+1, j+1])
                    continue

                # 반복시 polyfit 함수 한 번만 호출하기 위해 새로운 기울기와 값을 비교 후, 이전 기울기에 저장하는 방식으로 진행
                slopeLeft = (np.polyfit(sepMeanList[:2], sepVarList[:2], 1))[0]
                for k in range(3, self.expNum + 1):
                    slopeRight = (np.polyfit(sepMeanList[:k], sepVarList[:k], 1))[0]

                    if np.abs(slopeRight / slopeLeft - 1) <= slopeRatio:
                        slopeLeft = slopeRight
                        continue
                    else:
                        # 컨버전게인 튀는값 확인하기 위한 테스트용
                        # if slopeLeft > 10 or slopeLeft < 0:
                        #     print("debug")

                        conversionGain[i, j] = 1 / slopeLeft
                        ADU_Max = sepMeanList[sepVarList.argmax()]
                        FullWell[i, j] = conversionGain[i, j] * ADU_Max
                        break

                # 샘플데이터 테스트용 1
                # 분산이 최대가 되는 지점을 찾아 거기까지만 추세선 기울기를 구함
                # idx = sepVarList.argmax()
                # slope = np.polyfit(sepMeanList[:idx+1], sepVarList[:idx+1], 1)[0]
                # conversionGain[i, j] = 1 / slope
                # ADU_Max = sepMeanList[idx]
                # FullWell[i, j] = conversionGain[i, j] * ADU_Max

                # 샘플데이터 테스트용 2
                # 분산이 감소하기 시작하는 지점을 찾아 거기까지만 추세선 기울기를 구함
                # for k in range(0, len(sepVarList)-2):
                #     if sepVarList[k] >= sepVarList[k+1] and sepVarList[k+1] >= sepVarList[k+2]:
                #         slope = np.polyfit(sepMeanList[:k+1], sepVarList[:k+1], 1)[0]
                #         conversionGain[i, j] = 1 / slope
                #         ADU_Max = sepMeanList[sepVarList.argmax()]
                #         FullWell[i, j] = conversionGain[i, j] * ADU_Max

            runTime = time.time() - runTime
            print('(%d/%d) : %f' %(i+1, self.row, runTime))

        start = time.time() - start
        m = start // 60
        s = int(start - m*60)
        print('종료 : %d분 %d초' %(m, s))
        print("deadPixel : ", deadPixel)

        # 엑셀 파일로 저장
        df = pd.DataFrame(conversionGain)
        df.to_csv('conversionGain.csv')
        df = pd.DataFrame(FullWell)
        df.to_csv('FullWell.csv')

        return conversionGain, FullWell

    def drawGraph(self):
        conversionGain = pd.read_csv('conversionGain.csv')
        conversionGain = np.array(conversionGain)[:, 1:]
        fullWell = pd.read_csv('FullWell.csv')
        fullWell = np.array(fullWell)[:, 1:]

        conversionGain[1935, 1040] = 0
        fullWell[1935, 1040] = 0

        # 빈도수 그래프
        # fullwell
        fullWell = np.array(fullWell, dtype=np.int32)[:, 1:]
        x = np.ravel(fullWell, order = 'C')

        # 데이터들을 int형으로 바꿔 데이터별 빈도수가 유효하게 함
        count = Counter(x)
        count = sorted(count.items())
        count = np.array(count)

        x = count[:, 0]
        x = x[1:]
        y = count[:, 1]
        y = y[1:]

        plt.figure(figsize=(13, 8))
        plt.plot(x, y, color='r')
        plt.title("Frequency(FullWell)", size = 18)
        plt.xlabel("FullWell", size = 15)
        plt.ylabel("Frequency", size = 15)

        line_x = x[np.argmax(y)]
        line_x = [line_x, line_x]
        line_y = [0, np.max(y)]
        plt.plot(line_x, line_y, '-r', LineWidth=3)
        S = "x = " + str(line_x[0])
        plt.text(line_x[0] * 1.1, np.mean(line_y), S, size = 15)
        plt.show()

        # conversionGain
        conversionGain = np.array(conversionGain)[:, 1:]
        conversionGain = np.round(conversionGain, 4)
        x = np.ravel(conversionGain, order='C')

        # 데이터들을 소숫점 4번째 자리까지만 나타내도록 바꿔 데이터별 빈도수가 유효하게 함
        count = Counter(x)
        count = sorted(count.items())
        count = np.array(count)

        x = count[:, 0]
        x = x[1:]
        y = count[:, 1]
        y = y[1:]

        plt.figure(figsize = (13, 8))
        plt.plot(x, y, color = 'r')
        plt.title("Frequency(ConversionGain)", size = 18)
        plt.xlabel("ConversionGain", size = 15)
        plt.ylabel("Frequency", size = 15)

        line_x = x[np.argmax(y)]
        line_x = [line_x, line_x]
        line_y = [0, np.max(y)]
        plt.plot(line_x, line_y, '-r', LineWidth=3)
        S = "x = " + str(line_x[0])
        plt.text(line_x[0] * 1.1, np.mean(line_y), S, size=15)
        plt.show()

        # 히트맵 그래프
        # FullWell
        meanFullWell = np.mean(fullWell)
        minValue = meanFullWell - meanFullWell * 0.99
        maxValue = meanFullWell + meanFullWell * 0.99
        fullWell = pd.DataFrame(fullWell)
        plt.figure(figsize = (18, 10))
        plt.title("HeatMap(FullWell)", size = 20)
        heatmap = sns.heatmap(fullWell, vmin = minValue, vmax = maxValue)
        plt.show()

        # ConversionGain
        meanCG = np.mean(conversionGain)
        minValue = meanCG - meanCG * 0.99
        maxValue = meanCG + meanCG * 0.99
        conversionGain = pd.DataFrame(conversionGain)
        plt.figure(figsize=(18, 10))
        plt.title("HeatMap(ConversionGain)", size = 20)
        heatmap = sns.heatmap(conversionGain, vmin=minValue, vmax=maxValue)
        plt.show()

        print("end")

class QE :
    def __init__(self, biasPath, QEPath, cal_Pd_Path, chA_path, chB_path, CG_Path, setUp):
        self.biasPath = biasPath
        self.biasFileList, self.biasFileHeader = getFileList(biasPath, 0)
        self.chA_path = chA_path
        self.chB_path = chB_path
        self.QEPath = QEPath
        self.CGPath = CG_Path
        self.QE_FileList, self.QE_FileHeader = getFileList(QEPath, 1)
        self.cal_pd_path = cal_Pd_Path
        self.Exp, self.Pixel_size, self.q = setUp
        self.col = self.QE_FileHeader[1]
        self.row = self.QE_FileHeader[0]

        self.drawGraph()

    def getCalPD(self):
        f = open(self.cal_pd_path, 'r')
        lines = f.readlines()
        cal_pd = []

        for line in lines:
            line = line.rstrip().split("\t")
            line = list(map(float, line))
            cal_pd.append(line)

        cal_pd = np.array(cal_pd)

        lambdas = cal_pd[:, 0]
        cal_pd_chA = cal_pd[:, 1]
        cal_pd_chB = cal_pd[:, 2]

        n_iter = 5
        n_lambda = len(lambdas) // n_iter

        medianLambda = np.median(lambdas.reshape(n_lambda, n_iter), 1).reshape([n_lambda, 1])
        medianChA = np.median(cal_pd_chA.reshape(n_lambda, n_iter), 1).reshape([n_lambda, 1])
        medianChB = np.median(cal_pd_chB.reshape(n_lambda, n_iter), 1).reshape([n_lambda, 1])

        return medianLambda, medianChA, medianChB

    def getChResp(self):
        medianLambda, medianChA, medianChB = self.getCalPD()

        f = open(self.chA_path)
        lines = f.readlines()
        chA_RespAll = []
        index = []

        for line in lines :
            line = line.rstrip().split("\t")
            line = list(map(float, line))
            chA_RespAll.append(line)

        for i in range(len(medianLambda)):
            for j in range(len(chA_RespAll)):
                if medianLambda[i] == chA_RespAll[j][0]:
                    index.append(j)

        f = open(self.chB_path)
        lines = f.readlines()
        chB_RespAll = []

        for line in lines :
            line = line.rstrip().split("\t")
            line = list(map(float, line))
            chB_RespAll.append(line)

        chA_RespAll = np.array(chA_RespAll)
        chB_RespAll = np.array(chB_RespAll)

        chA_RespAll = chA_RespAll[index[0]:index[len(index)-1]+1, 1]
        chB_RespAll = chB_RespAll[index[0]:index[len(index) - 1] + 1, 1]

        return chA_RespAll, chB_RespAll

    def getMeanBiasData(self):
        biasFileList = self.biasFileList
        biasDataList = []

        for file in biasFileList:
            data = fits.open(self.biasPath + "/" + file)[0].data
            biasDataList.append(data)

        meanBiasData = np.mean(biasDataList, axis = 0, dtype=np.float32)

        return meanBiasData

    # QE 데이터 불러오면서 bias평균을 바로 빼줌
    def getQE_DataList(self):
        meanBiasData = self.getMeanBiasData()
        QE_FileList = self.QE_FileList
        QE_DataList = []

        for file in QE_FileList:
            data = fits.open(self.QEPath + "/" + file)[0].data - meanBiasData
            QE_DataList.append(data)

        QE_DataList = np.array(QE_DataList)
        # 윈도우 적용 : 0.935 / 미적용 : 1
        transmission = 1
        QE_DataList /= transmission

        return QE_DataList

    def getQE(self):
        QE_DataList = self.getQE_DataList()
        medianLambda, medianChA, medianChB = self.getCalPD()
        chA_RespAll, chB_RespAll = self.getChResp()
        CG = pd.read_csv(self.CGPath)
        CG = np.array(CG)[:, 1:]

        # 샘플데이터의 아래와 같은 픽셀이 값이 오버되서 제외
        # CG[1935, 1040] = 0

        medianChA = medianChA[:, 0]
        medianChB = medianChB[:, 0]
        medianLambda = medianLambda[:, 0]

        Ad = 1
        Ps = (self.Pixel_size ** 2) * (10 ** -8)
        QEd = (1239 * chB_RespAll) / medianLambda
        Sd = (medianChB * chB_RespAll) / (1.6e-19)


        QE = []
        # 넘파이를 사용, 픽셀별로 읽는것이 아니라 파장대 별 3000*4096 데이터를 한 번에 처리,
        # 최빈값을 구해 파장대 수 만큼 QE값을 저장
        for i in range(len(QE_DataList)):
            Sccd = QE_DataList[i] * CG / self.Exp
            Cal_PD_QE = (Ad * Sccd * QEd[i]) / (Ps * Sd[i])
            Cal_PD_QE = Cal_PD_QE * 100
            Cal_PD_QE = np.ravel(Cal_PD_QE)
            Cal_PD_QE = np.array(Cal_PD_QE, dtype=np.int32)

            count = Counter(Cal_PD_QE)
            key = list(count.keys())
            value = list(count.values())
            idx = np.argmax(value)
            QE.append(key[idx])

        return QE, medianLambda

    def drawGraph(self):
        QE, medianLambda = self.getQE()
        QE = np.array(QE)
        plt.figure(figsize=(10, 5))
        plt.plot(medianLambda, QE, marker = 'o', color = 'r')
        plt.show()

if __name__ == '__main__':

    # ----------------- CCD 측정 및 파라미터 입력 -----------------
    # biasPath = "EM003\Bias_1ms"
    # lightPath = "EM003\Light"
    # roi = 50
    # removeFileNum = 2
    # ccd = CCDCamera(biasPath, lightPath, roi, removeFileNum)
    # ------------------------------------------------------------


    # ----------------- CMOS측정 및 파라미터 입력 -----------------
    # biasPath = "Code_Sample\Dark_1s"
    # lightPath = "Code_Sample\Light"
    # removeFileNum = 0
    #cmos = CMOSCamera(biasPath, lightPath, removeFileNum)
    # ------------------------------------------------------------



    # ----------------- QE측정 및 파라미터 입력 -------------------
    biasPath = "EM005\Dark_20s"
    EM_Path = "EM005\QE"
    calPD_Path = "./EM005/EM005_QE_TEST.txt"
    chA_Path = "./EM005/1007/71650_71580_1007.txt"
    chB_Path = "./EM005/1009/71650_71580_1009.txt"
    CG_Path = "conversionGain.csv";
    Exp = 20
    PixelSize = 3.45
    q = 1.6e-19
    setUp = [Exp, PixelSize, q]

    qe = QE(biasPath, EM_Path, calPD_Path, chA_Path, chB_Path, CG_Path, setUp)
    # ------------------------------------------------------------