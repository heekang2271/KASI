import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd

# 파일 리스트 생성
def getFileList(path, removeNum):
    fileList = os.listdir(path)
    fileList = fileList[4:]

    header = (fits.open(path + "/" + fileList[0]))[0].header["NAXIS2"], \
             (fits.open(path + "/" + fileList[0]))[0].header["NAXIS1"], \
             (fits.open(path + "/" + fileList[0]))[0].header["EXPTIME"], \
             (fits.open(path + "/" + fileList[len(fileList)-1]))[0].header["EXPTIME"]


    if removeNum != 0:
        if "Light" in path :
            fileList = removeJuckFile(fileList, removeNum)
        else:
            fileList = removeJuckFile(fileList, len(fileList)//removeNum)

    return fileList, header


# 파일 리스트에서 필요없는 파일 제거
# num은 노출시간당 촬영 횟수
def removeJuckFile(fileList, num):
    newFileList = []

    for i in range(len(fileList)//num):
        if i % 2 == 1:
            newFileList += fileList[i*num : i*num + num]

    return newFileList

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
        meanBiasData = np.mean(biasDataList)

        lightDataList = getData(self.lightFileList, self.lightPath, x = x, y = y, roi = self.roi)

        for i in range(len(lightDataList)):
            lightDataList[i] = (lightDataList[i]) - meanBiasData

        for i in range(1, len(lightDataList), 2):
            diffData = lightDataList[i] - lightDataList[i-1]
            average = (lightDataList[i] + lightDataList[i-1]) / 2.0
            variance.append(np.var(diffData) / 2.0)
            signal.append(np.mean(average))

        exp = [i for i in range(int(self.lightFileHeader[2]), int(self.lightFileHeader[3] + 1))]

        return variance, signal, exp

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
        full_well = round(signal[idx], 4)
        S = "Max = (" + str(round(max_x, 4)) + ", " + str(round(max_y, 4)) + ")\n"
        S += "Full Well Capacity : " + str(full_well)
        plt.text(logScaleSignal[0], 0.0, S, size=15)
        plt.grid()

        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95, wspace=0.8, hspace=0.4)
        plt.show()


class CMOSCamera :
    def __init__(self, biasPath, lightPath, removeNum):
        self.biasFileList, self.biasFileHeader = getFileList(biasPath, removeNum)
        self.lightFileList, self.lightFileHeader = getFileList(lightPath, removeNum)
        self.col = self.lightFileHeader[1]
        self.row = self.lightFileHeader[0]
        self.biasPath = biasPath
        self.lightPath = lightPath

        self.run()

    def run(self):
        self.dataProcessing(200)

    def dataScan(self, iteration):
        t = time.time()
        biasDataList = getData(self.biasFileList, self.biasPath)
        meanBiasDataList = np.mean(biasDataList, axis = 0)
        expStart = int(self.lightFileHeader[2])
        expEnd = int(self.lightFileHeader[3])

        varDataList = np.zeros((expEnd - expStart + 1, self.row, self.col))
        stdDataList = np.zeros((expEnd - expStart + 1, self.row, self.col))
        meanDataList = np.zeros((expEnd - expStart + 1, self.row, self.col))
        print("바이어스 처리 : ", time.time() - t)
        for i in range(0, expEnd - expStart + 1):
            start = time.time()

            t = time.time()
            minusLightDataList = np.zeros((iteration//2, self.row, self.col), dtype=np.float32)
            meanLightDataList = np.zeros((iteration//2, self.row, self.col), dtype=np.float32)
            sumMinusDataList = np.zeros((1, self.row, self.col), dtype=np.float32)
            sumMeanDataList = np.zeros((1, self.row, self.col), dtype=np.float32)
            print("제로데이터 생성 : ", time.time() - t)

            t = time.time()
            for j in range(i*iteration, i*iteration + iteration, 2):
                data1 = (fits.open(self.lightPath + "/" + self.lightFileList[j]))[0].data
                data2 = (fits.open(self.lightPath + "/" + self.lightFileList[j+1]))[0].data
                data1 = np.array(data1, dtype=np.int32)
                data2 = np.array(data2, dtype=np.int32)
                idx = j//2
                minusLightDataList[idx] = np.abs((data2 - data1)) / 2
                meanLightDataList[idx] = (data1 + data2) / 2 - meanBiasDataList
                sumMeanDataList += meanLightDataList[idx]
                sumMinusDataList += minusLightDataList[idx]
            print("데이터 처리 : ", time.time() - t)
            '''
            # 노출시간 별, 모든 픽셀마다 var, std, mean값 생성
            v = np.var(minusLightDataList, axis = 0)
            varDataList[i] = np.var(minusLightDataList, axis = 0)
            stdDataList[i] = np.std(minusLightDataList) # 라이브러리 안쓰고 생성된 var 사용해 시간 약 200초 단축
            meanDataList[i] = np.mean(meanLightDataList, axis = 0)
            '''
            t = time.time()
            # 평균
            meanDataList[i] = sumMeanDataList / (iteration//2)

            # 분산
            varData = np.zeros((1, self.row, self.col), dtype=np.float32)
            minusMean = sumMinusDataList / (iteration//2)
            for k in range(len(minusLightDataList)):
                var = minusLightDataList[k] - minusMean
                var *= var
                varData += var
            varData /= (iteration//2)
            varDataList[i] = varData

            # 표준편차
            stdDataList[i] = np.sqrt(varDataList[i])
            print("평균, 분산, 표준편차 : ", time.time() - t)
            print('#(%d/%d) : %f' %(i, expEnd - expStart + 1, time.time() - start))

        return varDataList, stdDataList, meanDataList

    def dataProcessing(self, slopeRatio):
        varList, stdList, meanList = self.dataScan(200)

        linearity = np.zeros((self.row, self.col))
        conversionGain = np.zeros((self.row, self.col))
        ADU_Max = np.zeros((self.row, self.col))
        FullWell = np.zeros((self.row, self.col))
        LogMeanStdSlope = np.zeros((self.row, self.col))

        for i in range(self.row):
            for j in range(self.col):
                sepVarList = varList[:, i, j]
                sepStdList = stdList[:, i, j]
                sepMeanList = meanList[:, i, j]

                for k in range(2, len(varList)):
                    slopeLeft = (np.polyfit(sepMeanList[:k], sepVarList[:k], 1))[0]
                    slopeRight = (np.polyfit(sepMeanList[:k + 1], sepVarList[:k + 1], 1))[0]

                    if np.abs(slopeRight / slopeLeft - 1) <= slopeRatio:
                        continue
                    else:
                        linearity[i, j] = k
                        conversionGain[i, j] = 1 / slopeLeft
                        LogMeanStdSlope[i, j] = (np.polyfit(np.log10(sepMeanList[:k + 1]), np.log10(sepStdList[:k + 1]), 1))[0]
                        ADU_Max[i, j] = sepMeanList[sepVarList.argmax()]
                        FullWell[i, j] = conversionGain[i, j] * ADU_Max[i, j]
                        break

        return linearity, conversionGain, ADU_Max, FullWell, LogMeanStdSlope

if __name__ == '__main__':
    #ccd = CCDCamera("EM003\Bias_1ms", "EM003\Light", 50, 2)
    cmos = CMOSCamera("Sample_Image\Dark_3s20200810", "Sample_Image\Light_3s20200810", 0)