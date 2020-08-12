import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import time

# 파일 리스트 생성
def getFileList(path):
    fileList = os.listdir(path)
    fileList = fileList[4:]

    header = (fits.open(path + "/" + fileList[0]))[0].header["NAXIS2"], \
             (fits.open(path + "/" + fileList[0]))[0].header["NAXIS1"], \
             (fits.open(path + "/" + fileList[0]))[0].header["EXPTIME"], \
             (fits.open(path + "/" + fileList[len(fileList)-1]))[0].header["EXPTIME"]

    '''
    if "Light" in path :
        fileList = removeJuckFile(fileList, 2)
    else:
        fileList = removeJuckFile(fileList, len(fileList)//2)
    '''
    return fileList, header


# 파일 리스트에서 필요없는 파일 제거
# num은 노출시간당 촬영 횟수
def removeJuckFile(fileList, num):
    newFileList = []

    for i in range(len(fileList)//num):
        if i % 2 == 1:
            newFileList += fileList[i*num : i*num + num]

    return newFileList


class CCDCamera :
    def __init__(self, biasPath, lightPath):
        self.biasFileList, self.biasFileHeader = getFileList(biasPath)
        self.lightFileList, self.lightFileHeader = getFileList(lightPath)
        self.col = self.lightFileHeader[1]
        self.row = self.lightFileHeader[0]
        self.biasPath = biasPath
        self.lightPath = lightPath

        self.run()

    def run(self):
        variance, signal, exp = self.dataProcessing()
        self.drawGraph(variance, signal, exp)

    def roiLocation(self, row, col, roi):
        return row//2 - roi , row//2 + roi, col//2 - roi, col//2 + roi

    def dataGeneration(self, fileList, path, x1, x2, y1, y2):
        dataList = []

        for file in fileList:
            data = (fits.open(path + "/" + file))[0].data
            dataList.append(data[y1:y2, x1:x2])

        return dataList

    def dataProcessing(self):
        variance = []
        signal = []
        x1, x2, y1, y2 = self.roiLocation(self.row, self.col, 50)

        biasDataList = self.dataGeneration(self.biasFileList, self.biasPath, 0, self.col, 0, self.row)
        biasDataList = np.array(biasDataList)
        meanBiasData = np.mean(biasDataList)

        lightDataList = self.dataGeneration(self.lightFileList, self.lightPath, x1, x2, y1, y2)

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
    def __init__(self, biasPath, lightPath):
        self.biasFileList, self.biasFileHeader = getFileList(biasPath)
        self.lightFileList, self.lightFileHeader = getFileList(lightPath)
        self.col = self.lightFileHeader[1]
        self.row = self.lightFileHeader[0]
        self.biasPath = biasPath
        self.lightPath = lightPath

        self.run()

    def run(self):
        self.dataScan(200)

    def dataGeneration(self, fileList, path, start, end):
        sepFileList = fileList[start:end]
        dataList = []

        for file in sepFileList:
            data = (fits.open(path + "/" + file))[0].data
            dataList.append(data)

        return dataList

    def dataScan(self, iteration):
        start = time.time()
        biasDataList = self.dataGeneration(self.biasFileList, self.biasPath, 0, len(self.biasFileList))
        biasDataList = np.array(biasDataList)
        meanBiasDataList = np.mean(biasDataList, axis = 0)
        biasDataList = []
        expStart = int(self.lightFileHeader[2])
        expEnd = int(self.lightFileHeader[3])

        varDataList = []
        stdDataList = []
        meanDataList = []

        for i in range(0, expEnd - expStart + 1):
            minusLightDataList = np.zeros((int(iteration/2), self.row, self.col)).astype(np.float32)
            meanLightDataList = np.zeros((int(iteration/2), self.row, self.col)).astype(np.float32)

            for j in range(i*iteration, i*iteration + iteration, 2):
                data1 = (fits.open(self.lightPath + "/" + self.lightFileList[j]))[0].data
                data2 = (fits.open(self.lightPath + "/" + self.lightFileList[j+1]))[0].data
                data1 = np.array(data1).astype(np.int32)
                data2 = np.array(data2).astype(np.int32)
                idx = int(j/2)
                minusLightDataList[idx] = np.abs((data2 - data1)) / 2
                meanLightDataList[idx] = (data1 + data2) / 2 - meanBiasDataList
                data1 = []
                data2 = []

            # 노출시간 별, 모든 픽셀마다 var, std, mean값 생성
            v = np.var(minusLightDataList, axis = 0)
            varDataList.append(v)
            stdDataList.append(np.sqrt(v))
            meanDataList.append(np.mean(meanLightDataList, axis = 0))

        print("총 : ", time.time() - start)


if __name__ == '__main__':
    #ccd = CCDCamera("EM003\Bias_1ms", "EM003\Light")
    cmos = CMOSCamera("Sample_Image\Dark_3s20200810", "Sample_Image\Light_3s20200810")