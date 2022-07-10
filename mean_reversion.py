import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from abc import ABC, abstractmethod

nInst=100
currentPos = np.zeros(nInst)

def getMyPosition4(prcSoFar):
    global currentPos

    prcSoFar = pd.DataFrame(prcSoFar.T)
    period = 28

    if prcSoFar.shape[0] < period:
        return currentPos

    rsi = prcSoFar.apply(lambda x: metrics.calcRSI(x, period))

    for i in range(nInst):
        # if i > 0:
        #     continue
        if rsi.iloc[-1][i] == 0.:
            continue

        x = (rsi.iloc[-1][i] - 50) / 50
        currentPos[i] -= 100 * (1 / (1 + np.exp(10 * x)) - 0.5)

    return currentPos

def getMyPosition3(prcSoFar, period, c1):
    global currentPos

    prcSoFar = pd.DataFrame(prcSoFar.T)

    if prcSoFar.shape[0] < period:
        return currentPos

    rsi = prcSoFar.apply(lambda x: metrics.calcRSI(x, period))

    #print("stock 0 rsi:", rsi.iloc[-1][0])

    exposure = np.sum(currentPos)

    for i in range(nInst):
        # if i > 0:
        #     continue
        if rsi.iloc[-1][i] == 0.:
            continue


        if rsi.iloc[-1][i] < 30:
            currentPos[i] += np.floor(10 * c1 * (30 - rsi.iloc[-1][i]))
        if rsi.iloc[-1][i] > 70:
            currentPos[i] -= np.floor(10 * c1 * (rsi.iloc[-1][i] - 70))

    return currentPos

def getMyPosition2(prcSoFar):
    global currentPos

    """
    mean reversion using bollinger bands
    """
    prcSoFar = pd.DataFrame(prcSoFar.T)
    period = 28
    # if there are no prices so far just hodl
    if prcSoFar.shape[0] < period:
        return currentPos

    # if there are prices calculate one standard deviation bolinger bands
    lowers05, uppers05 = metrics.calcBollinger(prcSoFar, period, 0.5)
    lowers1, uppers1 = metrics.calcBollinger(prcSoFar, period, 1)
    lowers15, uppers50 = metrics.calcBollinger(prcSoFar, period, 1.5)

    for i in range(nInst):
        if prcSoFar.iloc[-1][i] < lowers15.iloc[-1][i]:
            currentPos[i] -= np.random.randint(150, 250)
        elif (prcSoFar.iloc[-1][i] > lowers15.iloc[-1][i]) & (prcSoFar.iloc[-1][i] < lowers1.iloc[-1][i]):
            currentPos[i] -= np.random.randint(100, 150)
        elif (prcSoFar.iloc[-1][i] > lowers1.iloc[-1][i]) & (prcSoFar.iloc[-1][i] < lowers05.iloc[-1][i]):
            currentPos[i] -= np.random.randint(0, 50)

        if prcSoFar.iloc[-1][i] > uppers50.iloc[-1][i]:
            currentPos[i] += np.random.randint(150, 250)
        elif (prcSoFar.iloc[-1][i] < uppers50.iloc[-1][i]) & (prcSoFar.iloc[-1][i] > uppers1.iloc[-1][i]):
            currentPos[i] += np.random.randint(100, 200)
        elif (prcSoFar.iloc[-1][i] < uppers1.iloc[-1][i]) & (prcSoFar.iloc[-1][i] > uppers05.iloc[-1][i]):
            currentPos[i] += np.random.randint(0, 50)

    return currentPos

def getMyPosition(prcSoFar):
    global currentPos

    """
    mean reversion using bollinger bands
    """
    prcSoFar = pd.DataFrame(prcSoFar.T)
    period = 14
    # if there are no prices so far just hodl
    if prcSoFar.shape[0] < period:
        return currentPos

    mean = prcSoFar.rolling(period).mean()
    previousPosition = currentPos.copy()
    for i in range(nInst):
        if prcSoFar.iloc[-1][i] < mean.iloc[-1][i]:
            # go long
            if previousPosition[i] < 0:
                currentPos[i] = 0
            else:
                currentPos[i] += 1
        else:
            # go short
            if previousPosition[i] > 0:
                currentPos[i] = 0
            else:
                currentPos[i] -= 1



    return currentPos
