import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from abc import ABC, abstractmethod

nInst=100
currentPos = np.zeros(nInst)

def getMyPosition (prcSoFar):
    global currentPos

    """
    mean reversion using bollinger bands
    """
    prcSoFar = pd.DataFrame(prcSoFar)
    period = 14
    # if there are no prices so far just hodl
    if prcSoFar.shape[0] < 14:
        return currentPos

    # if there are prices calculate one standard deviation bolinger bands
    lowers, uppers = metrics.calcBollinger(prcSoFar, period, 1)
    lower_bound = np.array(lowers.iloc[-1])
    upper_bound = uppers.iloc[-1]

    # buy stuff that is below the lower bound, sell stuff that is above the upper bound
    to_buy = prcSoFar.iloc[-1] < lower_bound
    to_sell = prcSoFar.iloc[-1] > upper_bound

    # ok now we do buying and selling, is there a way to vectorize/optimize this shit
    # also how much od we buy and sell
    for i in range(nInst):
        if to_buy[i]:
            # idk buy 10 instruments why not
            currentPos[i] += 10
        if to_sell[i]
            currentPos[i] -= 10

    return currentPos


