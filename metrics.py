"""
to compute certain financial metrics from prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

def calculate_ema(arr, window, alpha):
    ema = [sum(arr[:window]) / window]
    for x in arr[window:]:
        #ema.append(x * alpha / (1 + window) + ema[-1] * (1 - alpha / (1 + window)))
        ema.append(ema[-1] + alpha * (x - ema[-1]))
    return ema


def calcBollinger(prices, window, nstds):
    """
    calculate bolinger bands for a lookback period
    :param prices: prices, numpy list or dataframe
    :param lookback: rolling window
    :param nstds: how many standard deviations below or above the mean
    :return:
    """
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    lower = mean - nstds*std
    upper = mean + nstds*std
    return lower, upper

def calcRSI(prices, window):
    """
    calculate relative strength indices for every period
    :param prices: prices
    :param window: how long the period is
    :return:
    """
    diff = np.diff(prices)
    U = np.maximum(0, diff)
    D = np.maximum(0, -diff)
    smmaU = calculate_ema(U, window, 1 / window)
    smmaD = calculate_ema(D, window, 1 / window)
    RS = np.array(smmaU) / np.array(smmaD)
    RSI = 100 - 100 / (1 + RS)
    return RSI


