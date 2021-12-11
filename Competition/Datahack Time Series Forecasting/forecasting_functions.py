#################################################################################################################################################
#                                                                                                                                               #
#                                                        IMPORTING PACKAGES                                                                     #
#                                                                                                                                               #
#################################################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from scipy import signal
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


#################################################################################################################################################
#                                                                                                                                               #
#                                                           TOOLS FUNCTIONS                                                                     #
#                                                                                                                                               #
#################################################################################################################################################
def sortBy(l1, l2):
    """Sorts l1 with the l2 values"""
    n = len(l1)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if l2[j] < l2[j + 1]:
                l2[j], l2[j + 1] = l2[j + 1], l2[j]
                l1[j], l1[j + 1] = l1[j + 1], l1[j]
    return l1


def plotACF(serie):
    """Plot signal autocorrelation function"""
    # Get autocorrelation function of signal
    f = acf(serie, nlags=len(serie), fft=True)

    indexs = [i for i in range(len(f))]

    # Get local maximas indexs
    maxi = list(argrelextrema(f, np.greater_equal, order=1)[0])

    # Remove start and end : Not really local maximas
    if 0 in maxi:
        maxi.remove(0)
    if indexs[-1] in maxi:
        maxi.remove(indexs[-1])


    plt.plot(indexs, f, label='acf')
    plt.scatter(maxi, [f[m] for m in maxi], c='red', s=4)
    plt.title('autocorrelation function')
    plt.show()

    return maxi
    

def findBestPeriods(df, periods, col='y', show=True, strategy='median'):
    """Shox a graph of each period variance
    Use df[df.index%p].mean() ou df[df.index%p].median() pour calculer la periodicité rapidement"""
    periods = sorted(periods)
    n = len(df[col])

    df_copy = df.copy()
    var = df_copy[col].std()

    ind = []
    rat = []
    for p in periods:
        # Get seasonnal component
        df_copy[str(p)] = df_copy.index%p
        if strategy == 'mean':
            season = df_copy.groupby(df_copy[str(p)]).mean()
        elif strategy == 'median':
            season = df_copy.groupby(df_copy[str(p)]).median()
        
        # Calcul var
        ratio = season[col].std()/var

        # Substract seasonnal component from df_copy
        seasonnal_conponent = (list(season[col])*(int(n/p)+1))[:n]
        ind.append(p)
        rat.append(ratio)
        
        df_copy[col] = df_copy[col]-seasonnal_conponent

    plt.plot(ind, rat)
    plt.title("Periods importance")
    plt.savefig("Period_importance.png")
    plt.show()

    # get best periods
    SortedPeriods = sortBy(ind, rat)
    return SortedPeriods


#################################################################################################################################################
#                                                                                                                                               #
#                                                            PREDICT FUNCTIONS                                                                  #
#                                                                                                                                               #
#################################################################################################################################################    
def predictTrend(signal, model_trend=Ridge(), trend_degree=1, N=1):
    """Get and predict trend of signal"""
    y = signal
    x = [[i] for i in range(len(list(y)))]

    # Interpolate signal with model_trend to get the trend
    model = make_pipeline(PolynomialFeatures(degree=trend_degree), StandardScaler(), model_trend)
    model.fit(x, y)
    trend = model.predict(x)

    # Forecast trend
    pred_trend = model.predict([[i + len(list(y))] for i in range(N)])

    # Move pred_trend to make it coherent
    adjusted_pred_trend = pred_trend+(list(y)[-1]-pred_trend[0])
    
    return trend, pred_trend, adjusted_pred_trend



def extractSeasonnalities(serie, model_seasonnal=RandomForestRegressor(), periods=None, N=1):
    """Get and predict different seasonnalities of signal"""
    y = serie

    # Remove doubles
    periods = list(dict.fromkeys(periods))
    # Sort by ascending order is important
    periods.sort()

    # Extract seasonnalities for each period   
    seasonnalities = {}
    pred_seasonnalities = {}
    y_dt = list(y)
    i = 0
    for p in periods:
        i+=1
        x = [[i%p] for i in range(len(list(y)))]
        model = make_pipeline(StandardScaler(), model_seasonnal)
        model.fit(x, y)
        seasonnalities[str(p)] = model.predict(x)
        pred_seasonnalities[str(p)] = model.predict([[(i+len(list(y)))%p] for i in range(N)])

        # Remove p-seasonnality from signal
        y = y-seasonnalities[str(p)]
        

    return seasonnalities, pred_seasonnalities


#################################################################################################################################################
#                                                                                                                                               #
#                                                           NORMALIZATION                                                                       #
#                                                                                                                                               #
#################################################################################################################################################    
    

def Normalize(signal, rolling_period_trend=100, rolling_period_var=100):
    """Normalize the signal"""
    if rolling_period_trend > len(signal) or rolling_period_trend == 0:
        mean = signal-signal
    else:
        mean = signal.rolling(rolling_period_trend).mean()
    demean = signal - mean
    mean = mean.fillna(0)
    demean = demean.fillna(0)

    if rolling_period_var > len(signal) or rolling_period_var == 0:
        var = signal/signal
    else:
        var = demean.rolling(rolling_period_var).std()
    normalize = demean/var
    var = var.fillna(0)
    normalize = normalize.fillna(0)


    return mean, var, normalize