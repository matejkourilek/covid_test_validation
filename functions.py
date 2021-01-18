# Import packages
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np


class CovidAnalytics:

    def __init__(self, name, data):
        self.name = name
        self.xdata = np.array(range(1,51))
        self.ydata = data
        self.results = pd.DataFrame()



    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)


    xdata = df.loc[id, :].index.get_level_values(0).values
    ydata = np.array(df.loc[id, 'N gene'].iloc[:, -1])
    def fit_sigmoid(self):

        # this is an mandatory initial guess
        p0 = [max(self.ydata), np.median(self.xdata), 1, min(self.ydata)]

        self.popt, self.pcov = curve_fit(sigmoid, self.xdata, self.ydata, p0, method='dogbox')

    pred = sigmoid(xdata, *popt)
    plt.plot(xdata, pred, label='Fitted')
    #plt.plot(xdata, ydata, label='Actual')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.show()
    r2_score(ydata, pred)
    print(popt[1])

    L = popt[0]
    x0 = popt[1]
    k = popt[2]
    b = popt[3]

    def r2(self):

        self.r2 = r2_score(self.data, pred)

    def sigmoid_tangent_slope(x, L, x0, k):
        slope = (k * L * np.exp(-k * (x - x0))) / (np.exp(-k * (x - x0)) + 1) ** 2
        return slope

    def x_axis_intersection(slope,x, L, x0, k, b):
        y0 = sigmoid(x, L, x0, k, b)
        c = y0 - slope * x0
        x = -c/slope
        return x
