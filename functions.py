# Import packages
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

class CovidAnalytics:

    def __init__(self, name, data):
        self.name = name
        self.xdata = np.array(range(1, 51))
        self.ydata = data
        self.warning = False
        self.results = pd.DataFrame(columns=['id','Slope','x_intersec','r2','L','x0','k','b','Warning'])

    def sigmoid(self, x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    def fit_sigmoid(self):
        # this is an mandatory initial guess
        p0 = [max(self.ydata), np.median(self.xdata), 1, min(self.ydata)]

        self.popt, self.pcov = curve_fit(self.sigmoid, self.xdata, self.ydata, p0, method='dogbox')
        self.pred = self.sigmoid(self.xdata, *self.popt)

    def r2(self):
        self.r2_score = r2_score(self.ydata, self.pred)

    def sigmoid_tangent_slope(self, x, L, x0, k):
        slope = (k * L * np.exp(-k * (x - x0))) / (np.exp(-k * (x - x0)) + 1) ** 2
        return slope

    def x_axis_intersection(self, slope, x, L, x0, k, b):
        y0 = self.sigmoid(x, L, x0, k, b)
        c = y0 - slope * x0
        x = -c / slope
        return x

    def analyze_test(self):
        try:
            self.fit_sigmoid()
            self.r2()
            self.slope = self.sigmoid_tangent_slope(self.popt[1],*self.popt[0:3])
            self.x_intersec = self.x_axis_intersection(self.slope,self.popt[1],*self.popt)
            self.results = self.results.append({'id': self.name,'Slope': self.slope,
                                                'x_intersec': self.x_intersec,
                                                'r2': self.r2_score, 'L': self.popt[0],
                                                'x0': self.popt[1], 'k': self.popt[2],
                                                'b': self.popt[3], 'Warning': self.warning},
                                                ignore_index=True)
        except (RuntimeError, ValueError):
            self.warning = True
            self.results = self.results.append({'id': self.name, 'Slope': np.nan,
                                                'x_intersec': np.nan,
                                                'r2': np.nan, 'L': np.nan,
                                                'x0': np.nan, 'k': np.nan,
                                                'b': np.nan, 'Warning': self.warning},
                                               ignore_index=True)