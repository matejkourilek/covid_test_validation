# Import packages
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

class CovidAnalytics:

    def __init__(self, name, data, gens):
        self.name = name
        self.xdata = np.array(range(1, 51))
        self.ydata = data
        self.gens = [gen for gen in gens]
        self.results = pd.DataFrame(columns=['id','Slope','x_intersec','r2','L','x0','k','b','Warning', 'gen'])
        self.pred = pd.DataFrame(index=self.xdata, columns=self.gens)
        self.warning = False

    def sigmoid(self, x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    def fit_sigmoid(self, gen):
        # this is an mandatory initial guess
        p0 = [max(self.ydata[gen]), np.median(self.xdata), 1, min(self.ydata[gen])]

        popt, pcov = curve_fit(self.sigmoid, self.xdata, self.ydata[gen], p0, method='dogbox')
        self.pred[gen] = self.sigmoid(self.xdata, *popt)
        return popt, pcov

    def r2(self, gen):
        r2_s = r2_score(self.ydata[gen], self.pred[gen])
        return r2_s

    def sigmoid_tangent_slope(self, x, L, x0, k):
        slope = (k * L * np.exp(-k * (x - x0))) / (np.exp(-k * (x - x0)) + 1) ** 2
        return slope

    def x_axis_intersection(self, slope, x, L, x0, k, b):
        y0 = self.sigmoid(x, L, x0, k, b)
        c = y0 - slope * x0
        x = -c / slope
        return x

    def analyze_test(self):
        for gen in self.gens:
            try:
                f_param, cov = self.fit_sigmoid(gen)
                r2 = self.r2(gen)
                slope = self.sigmoid_tangent_slope(f_param[1], *f_param[0:3])
                x_intersec = self.x_axis_intersection(slope, f_param[1], *f_param)
                self.results = self.results.append({'id': self.name,'Slope': slope,
                                                    'x_intersec': x_intersec,
                                                    'r2': r2, 'L': f_param[0],
                                                    'x0': f_param[1], 'k': f_param[2],
                                                    'b': f_param[3], 'Warning': self.warning,
                                                    'gen': gen},
                                                    ignore_index=True)
            except (RuntimeError, ValueError):
                warning = True
                self.results = self.results.append({'id': self.name, 'Slope': np.nan,
                                                    'x_intersec': np.nan,
                                                    'r2': np.nan, 'L': np.nan,
                                                    'x0': np.nan, 'k': np.nan,
                                                    'b': np.nan, 'Warning': warning,
                                                    'gen': gen},
                                                   ignore_index=True)
