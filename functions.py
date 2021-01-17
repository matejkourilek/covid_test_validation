# Import packages
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


xdata = df.loc[id, :].index.get_level_values(0).values
ydata = np.array(df.loc[id, 'N gene'].iloc[:, -1])
p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')

y = sigmoid(xdata, *popt)
plt.plot(xdata, y, label='Fitted')
#plt.plot(xdata, ydata, label='Actual')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()
r2_score(ydata, y)
print(popt[1])

L = popt[0]
x0 = popt[1]
k = popt[2]
b = popt[3]


def sigmoid_tangent_slope(x, L, x0, k):
    slope = (k * L * np.exp(-k * (x - x0))) / (np.exp(-k * (x - x0)) + 1) ** 2
    return slope

def x_axis_intersection(slope,x, L, x0, k, b):
    y0 = sigmoid(x, L, x0, k, b)
    c = y0 - slope * x0
    x = -c/slope
    return x
