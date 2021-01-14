import os
import pandas as pd
import numpy as np

# Import the necessaries libraries
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()

data = pd.read_excel('Data/00-ResultsFull_JK.xlsx', sheet_name='Data')
results = pd.read_excel('Data/00-ResultsFull_JK.xlsx', sheet_name='Výsledky').dropna()
data['id'] = data['TestBatchCode'] + '_' + data['SamplingItemCode'] + '_' + data['TestResultPosition']
results['id'] = results['testbatchcode'] + '_' + results['samplingitemcode']

data.head()
results.head()

df = data.set_index(['id', 'Gen', 'Cycle'], drop=True)
df = df.drop(columns=['TestBatchCode', 'TestResultPosition', 'SamplingItemCode', 'Rn'])

df = df.unstack(level='Gen')
df = df.droplevel(level=0, axis=1)
df = df.fillna(method='ffill')

id = df.index.get_level_values(0).unique()[1]
df.loc[id, :].plot()
