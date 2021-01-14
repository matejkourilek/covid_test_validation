import os
import pandas as pd
import numpy as np
import matplotlib
data = pd.read_excel('data/00-ResultsFull_JK.xlsx', sheet_name='Data')
results = pd.read_excel('data/00-ResultsFull_JK.xlsx', sheet_name='Výsledky')
data['id'] = data['TestBatchCode'] + '_' + data['SamplingItemCode'] #+ '_' + data['TestResultPosition']
results['id'] = results['testbatchcode'] + '_' + results['samplingitemcode']
results.set_index(['id'],drop=True, inplace=True)
data.set_index(['id'],drop=False, inplace=True)

# clean results with NaNs and inconclusive test
results.result_value.unique()
res = results[results['result_value'].notna()]
res = res[~(res['result_value']=='INVALID')]
res.shape
res.samplestate.unique()
res.drop(['samplestate'],axis=1, inplace=True)
data.head()
res.head()
data.shape
df_f = res.merge(data, how='outer', left_index=True, right_index=True)
df_f.shape

df_f.dropna(subset=['TestResultPosition', 'SamplingItemCode'], axis=0, inplace=True)
df_f.drop(['id'], axis=1, inplace=True)
df_f.columns
df_f['id'] = df_f['TestBatchCode'] + '_' + df_f['SamplingItemCode'] + '_' + df_f['TestResultPosition']
df_f = df_f.set_index(['id', 'Gen', 'Cycle'], drop=True)
df_f = df_f.drop(columns=['TestBatchCode', 'TestResultPosition', 'SamplingItemCode', 'Rn'])
len(df_f)
df_f = df_f[~df_f.index.duplicated(keep='first')]
df = df_f.unstack(level='Gen')
df = df.droplevel(level=0, axis=1)
df = df.fillna(method='ffill')

# sample 200 is negative while graph shows nice positivity. Same for sample number 500.
# MIGHT HAVE ERROR IN MERGING
id = df.index.get_level_values(0).unique()[200]
df_f.loc[df_f.index.get_level_values(0)==id,'resultlevel'].unique()
df_f.loc[df_f.index.get_level_values(0)==id,'result_value'][1]
df.loc[id, :].plot(title=df_f.loc[df_f.index.get_level_values(0)==id,'result_value'][1])




