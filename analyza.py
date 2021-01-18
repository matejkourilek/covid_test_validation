# import libs
import os
import pandas as pd
import numpy as np
import matplotlib

# load data
data = pd.read_excel('data/00-ResultsFull_JK.xlsx', sheet_name='Data')
results = pd.read_excel('data/00-ResultsFull_JK.xlsx', sheet_name='Výsledky')
data['id'] = data['TestBatchCode'] + '_' + data['SamplingItemCode']
results['id'] = results['testbatchcode'] + '_' + results['samplingitemcode']
results.set_index(['id'], drop=True, inplace=True)
data.set_index(['id'], drop=True, inplace=True)

# clean results with NaNs and inconclusive test
#results.result_value.unique()
res = results[results['result_value'].notna()]
res = res[~(res['result_value'] == 'INVALID')]
res = res.loc[~res.duplicated()]
res.shape
res.samplestate.unique()
res.drop(['samplestate', 'sampleposition'], axis=1, inplace=True)
df1 = data.drop(['Rn'], axis=1)
df1 = df1.loc[(df1['SamplingItemCode'] != 'unknown') & (df1['SamplingItemCode'] != 'unknown2')]

# get rid of duplicities
df_f = res.merge(df1, how='inner', left_index=True, right_index=True)
df_f.shape
df_f.head()
#df_f.dropna(subset=['TestResultPosition', 'SamplingItemCode'], axis=0, inplace=True)
#df_f['id'] = df_f['TestBatchCode'] + '_' + df_f['SamplingItemCode']# + '_' + df_f['TestResultPosition']
df_f.drop(['samplingitemcode', 'testbatchcode', 'TestBatchCode', 'SamplingItemCode', 'TestResultPosition'], axis=1, inplace=True)
df_f = df_f.reset_index(drop=False).set_index(['id', 'Gen', 'Cycle'], drop=True)
#df_f = df_f.drop(columns=['TestBatchCode', 'TestResultPosition', 'SamplingItemCode', 'Rn'])

df_f = df_f[~df_f.index.duplicated(keep='first')]
df = df_f.unstack(level='Gen')
df = df['DeltaRn']
df = df.fillna(method='ffill')

# sample 200 is negative while graph shows nice positivity. Same for sample number 500.
# MIGHT HAVE ERROR IN MERGING
id = df.index.get_level_values(0).unique()[0]
df_f.loc[df_f.index.get_level_values(0)==id,'resultlevel'].unique()
df_f.loc[df_f.index.get_level_values(0)==id,'result_value'][1]
test_result = df_f.loc[df_f.index.get_level_values(0) == id, 'result_value'][1]
df.loc[id, :].plot(title=f'{test_result} od id: {id}')
