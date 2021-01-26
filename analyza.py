# import libs
import importlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# load data
data = pd.read_excel('data/00-ResultsFull_JK.xlsx', sheet_name='Data')
results = pd.read_excel('data/00-ResultsFull_JK.xlsx', sheet_name='Výsledky')
data['id'] = data['TestBatchCode'] + '_' + data['TestResultPosition'] # + '_' + data['SamplingItemCode']
results['id'] = results['testbatchcode'] + '_' + results['sampleposition'] # + '_' + results['samplingitemcode']
results.set_index(['id'], drop=True, inplace=True)
data.set_index(['id'], drop=True, inplace=True)

# clean data - replace bulshits with nans
condition = (data.DeltaRn.map(type) != int) & (data.DeltaRn.map(type) != float)
data.loc[condition, 'DeltaRn'] = np.nan
# clean results of NaNs and inconclusive test
res = results.loc[results['result_value'].notna(), :]
res = res.loc[~(res['result_value'] == 'INVALID'), :]
res = res.loc[~res.duplicated(), :]

res.samplestate.unique()
res.drop(['samplestate', 'sampleposition'], axis=1, inplace=True)

# merge results with data and get rid of duplicities
df1 = data.drop(['Rn'], axis=1)
df_f = res.merge(df1, how='inner', left_index=True, right_index=True)
df_f = df_f.reset_index(drop=False).set_index(['id', 'Gen', 'Cycle'], drop=True)
df_f = df_f.loc[~df_f.index.duplicated(keep='first'), :]
df = df_f.unstack(level='Gen')
df = df['DeltaRn']
df = df.astype(float)

import functions as covid_eval
#importlib.reload(covid_eval)
test_stats = pd.DataFrame()
for i, id in enumerate(df.index.get_level_values('id').unique()):
    df_tmp = df.loc[id].interpolate(method='linear', axis=0)
    test = covid_eval.CovidAnalytics(id, df.loc[id], df.loc[id].columns)
    test.analyze_test()
    tmp = test.results
    tmp = tmp.set_index('id')
    test_stats = test_stats.append(tmp)
    if i == 0:
        test_stats['result'] = df_f.loc[id, 'result_value'][0]
    else:
        test_stats.loc[id, 'result'] = df_f.loc[id, 'result_value'][0]


test_stats = test_stats.reset_index().set_index(['gen', 'id'], drop=True)
final = test_stats.unstack(level='gen')
final.columns = final.columns.swaplevel()
final.sort_index(axis=1,inplace=True)

plt.plot(test.xdata, test.pred, label='Fitted')
plt.plot(test.xdata, test.ydata, label='Actual')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()