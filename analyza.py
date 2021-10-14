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

# clean data - replace nonsense with nans
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
importlib.reload(covid_eval)
test_stats = pd.DataFrame()
for i, id in enumerate(df.index.get_level_values('id').unique()):
    df_tmp = df.loc[id].interpolate(method='linear', axis=0)
    test = covid_eval.CovidAnalytics(id, df_tmp, df.loc[id].columns)
    test.analyze_test()
    tmp = test.results
    tmp = tmp.set_index('id')
    test_stats = test_stats.append(tmp)
    if i == 0:
        test_stats['result'] = df_f.loc[id, 'result_value'][0]
    else:
        test_stats.loc[id, 'result'] = df_f.loc[id, 'result_value'][0]

    test.pred.columns = 'fit ' + test.pred.columns.values
    aa = pd.concat([test.pred, test.ydata], axis=1)
    test_result1 = df_f.loc[id, 'result_value'][0]
    plt.plot(aa)
    plt.xlabel('Cycle')
    plt.legend(aa.columns.values)
    plt.title(f'{test_result1} od id: {test.name}')
    plt.savefig('graphs/'+ id + '.pdf')
    plt.show()
    plt.close()


test_stats = test_stats.reset_index().set_index(['gen', 'id'], drop=True)

final = test_stats.unstack(level='gen')
final.columns = final.columns.swaplevel()
final.sort_index(axis=1,inplace=True)

neg = final.iloc[:, final.columns.get_level_values(1)=='Slope'].le(0).sum(axis=1) > 1
negative = final.loc[neg,final.columns.get_level_values(1)=='result']
id_batches = negative.loc[negative.iloc[:,1]!='N',].index.get_level_values(0).values


id_batch = id_batches[1]
test_result1 = df_f.loc[df_f.index.get_level_values(0) == id_batch, 'result_value'][1]
df.loc[id_batch, :].plot(title=f'{test_result1} od id: {id_batch}')

test = covid_eval.CovidAnalytics(id_batch, df.loc[id_batch], df.loc[id_batch].columns)
test.analyze_test()
unc_p = np.sum((test_stats.Warning) & ((test_stats.result == 'N') | (test_stats.result == 'REV-N')))/np.sum(test_stats.Warning)

'''
test_statsW = test_stats.loc[test_stats.Warning]
np.sum((test_statsW.result == 'P') | (test_statsW.result == 'REV-P'))
id_batch = test_statsW.loc[test_statsW.result=='P'].index.get_level_values(1)[0]
id_gen = test_statsW.loc[test_statsW.result=='P'].index.get_level_values(0)[0]
'''
