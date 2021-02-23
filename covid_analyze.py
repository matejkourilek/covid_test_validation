
# import libs
import matplotlib.pyplot as plt
import CovidAnalytics as covid_eval
import pandas as pd
import numpy as np
from datetime import datetime
import sys

print('Data loading...')
data = pd.read_excel(sys.argv[1], sheet_name=sys.argv[2])
#data = pd.read_excel('data/00-ResultsFull 20210217.xlsx', sheet_name='Data')
data['id'] = data['TestBatchCode'] + '_' + data['SamplingItemCode'] + '_' + data['TestResultPosition']
data.set_index(['id'], drop=True, inplace=True)

print('Data cleaning...')
# clean data - replace nonsense with nans
condition = (data.DeltaRn.map(type) != int) & (data.DeltaRn.map(type) != float)
data.loc[condition, 'DeltaRn'] = np.nan

# rename variable 'TargetName and get rid of duplicities
data.rename(columns = {'TargetName': 'Gen'}, inplace = True)
data = data[['TestBatchCode', 'TestResultPosition',   'SamplingItemCode', 'Cycle',
            'Gen','DeltaRn']]
data = data.reset_index(drop=False).set_index(['id', 'Gen', 'Cycle'], drop=True)
data = data.loc[~data.index.duplicated(keep='first'), :]
df = data.unstack(level='Gen')
df = df['DeltaRn']
df = df.astype(float)



# prepade dataframe for results
test_stats = pd.DataFrame()

# compute metrics and plot data
for i, id in enumerate(df.index.get_level_values('id').unique()):
    df_tmp = df.loc[id].interpolate(method='linear', axis=0)
    test = covid_eval.CovidAnalytics(id, df_tmp, df.loc[id].columns)
    test.analyze_test()
    tmp = test.results
    tmp = tmp.set_index('id')
    test_stats = test_stats.append(tmp)

    test.pred.columns = 'fit ' + test.pred.columns.values
    aa = pd.concat([test.pred, test.ydata], axis=1)
    plt.plot(aa)
    plt.xlabel('Cycle')
    plt.legend(aa.columns.values)
    plt.title(f' id: {test.name}')
    plt.savefig('graphs/'+ id + '.pdf')
    #plt.show()
    plt.close()


test_stats = test_stats.reset_index().set_index(['gen', 'id'], drop=True)

final = test_stats.unstack(level='gen')
final.columns = final.columns.swaplevel()
final.sort_index(axis=1,inplace=True)

# Save for Jiri
print('Saving data...')
final_tmp = final.copy()
final_tmp.columns = ['_'.join(col) for col in final_tmp.columns]
fileName = datetime.today().strftime('%Y-%m-%d') + '_analyzed_data.xlsx'
final_tmp.to_excel(fileName)
del final_tmp