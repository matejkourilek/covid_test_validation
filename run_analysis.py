
# import libs
import matplotlib.pyplot as plt
import CovidAnalytics as covid_eval
import pandas as pd
import numpy as np


data = pd.read_excel('data/00-ResultsFull 20210217.xlsx', sheet_name='Data')
results = pd.read_excel('data/00-ResultsFull 20210217.xlsx', sheet_name='Výsledky')
data['id'] = data['TestBatchCode'] + '_' + data['SamplingItemCode'] + '_' + data['TestResultPosition']
results['id'] = results['testbatch_code'] + '_' + results['samplingsetitem_code'] + '_' + results['result_position']
results.set_index(['id'], drop=True, inplace=True)
data.set_index(['id'], drop=True, inplace=True)

# clean data - replace nonsense with nans
condition = (data.DeltaRn.map(type) != int) & (data.DeltaRn.map(type) != float)
data.loc[condition, 'DeltaRn'] = np.nan

# clean results of NaNs
res = results.loc[results['result_value'].notna(), :]
res = res.loc[~res.duplicated(), :]

# keep only useful columns
res = res[['testbatch_code','samplingsetitem_code','result_position',
           'result_value', 'result_level']]

# merge results with data and get rid of duplicities
data.rename(columns = {'TargetName': 'Gen'}, inplace = True)
df1 = data[['TestBatchCode', 'TestResultPosition',   'SamplingItemCode', 'Cycle',
            'Gen','DeltaRn']]
df_f = res.merge(df1, how='inner', left_index=True, right_index=True)
df_f = df_f.reset_index(drop=False).set_index(['id', 'Gen', 'Cycle'], drop=True)
df_f = df_f.loc[~df_f.index.duplicated(keep='first'), :]
df = df_f.unstack(level='Gen')
df = df['DeltaRn']
df = df.astype(float)

# delete no longer needed objects
del data, results

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
    if i == 0:
        test_stats['result'] = df_f.loc[id, 'result_value'][0]
    else:
        test_stats.loc[id, 'result'] = df_f.loc[id, 'result_value'][0]

    test.pred.columns = 'fit ' + test.pred.columns.values
    """
    aa = pd.concat([test.pred, test.ydata], axis=1)
    test_result1 = df_f.loc[id, 'result_value'][0]
    plt.plot(aa)
    plt.xlabel('Cycle')
    plt.legend(aa.columns.values)
    plt.title(f'{test_result1} od id: {test.name}')
    plt.savefig('graphs/'+ id + '.pdf')
    plt.show()
    plt.close()
    """

test_stats = test_stats.reset_index().set_index(['gen', 'id'], drop=True)

final = test_stats.unstack(level='gen')
final.columns = final.columns.swaplevel()
final.sort_index(axis=1,inplace=True)

# Save for Jiri
final_tmp = final.copy()
final_tmp.columns = ['_'.join(col) for col in final_tmp.columns]
final_tmp.to_excel('analyzed_data_20210217.xlsx')
del final_tmp


### Look for test with unfitted sigmoid
inv_condition = final.loc[:, pd.IndexSlice[:,'Warning']].sum(axis=1) > 0
inv_test = final.loc[inv_condition]


#invStats = inv_test.groupby(pd.IndexSlice['MS2','result']).mean()
final_clean = final.loc[~inv_condition]
inf_condition = final_clean.loc[:, pd.IndexSlice['S gene' ,'x_intersec']] == np.inf
final_clean = final_clean.loc[~inf_condition]
# Stats = final_clean.groupby(pd.IndexSlice['MS2','result']).mean()
del df, df1, df_f, df_tmp

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
X = final_clean.copy()
X.columns = ['_'.join(col) for col in X.columns]
X.drop(["MS2_result", "S gene_result", "N gene_result"], axis=1, inplace = True)

X.rename(columns={"ORF1ab_result": "result"}, inplace= True)
X.loc[X['result'] == 'REV-N','result'] = 'N'
X.loc[X['result'] == 'REV-P','result'] = 'P'
X_train, X_test, y_train, y_test = train_test_split(X.loc[:,X.columns != 'result'], X.result, test_size=0.2,random_state=109)

#from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average='macro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average='macro'))

metrics.confusion_matrix(y_test,y_pred)
fpr, tpr, thresholds = metrics.roc_curve(pd.Categorical(y_test).codes, pd.Categorical(y_pred).codes, pos_label=2)
metrics.auc(fpr, tpr)

from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
y_train_rf = pd.Categorical(y_train).codes
rf = rf.fit(X_train, y_train_rf)


#Predict the response for test dataset
y_pred_rf = rf.predict(X_test)
y_pred_rf = np.round(y_pred_rf)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(pd.Categorical(y_test).codes, y_pred_rf))

metrics.confusion_matrix(y_test,y_pred)
fpr, tpr, thresholds = metrics.roc_curve(pd.Categorical(y_test).codes, pd.Categorical(y_pred).codes, pos_label=2)
metrics.auc(fpr, tpr)