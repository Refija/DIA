from ensurepip import bootstrap
import pandas as pd
import glob
import os
import numpy as np
import sklearn
import shap
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree

#Load CSVfiles
files = os.path.join("C:/Users/refij/OneDrive/Desktop/Arch ML", "winequality*.csv")
files = glob.glob(files)
print("Data frame");
df = pd.concat(map(pd.read_csv, files), ignore_index=True)

#add a new column type based on the name condition, 0 for red and 1 for white
df['type'] = np.where(df['name']!= 'red', '1', '0')
df = df.drop(columns = "name")
df = df.drop(columns = "type")
#Plot correlation matrix
plt.figure(figsize=(8,6))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#print corrlation matrix
df_corr = df.corr()
print(df_corr)

#Find and Drop Hingly correlated features, Alpha = 0.5
threshold = 0.5

columns = np.full((df_corr.shape[0],), True, dtype=bool)
for i in range(df_corr.shape[0]):
    for j in range(i+1, df_corr.shape[0]):
        if df_corr.iloc[i,j] >= threshold:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
selected_columns
df = df[selected_columns]
for col_name in df.columns: 
    print(col_name)

#Train and test set
array = df.values
X = array[:,0:9]
y = array[:,9] 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

#Normalize data
norm = MinMaxScaler()
norm_fit = norm.fit(x_train)
xtrain = norm_fit.transform(x_train)
xtest = norm_fit.transform(x_test)
#Tune hyper-parameters and use GridSearch Cross validation Hyper-parameter optimization
n_estimators = [ int(x) for x in np.linspace(start=10, stop=20, num=8)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(start=10, stop=20, num=1)]
min_samples_split = [2, 3]
min_samples_leaf  = [1,2]
bootstrap = [True, False]

param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap,
     }
pprint(param_grid)

rf_regression = RandomForestRegressor()
rfR_grid = GridSearchCV(estimator = rf_regression, param_grid = param_grid,  cv = 3, verbose=2,  n_jobs = -1)
rfR_grid.fit(xtrain, y_train)
cvres = rfR_grid.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
   print(np.sqrt(-mean_score), params)
  
pprint(rfR_grid.best_params_)
# calculate mean absolute percentage error and Accuracy
best_grid = rfR_grid.best_estimator_.predict(xtrain)
errors = abs(best_grid - y_train)
mape = np.mean(100 * (errors / y_train))
accuracy = 100 - mape    
#print result
print('The best model from grid-search has an accuracy of', round(accuracy, 2),'%')
#RMSE
grid_mse = mean_squared_error(y_train, best_grid)
grid_rmse = np.sqrt(grid_mse)
print('The best model from the grid search has a RMSE of', round(grid_rmse, 2))