import csv
import ngram
import numpy
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

perfectmatch = pd.read_csv('Data/DBLP-ACM_perfectMapping.csv')

acmdata = pd.read_csv('Data/ACM.csv')
# print(acmdata)

dblp2data = pd.read_csv('Data/DBLP2.csv', encoding="ISO-8859-1")
# print(dblp2data)
## Begin of pipeline
# Check if values are missing.
# Have a look at https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b
acmdata.dropna(
    axis=0,
    how='any',  # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
# print("Missing values")
# print(acmdata)
dblp2data.dropna(
    axis=0,
    how='any',  # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
# print("Missing values")
# print(acmdata)


# Replace umlaut charaters
# Have a look at https://towardsdatascience.com/data-processing-example-using-python-bfbe6f713d9c ,
# https://www.python-lernen.de/string-replace.htm and
# https://www.designerinaction.de/tipps-tricks/web-development/html-umlaute-sonderzeichen/

dictionary = {'&#196;': 'Ä', '&#228;': 'ä', '&#203;': 'Ë', '&#235;': 'ë', '&#207;': 'Ï', '&#239;': 'ï',
              '&#214;': 'Ö', '&#246;': 'ö', '&#220;': 'Ü', '&#252;': 'ü', '&#223;': 'ß', '&#192;': 'À',
              '&#224;': 'à', '&#193;': 'Á', '&#225;': 'á', '&#194;': 'Â', '&#226;': 'â', '&#199;': 'Ç',
              '&#231;': 'ç', '&#200;': 'È', '&#232;': 'è', '&#201;': 'É', '&#234;': 'ê', '&#209;': 'Ñ',
              '&#241;': 'ñ', '&#210;': 'Ò', '&#242;': 'ò', '&#211;': 'Ó', '&#243;': 'ó', '&#212;': 'Ô',
              '&#244;': 'ô', '&#245;': 'õ', '&#195;': 'Ÿ', '&#255;': 'ÿ', '&mdash;': '—'}
acmdata.replace(dictionary, regex=True, inplace=True)
# print("replace umlaut")
# print(dataframe)
dblp2data.replace(dictionary, regex=True, inplace=True)
# print("replace umlaut")
# print(dataframe)

# Check for duplicates
# Have a look at https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/
acmdata = acmdata.drop_duplicates()
# print('drop dublicate')
# print(acmdata)
dblp2data = dblp2data.drop_duplicates()
# print('drop dublicate')
# print(dblp2data)

# Check for abbreviations and similar venue
dictionary = {'Inc\.': 'Incoperated', 'vs\.': 'versus', 'ed\.': 'edition', 'Jr\.': 'Junior', 'Corp\.': 'Corporation',
              'Oct\.': 'October', 'Univ\.': 'University', 'Dr\.': 'Doctor', 'Dept\.': 'Department',
              'Trans\.': 'Transaction', 'Syst\.': 'System', 'Vol\.': 'Volume', 'J\.': 'Journal',
              'VLDB': 'Very Large Data Bases', 'MOD': ' International Conference on Management of Data'} #DANGER!!!
acmdata.replace(dictionary, regex=True, inplace=True)
dblp2data.replace(dictionary, regex=True, inplace=True)


###Task 2
#concat 2 tables, take score table and match with the concat table, if similarity is greater than 0.5

results = pd.read_csv('result.csv')
scores = pd.read_csv('score.csv')
scores = scores.iloc[:,1]

dataset = pd.merge(results, scores, left_index=True, right_index=True)
#dataset.to_csv(r'C:\Users\refij\OneDrive\Desktop\Practical4\export_dataframe.csv', index=False, header=True)

score = []
for row in dataset['Score']:
    if row < 0.5 :    score.append('0')
    elif row >= 0.5:   score.append('1')

dataset['Score'] = score
#Transform the idDBPL attribute into number, since those are all unique string values -> into unique numbers in asc order
dataset['idDBLP'] = range(1,len(dataset)+1) 


array = dataset.values
X = array[:,0:2]
y = array[:,2] 

#Create KNN model
model = KNeighborsClassifier(n_neighbors=45)
model.fit(X, y)
predict_model = model.predict(X)
print("Accuracy score for KNN predicted")
print(accuracy_score(y, predict_model))

# split the data with 20% of the data for testing, 80% for training and validation
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)


# fit the model on one set of data
model.fit(x_train, y_train)

# evaluate the model on the second set of data
validation_model = model.predict(x_test)
print("Accuracy score for KNN validated")
print(accuracy_score(y_test, validation_model))

validation_model = model.fit(x_test, y_test).predict(x_train)
test_model = model.fit(x_train, y_train).predict(x_test)
print("Accuracy score for KNN predicted and tested")
print(accuracy_score(y_train, validation_model), accuracy_score(y_test, test_model))

#Random forest model
rF_model = RandomForestClassifier(random_state=2)
rF_model.fit(x_train, y_train)
print("Accuracy score for RandomForest")
print(rF_model.score(x_test, y_test))

param_grid = {
    'n_estimators': [5,10,15,20],
    'max_features': [1.0,'sqrt'],
    'max_depth': range(2,10,1),
    'min_samples_split': [1.0, 2],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False],
     }

rF_grid = GridSearchCV(estimator = rF_model, param_grid = param_grid, cv = 3, verbose=2,  n_jobs = -1)
rF_grid.fit(x_train, y_train)
print("RandomForest grid search - best parameters")
print(rF_grid.best_params_)

rfClass = RandomForestClassifier(max_depth = 2, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 2, n_estimators = 5)
rfClass.fit(x_train,y_train)
print("Accuracy score for RandomForest with parameters that give the best accuracy")
print(rfClass.score(x_test, y_test))
#when tune the best parameters given by grid search, the score is higher for 0.03 (3%)
