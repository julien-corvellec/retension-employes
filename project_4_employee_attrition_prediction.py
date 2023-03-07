
"""Project 4 - Employee attrition prediction.py

# Part 1: Data preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Employee Attrition.csv')

dataset['StandardHours'].unique()

dataset['Over18'].unique()

dataset['EmployeeCount'].unique()

dataset.drop(columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], inplace=True)

dataset.select_dtypes(include=['object']).columns

len(dataset.select_dtypes(include=['object']).columns)

dataset.select_dtypes(include=['int64']).columns

len(dataset.select_dtypes(include=['int64']).columns)

if dataset.isnull().values.any():
    return "Error"

plt.figure(figsize=(16,9))
sns.heatmap(data=dataset.isnull(), cmap='coolwarm')
plt.show()

sns.countplot(dataset['Attrition'], label='Count')
plt.show()

# Attrition (Yes)
(dataset.Attrition == 'Yes').sum()

# Attrition (Yes)
(dataset.Attrition == 'No').sum()

plt.figure(figsize=[16, 9])
sns.countplot(x='Age', hue='Attrition', data=dataset)

dataset.columns

plt.figure(figsize=[20,20])

plt.subplot(311)
sns.countplot(x='Department', hue='Attrition', data=dataset)
plt.subplot(312)
sns.countplot(x='JobRole', hue='Attrition', data=dataset)
plt.subplot(313)
sns.countplot(x='JobSatisfaction', hue='Attrition', data=dataset)

"""## Correlation matrix and Heatmap"""

dataset.head()

# Create Correlation Matrix
corr = dataset.corr()

# correlation heatmap
plt.figure(figsize=(16,9))
ax = sns.heatmap(corr, annot=True, cmap = 'coolwarm')

# Set Up Mask To Hide Upper Triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# np.zeros_like() returns an array of zeros with the same shape and type as the given array
# dtype=np.bool parameter overrides the data type, so our array is an array of booleans (true / False)

# np.triu_indices_from(mask) returns the indices for the upper triangle of the array

plt.figure(figsize=(32,18))
ax = sns.heatmap(corr, mask=mask, annot=True, linewidths=0.5, cmap = 'coolwarm')

"""## Dealing with categorical data"""

# categorical columns
dataset.select_dtypes(include=['object']).columns

len(dataset.select_dtypes(include=['object']).columns)

dataset = pd.get_dummies(data=dataset, drop_first=True)

# categorical columns
dataset.select_dtypes(include=['object']).columns

"""## Split the dataset into train and test set"""

dataset.rename(columns={"Attrition_Yes": "Attrition"}, inplace=True)

# matrix of features / independent variables
x = dataset.drop(columns='Attrition')

# dependent variable
y = dataset['Attrition']

# split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""# Part 2: Building the model

## 1) Logistic regression
"""

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(x_train, y_train)

y_pred = classifier_lr.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""## 2) Random forest"""

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(x_train, y_train)

y_pred = classifier_rf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


results = results.append(model_results, ignore_index = True)

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""## 3) XGBoost"""

from xgboost import XGBClassifier
classifier_xgb = XGBClassifier(random_state=0)
classifier_xgb.fit(x_train, y_train)

y_pred = classifier_xgb.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['XGBoost', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


results = results.append(model_results, ignore_index = True)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""# Part 3: Randomized Search to find the best parameters (Logistic regression)"""

from sklearn.model_selection import RandomizedSearchCV

parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'],
              'C':[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [50, 100, 200, 500]
               }

randomized_search = RandomizedSearchCV(estimator = classifier_lr, param_distributions = parameters,
                                 n_iter = 10, scoring='roc_auc', n_jobs = -1, cv = 5, verbose=3)

# cv: cross-validation
# n_jobs = -1:
# Number of jobs to run in parallel. -1 means using all processors

randomized_search.fit(x_train, y_train)

randomized_search.best_estimator_

randomized_search.best_params_

randomized_search.best_score_

"""# Part 4: Final Model (Logistic regression)"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=1.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=500,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=0, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

final_results = pd.DataFrame([['XGBoost', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


final_results

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""## k-fold cross validation"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

"""# Part 6: Predicting a single observation"""

single_obs = [[41, 1102,	1, 2,	2,	94,	3,	2,	4,	5993,	19479,	8,	11,	3,	1,	0,	8,	0,	1,	6,	4,	0,	5, 
               0,	1,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	1]]

print(classifier.predict(sc.transform(single_obs)))
