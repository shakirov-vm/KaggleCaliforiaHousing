# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

'''
:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude

An household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surpinsingly large values for block groups with few households and many empty houses, such as vacation resorts.
'''

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import seaborn as sns

train = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')

train.drop('id',axis=1,inplace=True) 
test.drop('id',axis=1,inplace=True)

print('Train size:', train.shape)
print('Test size', test.shape)

train.head()

info = pd.DataFrame(train.dtypes, columns=['dtypes'])
info['missing_values'] = train.isna().sum()
info['unique_values'] = train.nunique().values

info.style.background_gradient()

train['OccupancyPerRoom'] = np.log1p(train['AveOccup'] / train['AveRooms']);
test['OccupancyPerRoom'] = np.log1p(test['AveOccup'] / test['AveRooms']);

train['OccupancyPerBedRoom'] = np.log1p(train['AveOccup'] / train['AveBedrms']);
test['OccupancyPerBedRoom'] = np.log1p(test['AveOccup'] / test['AveBedrms']);

train['AvgRoomtoBedRoom'] = np.log1p(train['AveRooms'] / train['AveBedrms']);
test['AvgRoomtoBedRoom'] = np.log1p(test['AveRooms'] / test['AveBedrms']);

train['MedIncPop'] = np.log1p(train['MedInc'] / train['Population']);
test['MedIncPop'] = np.log1p(test['MedInc'] / test['Population']);

plt.figure(figsize=(12, 12))
sns.heatmap(train.corr(), cmap="coolwarm")
plt.title('Matrix of correlations')
plt.show()

train.drop('OccupancyPerBedRoom',axis=1,inplace=True)
test.drop('OccupancyPerBedRoom',axis=1,inplace=True)

train.drop('OccupancyPerRoom',axis=1,inplace=True)
test.drop('OccupancyPerRoom',axis=1,inplace=True)

sns.heatmap(train.corr(), cmap="coolwarm")
plt.title('Matrix of correlations')
plt.show()

'''
We have dropped the id here. We see that all features are numeric, not labels, so no encoding is needed here. The DataFrame tells us that there are no gaps in the data set. The correlation matrix tells us that almost all features are uncorrelated with each other (except MedInc and AveRooms). Gradient boosting is not sensitive to the scale of the data, so no normalization is needed. We tried to add 4 features: OccupancyPerRoom, OccupancyPerBedRoom, AvgRoomtoBedRoom, MedIncPop, but removed 2 of them, since they are correlated.
'''

def rmse(actual: np.ndarray, predicted: np.ndarray):
    return mean_squared_error(actual, predicted, squared=False)
    
model = GradientBoostingRegressor()

X = train.drop("MedHouseVal", axis = 1)
y = train["MedHouseVal"]

kf = KFold(n_splits=4, shuffle=True, random_state=0)
rmses = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmses.append(rmse(y_test, preds))

print(f'mean RMSE across all folds: {np.mean(rmses)}')

# So, let's add more features

train['AveOthRooms'] = np.log1p(train['AveRooms'] - train['AveBedrms']);
test['AveOthRooms'] = np.log1p(test['AveRooms'] - test['AveBedrms']);

train['Mult'] = train['Longitude'] * train['Latitude'];
test['Mult'] = test['Longitude'] * test['Latitude'];

sns.heatmap(train.corr(), cmap="coolwarm")
plt.title('Matrix of correlations')
plt.show()

model = GradientBoostingRegressor()

X = train.drop("MedHouseVal", axis = 1)
y = train["MedHouseVal"]

kf = KFold(n_splits=4, shuffle=True, random_state=0)
rmses = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmses.append(rmse(y_test, preds))

print(f'mean RMSE across all folds: {np.mean(rmses)}')

# As we can see, these features gave a small increase

model = GradientBoostingRegressor(learning_rate=0.2, n_estimators=300, max_depth=4)

X = train.drop("MedHouseVal", axis = 1)
y = train["MedHouseVal"]

kf = KFold(n_splits=4, shuffle=True, random_state=0)
rmses = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmses.append(rmse(y_test, preds))

print(f'mean RMSE across all folds: {np.mean(rmses)}')

# Tweaking the parameters that contribute to regularization reduces the error. Reducing the depth and increasing the number of trees reduces overfitting

# Let's try to take several regressors: the already used GradientBoostingRegressor, as well as XGBRegressor and CatBoostRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

gb = GradientBoostingRegressor(learning_rate=0.2, n_estimators=300, max_depth=4)
xgr = XGBRegressor()
catr = CatBoostRegressor()

models = [
    ("gb", gb),
    ("xgr", xgr),
    ("catr", catr),
]

ensemble = VotingRegressor(estimators=models)

kf = KFold(n_splits=4, shuffle=True, random_state=0)
rmses = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
    rmses.append(rmse(y_test, preds))

print(f'FINAL mean RMSE across all folds: {np.mean(rmses)}')

# Ensembling helped to reduce the error even more.

submission = pd.read_csv('/kaggle/input/playground-series-s3e1/sample_submission.csv')
submission['MedHouseVal'] = ensemble.predict(test)
submission.to_csv('submission.csv', index=False)
print(submission)

# Final is 0.5646516870431609