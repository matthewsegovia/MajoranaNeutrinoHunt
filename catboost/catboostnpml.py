# %% 
# imports
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
npml = pd.read_csv('C:/Users/marco/Downloads/MJD_NPML_PROCESSED.csv')

npml = npml.drop(['id'], axis = 1)

train_data = train.drop(['id','energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']


# %%
best_model = CatBoostClassifier(
    depth=9,
    iterations=825,
    learning_rate=0.16903854138336874,
    thread_count=-1,
    verbose=50
)

best_model.fit(train_data, train_target)

npml_predictions = best_model.predict(npml)

npml_predictions_df = pd.DataFrame(npml_predictions, columns=["high_avse"])

npml_predictions_df.to_csv('npml_predictions.csv', index=False)

print("Predictions saved to 'npml_predictions.csv'")
# %%
