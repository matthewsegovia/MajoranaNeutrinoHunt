# %%
# imports
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

train = pd.read_csv('/Users/marcosanchez/Downloads/MJD_TRAIN_PROCESSED.csv')
npml = pd.read_csv('/Users/marcosanchez/Downloads/MJD_NPML_PROCESSED.csv')

train_data = train.drop(['highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']




model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=10,
    verbose=50
)

model.fit(train_data, train_target)


pred = model.predict(npml)
print(pred)
