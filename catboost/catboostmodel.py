# %% 
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from imblearn.over_sampling import SMOTE


# %% 
train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
test = pd.read_csv('C:/Users/marco/Downloads/MJD_TEST_PROCESSED.csv')

train = train.dropna()

train_data = train.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']

test_data = test.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
test_target = test['highavse']

smote = SMOTE()
train_data, train_target = smote.fit_resample(train_data, train_target)


# %% 
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    verbose=50,
    early_stopping_rounds=50, 
    thread_count=-1  
)

model.fit(train_data, train_target, eval_set=(test_data, test_target))

# %% 
pred = model.predict(test_data)
print(f"Accuracy: {accuracy_score(test_target, pred)}")

# %% Feature Importance
features = model.get_feature_importance()
names = train_data.columns

for name, importance in zip(names, features):
    print(f"{name}: {importance}")

# %% 
train_pool = Pool(train_data, train_target)

params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss'
}

cv_results = cv(
    pool=train_pool,
    params=params,
    fold_count=5,
    verbose=False
)

print(cv_results)

# %% 
param_dist = {
    'iterations': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'depth': randint(4, 10)
}

random_search = RandomizedSearchCV(estimator=CatBoostClassifier(thread_count=-1, verbose=0), 
                                   param_distributions=param_dist, 
                                   n_iter=10,
                                   cv=3, 
                                   scoring='accuracy', 
                                   n_jobs=-1)

random_search.fit(train_data, train_target)


print("Best Parameters found by RandomizedSearchCV:")
print(random_search.best_params_)
print("Best Accuracy Score: ", random_search.best_score_)



# %%
