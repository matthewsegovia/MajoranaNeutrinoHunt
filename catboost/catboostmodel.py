# %%
# imports
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# %%

train = pd.read_csv('/Users/marcosanchez/Downloads/MJD_TRAIN_PROCESSED.csv')
test = pd.read_csv('/Users/marcosanchez/Downloads/MJD_TEST_PROCESSED.csv')


# %%

train_data = train.drop(['id', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']

test_data = test.drop(['id', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
test_target = test['highavse']


#model training
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    verbose=50
)

# %%
model.fit(train_data, train_target)


# %%
pred = model.predict(test_data)
print(f"Accuracy: {accuracy_score(test_target, pred)}")

# %%
# finding feature importance
features = model.get_feature_importance()
names = train_data.columns

for name, importance in zip(names, features):
    print(f"{name}: {importance}")

# %%
# cross validation
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
#hyperparameter tuning

model = CatBoostClassifier(verbose=0)

param_grid = {
    'iterations': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(train_data, train_target)

print(grid_search.best_params_)
print(grid_search.best_score_)
