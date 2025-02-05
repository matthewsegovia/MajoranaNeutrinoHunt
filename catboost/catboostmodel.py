# %% 
# imports
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# %% Load Data
train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
test = pd.read_csv('C:/Users/marco/Downloads/MJD_TEST_PROCESSED.csv')
npml = pd.read_csv('C:/Users/marco/Downloads/MJD_NPML_PROCESSED.csv')

train_data = train.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']

test_data = test.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
test_target = test['highavse']

# %% CatBoost Model with Early Stopping
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    verbose=50,
    early_stopping_rounds=50,  # stops if no improvement in 50 rounds
    thread_count=-1  # Utilizes all available CPUs
)

model.fit(train_data, train_target, eval_set=(test_data, test_target))

# %% Predict and Evaluate Accuracy
pred = model.predict(test_data)
print(f"Accuracy: {accuracy_score(test_target, pred)}")

# %% Feature Importance
features = model.get_feature_importance()
names = train_data.columns

for name, importance in zip(names, features):
    print(f"{name}: {importance}")

# %% Cross-validation
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

# %% Hyperparameter Tuning with RandomizedSearchCV for Speed
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

# Best parameters found by RandomizedSearchCV
print("Best Parameters found by RandomizedSearchCV:")
print(random_search.best_params_)
print("Best Accuracy Score: ", random_search.best_score_)



# %%
best_model = CatBoostClassifier(
    depth=6,
    iterations=594,
    learning_rate=0.18303897637982314,
    thread_count=-1,
    verbose=50
)

best_model.fit(train_data, train_target)

npml_predictions = best_model.predict(npml)

npml_predictions_df = pd.DataFrame(npml_predictions, columns=["high_avse"])

npml_predictions_df.to_csv('npml_predictions.csv', index=False)

print("Predictions saved to 'npml_predictions.csv'")
