import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, make_scorer, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

directory = 'processed/'
files = ['MJD_TRAIN_PROCESSED.csv', 'MJD_TEST_PROCESSED.csv', 'MJD_NPML_PROCESSED.csv']
train = pd.read_csv(directory + files[0]).dropna()
test = pd.read_csv(directory + files[1]).dropna()

to_drop = ['highavse', 'lowavse', 'truedcr', 'lq', 'id']

# Splitting features and target
X_train = train.drop(columns=to_drop)
y_train = train["lowavse"]
X_test = test.drop(columns=to_drop)
y_test = test['lowavse']

sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)
X_test_res, y_test_res = sm.fit_resample(X_test, y_test)

lgb_clf = lgb.LGBMClassifier()

param_test = {
    'num_leaves': sp_randint(6, 50), 
    'min_child_samples': sp_randint(100, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': sp_uniform(loc=0.2, scale=0.8), 
    'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}


random_search = RandomizedSearchCV(
    estimator=lgb_clf,
    param_distributions=param_test,
    n_iter=50, 
    scoring='accuracy',
    cv=3,
    n_jobs=1,
    verbose=100
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lgbm', random_search)
])

# Train the model on the training set
pipeline.fit(X_res, y_res)

file_path = 'npml_low_avse.csv'
np.savetxt(file_path, pipeline.predict(npml.drop('id', axis=1)), delimiter=',', fmt='%d')

print(f"Array saved to {file_path}")