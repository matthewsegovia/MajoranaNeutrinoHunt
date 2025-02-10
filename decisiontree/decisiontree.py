# %%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from scipy.stats import randint

# %% Load Data
train = pd.read_csv('/Users/ketki/Downloads/MJD_TRAIN_PROCESSED.csv')
test = pd.read_csv('/Users/ketki/Downloads/MJD_TEST_PROCESSED.csv')

train = train.dropna()  #remove missing and na values

# Features
train_data = train.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis=1)
train_target = train['highavse']

test_data = test.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis=1)
test_target = test['highavse']

# Balancing the labels
smote = SMOTE(random_state=42)
train_data, train_target = smote.fit_resample(train_data, train_target)

# %% Train Decision Tree Model
model = DecisionTreeClassifier(
    random_state=42, 
    max_depth=6, 
    min_samples_split=10, 
    min_samples_leaf=5
)

model.fit(train_data, train_target)

# %% Evaluate on Test Set
pred = model.predict(test_data)
print("Decision Tree Classifier Report:")
print(classification_report(test_target, pred))
print(f"Accuracy: {accuracy_score(test_target, pred)}")

# %% Feature Importance
features = model.feature_importances_
names = train_data.columns

print("\nFeature Importance:")
for name, importance in zip(names, features):
    print(f"{name}: {importance:.4f}")

# %% Cross-Validation
cv_scores = cross_val_score(model, train_data, train_target, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# %% Hyperparameter Tuning with RandomizedSearchCV
param_dist = {
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42), 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=3, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=2
)

random_search.fit(train_data, train_target)

print("\nBest Parameters found by RandomizedSearchCV:")
print(random_search.best_params_)
print("Best Accuracy Score: ", random_search.best_score_)
