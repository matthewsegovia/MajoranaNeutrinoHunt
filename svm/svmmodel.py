# %% 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint

# %% 
train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
test = pd.read_csv('C:/Users/marco/Downloads/MJD_TEST_PROCESSED.csv')

train = train.dropna()
test = test.dropna()

train_data = train.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis=1)
train_target = train['highavse']

test_data = test.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis=1)
test_target = test['highavse']

smote = SMOTE() 
train_data, train_target = smote.fit_resample(train_data, train_target)

# %% 
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data) 

# %% Train initial SVM model
svm_model = SVC(kernel="rbf", C=1.0)
svm_model.fit(train_data, train_target)

svm_predictions = svm_model.predict(test_data)
svm_accuracy = accuracy_score(test_target, svm_predictions)

print(f"SVM Model Accuracy: {svm_accuracy:.4f}")

# %% 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

linear_svm_model = LinearSVC(max_iter=1000, C=1.0, random_state=42, dual=False, verbose=0)

cv_scores = cross_val_score(linear_svm_model, train_data, train_target, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")
print(f"Standard Deviation of Cross-Validation Accuracy: {cv_scores.std()}")

# %% 
param_dist = {
    'C': uniform(0.1, 10),
    'loss': ['hinge', 'squared_hinge'],
    'tol': uniform(1e-5, 1e-2)
}

random_search = RandomizedSearchCV(LinearSVC(max_iter=1000, dual=False), param_distributions=param_dist, 
                                   n_iter=10, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
random_search.fit(train_data, train_target)

print("Best Parameters found by RandomizedSearchCV:")
print(random_search.best_params_)
print("Best Accuracy Score: ", random_search.best_score_)

# %% 
param_grid = {
    'C': [0.1, 1, 10],
    'loss': ['hinge', 'squared_hinge'],
    'tol': [1e-3, 1e-4, 1e-5]
}

grid_search = GridSearchCV(LinearSVC(max_iter=1000, dual=False), param_grid, cv=cv, 
                           scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(train_data, train_target)

print("Best Parameters found by GridSearchCV:")
print(grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)
