# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# %%
train = pd.read_csv('/Users/marcosanchez/Downloads/MJD_TRAIN_PROCESSED.csv')
test = pd.read_csv('/Users/marcosanchez/Downloads/MJD_TEST_PROCESSED.csv')
train = train.dropna()
test = test.dropna()

# %%
train_data = train.drop(['id', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']


test_data = test.drop(['id', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
test_target = pd.DataFrame(test['highavse'])

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
# %%
svm_model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
svm_model.fit(train_data, train_target)

# %%
preds = svm_model.predict(test_data)
print("Accuracy:", accuracy_score(test_target, preds))
print("\nClassification Report:\n", classification_report(test_target, preds))
print("\nConfusion Matrix:\n", confusion_matrix(test_target, preds))


# %%
# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Grid search
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_data, train_target)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)
