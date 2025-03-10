import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import 

directory = 'result/'
files = ['MJD_TRAIN_PROCESSED.csv', 'MJD_TEST_PROCESSED.csv', 'MJD_NPML_PROCESSED.csv']
train = pd.read_csv(directory + files[0]).dropna()
test = pd.read_csv(directory + files[1]).dropna()

to_drop = ['highavse', 'lowavse', 'truedcr', 'lq', 'id']

# Splitting features and target
X_train = train.drop(columns=to_drop)
y_train = train["lowavse"]
X_test = test.drop(columns=to_drop)
y_test = test['lowavse']

smote = SMOTE()

X_train, y_train = smote.fit_resample(X_train, y_train)

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

random_search.fit(X_train, y_train, eval_metric='binary_logloss')

print("Best parameters found:", random_search.best_params_)
print("Best Accuracy score:", random_search.best_score_)

best_model = random_search.best_estimator_

import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import shap

best_model.fit(X_train, y_train)

# Feature Importances
plt.figure(figsize=(8, 6))
lgb.plot_importance(best_model, max_num_features=15)
plt.title("Feature Importances")
plt.savefig('feature_importance_lgb.png')

# ROC AUC Curve
y_pred_probs = best_model.predict_proba(X_test)[:, 1]  # Probability of positive class

fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Diagonal line for random prediction
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_auc_lgb.png')

# Confusion Matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - LightGBM")
plt.savefig('confusion_matrix_lightgbm.png')

# SHAP summary plot
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test, check_additivity=False)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('summary_plot_lgb.png')