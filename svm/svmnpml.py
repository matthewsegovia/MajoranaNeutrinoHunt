# %% 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint



train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
npml = pd.read_csv('C:/Users/marco/Downloads/MJD_NPML_PROCESSED.csv')

npml = npml.drop(['id'], axis = 1)

train_data = train.drop(['id','energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']

best_model = SVC(C=10, loss='squared_hinge', tol=1e-5, max_iter=1000, dual=False)

best_model.fit(train_data, train_target)

# %% 
npml_predictions = best_model.predict(npml)

npml_predictions_df = pd.DataFrame(npml_predictions, columns=["high_avse"])

npml_predictions_df.to_csv('npml_predictions.csv', index=False)

print("Predictions saved to 'npml_predictions.csv'")