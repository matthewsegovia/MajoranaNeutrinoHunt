# %% 
# imports
import pandas as pd
from catboost import CatBoostClassifier


train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
npml = pd.read_csv('C:/Users/marco/Downloads/MJD_NPML_PROCESSED.csv')

npml = npml.drop(['id'], axis = 1)

train_data = train.drop(['id','energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis = 1)
train_target = train['highavse']


# %%
best_model = CatBoostClassifier(
    depth=9,
    iterations=938,
    learning_rate=0.16414890742520763,
    thread_count=-1,
    verbose=50
)

best_model.fit(train_data, train_target)

npml_predictions = best_model.predict(npml)

npml_predictions_df = pd.DataFrame(npml_predictions, columns=["high_avse"])

npml_predictions_df.to_csv('npml_predictions.csv', index=False)

print("Predictions saved to 'catboost_npml_predictions.csv'")
# %%
