# %%
import pandas as pd
from catboost import CatBoostClassifier


train = pd.read_csv('C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv')
npml = pd.read_csv('C:/Users/marco/Downloads/MJD_NPML_PROCESSED.csv')

train['energylabel'] = train['energylabel'].astype('float32')

npml = npml.drop(['id'], axis=1)

train_data = train.drop(['id', 'energylabel', 'highavse', 'lowavse', 'truedcr', 'lq'], axis=1)
targets = ['highavse']

npml_predictions_df = pd.DataFrame()

for target in targets:
    print(f"Training model for {target}...")
    
    train_target = train[target]
    
    model = CatBoostClassifier(
        depth=9,
        iterations=938,
        learning_rate=0.16414890742520763,
        thread_count=-1,
        verbose=50,
        task_type='GPU'
         )
    
    model.fit(train_data, train_target)
    
    npml_predictions_df[target] = model.predict(npml)

npml_predictions_df.to_csv('npml_highavse_predictions.csv', index=False)

# %%
