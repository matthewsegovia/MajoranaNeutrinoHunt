# %%
import pandas as pd

high_avse = pd.read_csv("C:/Users/marco/Downloads/npml_highavse_predictions.csv")
low_avse = pd.read_csv("C:/Users/marco/Downloads/npml_low_avse.csv")
true_dcr = pd.read_csv("C:/Users/marco/Downloads/truedcr.csv")
lq = pd.read_csv("C:/Users/marco/Downloads/predicted_lq_results.csv")

high_avse = high_avse.astype(float)
lq['predicted_lq'] = lq['predicted_lq'].astype(float)

low_avse.rename(columns={'0': 'lowavse'}, inplace=True)
true_dcr.rename(columns={'0': 'truedcr'}, inplace=True)

results = pd.concat([high_avse, low_avse, true_dcr, lq], axis = 1)

results = results[['id', 'highavse', 'lowavse', 'truedcr', 'predicted_lq']]

results = results.dropna()

final = results[results.drop(columns='id').eq(1.0).all(axis=1)]

final.to_csv('npmlpredictions.csv', index=False)


# %%
