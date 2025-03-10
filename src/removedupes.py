# %%
import pandas as pd
PROCESSED_MJD_NPML = pd.read_csv('/Users/marcosanchez/MajoranaHunt/ProcessedData/PROCESSED_MJD_NPML.csv')
PROCESSED_MJD_TEST = pd.read_csv('/Users/marcosanchez/MajoranaHunt/ProcessedData/PROCESSED_MJD_TEST.csv')
PROCESSED_MJD_TRAIN = pd.read_csv('/Users/marcosanchez/MajoranaHunt/ProcessedData/PROCESSED_MJD_TRAIN.csv')

print(PROCESSED_MJD_NPML['id'].duplicated().sum())
PROCESSED_MJD_NPML = PROCESSED_MJD_NPML.drop_duplicates(subset = ['id'])
PROCESSED_MJD_NPML.to_csv('PROCESSED_MJD_NPML.csv', index = False)

print(PROCESSED_MJD_TEST['id'].duplicated().sum())
PROCESSED_MJD_TEST = PROCESSED_MJD_TEST.drop_duplicates(subset = ['id'])
PROCESSED_MJD_TEST.to_csv('PROCESSED_MJD_TEST.csv', index = False)

print(PROCESSED_MJD_TRAIN['id'].duplicated().sum())
PROCESSED_MJD_TRAIN = PROCESSED_MJD_TRAIN.drop_duplicates(subset = ['id'])
PROCESSED_MJD_TRAIN.to_csv('PROCESSED_MJD_TRAIN.csv', index = False)
