# %%
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

train = pd.read_csv("C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv")
train_data = train.drop(["id", "energylabel", "highavse", "lowavse", "truedcr", "lq"], axis=1)
train_target = train["highavse"]

model = CatBoostClassifier(depth=9, iterations=938, learning_rate=0.164, verbose=50)
model.fit(train_data, train_target)

feature_importance = model.get_feature_importance()
features = train_data.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - CatBoost")
plt.gca().invert_yaxis()
plt.show()

test = pd.read_csv("C:/Users/marco/Downloads/MJD_TEST_PROCESSED.csv")
test_data = test.drop(["id", "energylabel", "highavse", "lowavse", "truedcr", "lq"], axis=1)
test_target = test["highavse"]

catboost_preds = model.predict(test_data)

cm = confusion_matrix(test_target, catboost_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - CatBoost")
plt.show()

catboost_probs = model.predict_proba(test_data)[:, 1]
fpr_cat, tpr_cat, _ = roc_curve(test_target, catboost_probs)
auc_cat = auc(fpr_cat, tpr_cat)

plt.figure(figsize=(8, 6))
plt.plot(fpr_cat, tpr_cat, label=f"CatBoost (AUC = {auc_cat:.2f})", linestyle="-")
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CatBoost")
plt.legend()
plt.show()

# %%
