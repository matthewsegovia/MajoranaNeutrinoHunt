# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv")
train = train.dropna()
train_data = train.drop(["id", "energylabel", "highavse", "lowavse", "truedcr", "lq"], axis=1)
train_target = train["highavse"]

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

model = LinearSVC(C=10, loss="squared_hinge", tol=1e-5, max_iter=1000, dual=False)
model.fit(train_data_scaled, train_target)

feature_importance = np.abs(model.coef_).flatten()
features = train_data.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance - SVM")
plt.gca().invert_yaxis()
plt.show()

test = pd.read_csv("C:/Users/marco/Downloads/MJD_TEST_PROCESSED.csv")
test = test.dropna()
test_data = test.drop(["id", "energylabel", "highavse", "lowavse", "truedcr", "lq"], axis=1)
test_target = test["highavse"]

test_data_scaled = scaler.transform(test_data)

svm_preds = model.predict(test_data_scaled)

cm = confusion_matrix(test_target, svm_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - SVM")
plt.show()

svm_probs = model.decision_function(test_data_scaled)
fpr_svm, tpr_svm, _ = roc_curve(test_target, svm_probs)
auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})", linestyle="-")
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.show()

# %%
