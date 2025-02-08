import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier

# Load original training data
train = pd.read_csv("C:/Users/marco/Downloads/MJD_TRAIN_PROCESSED.csv")
train_data = train.drop(["id", "energylabel", "highavse", "lowavse", "truedcr", "lq"], axis=1)
train_target = train["highavse"]

# Train CatBoost model
model = CatBoostClassifier(depth=9, iterations=938, learning_rate=0.164, verbose=50)
model.fit(train_data, train_target)

# Get feature importance
feature_importance = model.get_feature_importance()
features = train_data.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - CatBoost")
plt.gca().invert_yaxis()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load original test set
test = pd.read_csv("C:/Users/marco/Downloads/MJD_TEST_PROCESSED.csv")
test_data = test.drop(["id", "energylabel", "highavse", "lowavse", "truedcr", "lq"], axis=1)
test_target = test["highavse"]  # Use original test labels

# Load predictions from both models
svm_preds = np.loadtxt("svm_npml_predictions.csv", delimiter=",", skiprows=1)
catboost_preds = model.predict(test_data)  # Use test_data, NOT resampled train_data

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, preds, title in zip(axes, [svm_preds, catboost_preds], ["SVM", "CatBoost"]):
    cm = confusion_matrix(test_target, preds)  # Use original test_target
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {title}")

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc

# Load predictions as probabilities (for models that support it)
svm_probs = svm_preds  # SVM outputs hard labels; probability estimates needed if available
catboost_probs = model.predict_proba(test_data)[:, 1]  # Get probability of positive class

# Compute ROC curves
fpr_svm, tpr_svm, _ = roc_curve(test_target, svm_probs)  # Use test_target
fpr_cat, tpr_cat, _ = roc_curve(test_target, catboost_probs)

# Compute AUC scores
auc_svm = auc(fpr_svm, tpr_svm)
auc_cat = auc(fpr_cat, tpr_cat)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})", linestyle="--")
plt.plot(fpr_cat, tpr_cat, label=f"CatBoost (AUC = {auc_cat:.2f})", linestyle="-")
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
