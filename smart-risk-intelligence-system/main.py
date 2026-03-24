from src.preprocessing import prepare_data
from src.regression import LinearRegressionScratch
from src.classification import SoftmaxRegressionScratch
from src.evaluation import mse, accuracy, confusion_matrix
from src.explainability import explain_attack

import numpy as np
import os

# ========================
# Load Data
# ========================
os.makedirs("output/graphs", exist_ok=True)
X_train, X_test, y_train, y_test, yc_train, yc_test = prepare_data()


# ========================
# Train Regression Model
# ========================
reg = LinearRegressionScratch()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("\n--- Cyber Risk Score (Regression) ---")
print("MSE:", mse(y_test, y_pred))


# ========================
# Train Classification Model
# ========================
clf = SoftmaxRegressionScratch()
clf.fit(X_train, yc_train)

yc_pred = clf.predict(X_test)

print("\n--- Attack Detection (Classification) ---")
print("Accuracy:", accuracy(yc_test, yc_pred))

cm = confusion_matrix(yc_test, yc_pred, 2)
print("Confusion Matrix:\n", cm)


# ========================
# Example Prediction
# ========================
sample = X_test[0]

risk_score = reg.predict(sample.reshape(1, -1))[0]
risk_class = clf.predict(sample.reshape(1, -1))[0]

print("\n--- Sample Analysis ---")
print(f"Risk Score: {risk_score:.2f}")

if risk_class == 0:
    print("Risk Level: NORMAL")
else:
    print("Risk Level: ATTACK")

reasons = explain_attack(sample)

print("Reasons:")
for r in reasons:
    print("-", r)

import matplotlib.pyplot as plt


# ========================
# Graph 1: Regression Loss
# ========================
plt.figure()
plt.plot(reg.losses)
plt.title("Regression Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join("output","graphs","regression_loss.png"))
plt.show()


# ========================
# Graph 2: Classification Loss
# ========================
plt.figure()
plt.plot(clf.losses)
plt.title("Classification Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join("output","graphs","classification_loss.png"))
plt.show()


# ========================
# Graph 3: Confusion Matrix
# ========================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join("output","graphs","confusion_matrix.png"))
plt.show()


# ========================
# Graph 4: Risk Score Distribution
# ========================
plt.figure()
plt.hist(y_test, bins=30)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.savefig(os.path.join("output","graphs","risk_distribution.png"))
plt.show()