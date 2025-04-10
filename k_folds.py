import pandas as pd
import json
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
import numpy as np

# 📥 Load data
df = pd.read_csv("ml_mcq_student_scores_corrected.csv")
with open("model/used_features.json") as f:
    used_features = json.load(f)

X = df[used_features]

# 🏷️ Create labels (weak topics)
threshold = 2
y = []
for _, row in X.iterrows():
    weak_topics = [col for col in X.columns if row[col] < threshold]
    y.append(weak_topics)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# 🔄 K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

hamming_losses = []
subset_accuracies = []

print("\n🔍 Performing 5-Fold Cross Validation...\n")

for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    h_loss = hamming_loss(Y_test, Y_pred)
    acc = accuracy_score(Y_test, Y_pred)

    hamming_losses.append(h_loss)
    subset_accuracies.append(acc)

    print(f"Fold {i+1}:")
    print(f"  Subset Accuracy: {acc:.4f}")
    print(f"  Hamming Loss: {h_loss:.4f}\n")

# 🧾 Final summary
print("📊 Average Results Across 5 Folds:")
print(f"✅ Mean Subset Accuracy: {np.mean(subset_accuracies):.4f}")
print(f"✅ Mean Hamming Loss: {np.mean(hamming_losses):.4f}")
